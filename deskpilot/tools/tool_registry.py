from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .desktop_tools import get_active_window_title, open_path as desktop_open_path, open_url as desktop_open_url
from .execution_tools import execute_command, execute_python_code
from .file_tools import (
    apply_file_move,
    build_organization_plan,
    build_supported_file_index,
    classify_folder,
    read_document,
    scan_folder,
    resolve_document_path,
)
from .permissions import assess_command, assess_path_write
from .write_tools import write_docx, write_file, write_markdown, write_pdf, write_text
from ..rag.web_research import WebResearchAgent, WebSearchClient
from ..mcp.email_mcp import EmailMCPService
from ..multimodal import MultimodalService


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    description: str
    parameters: list[ToolParameter]
    handler: ToolHandler
    # 审批入口不进入 LLM 可见 schema，只能由 Agent 的服务端审批状态机调用。
    approval_handler: ToolHandler | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type,
                    "required": parameter.required,
                    "description": parameter.description,
                    "default": parameter.default,
                }
                for parameter in self.parameters
            ],
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """路由阶段只暴露低 token 摘要，选中工具后再读取完整 schema。"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "slots": [
                {"name": parameter.name, "type": parameter.type, "required": parameter.required}
                for parameter in self.parameters
            ],
        }


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    output: Any = None
    error: str = ""
    status: str = "success"
    error_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "output": self.output,
            "error": self.error,
            "status": self.status,
            "error_code": self.error_code,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_tools(self, category: str | None = None) -> list[dict[str, Any]]:
        tools = list(self._tools.values())
        if category:
            tools = [tool for tool in tools if tool.category == category]
        return [tool.to_dict() for tool in sorted(tools, key=lambda tool: tool.name)]

    def list_tool_summaries(self, category: str | None = None) -> list[dict[str, Any]]:
        tools = list(self._tools.values())
        if category:
            tools = [tool for tool in tools if tool.category == category]
        return [tool.to_summary_dict() for tool in sorted(tools, key=lambda tool: tool.name)]

    def get_tool_spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if not spec:
            return ToolResult(ok=False, tool_name=name, error=f"Unknown tool: {name}", status="failed", error_code="UNKNOWN_TOOL")
        try:
            # 只把工具 schema 里声明过的参数传给处理函数，避免审批回放时被额外字段打断。
            allowed_keys = {parameter.name for parameter in spec.parameters}
            filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_keys}
            output = spec.handler(**filtered_kwargs)
            status, error_code = self._output_state(output)
            return ToolResult(ok=True, tool_name=name, output=output, status=status, error_code=error_code)
        except Exception as exc:
            return ToolResult(
                ok=False, tool_name=name, error=str(exc), status="failed",
                error_code=self._exception_code(exc),
            )

    def call_approved(self, name: str, **kwargs: Any) -> ToolResult:
        """执行已经由一次性审批状态确认的调用，禁止普通路由触达此入口。"""
        spec = self._tools.get(name)
        if not spec:
            return ToolResult(ok=False, tool_name=name, error=f"Unknown tool: {name}", status="failed", error_code="UNKNOWN_TOOL")
        if spec.approval_handler is None:
            return ToolResult(ok=False, tool_name=name, error=f"Tool does not support approved execution: {name}", status="failed", error_code="APPROVAL_NOT_SUPPORTED")
        try:
            allowed_keys = {parameter.name for parameter in spec.parameters}
            filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_keys}
            output = spec.approval_handler(**filtered_kwargs)
            status, error_code = self._output_state(output)
            return ToolResult(ok=True, tool_name=name, output=output, status=status, error_code=error_code)
        except Exception as exc:
            return ToolResult(
                ok=False, tool_name=name, error=str(exc), status="failed",
                error_code=self._exception_code(exc),
            )

    @staticmethod
    def _output_state(output: Any) -> tuple[str, str]:
        if not isinstance(output, dict):
            return "success", ""
        permission = output.get("permission")
        if isinstance(permission, dict):
            if permission.get("blocked"):
                reasons = " ".join(str(item) for item in permission.get("reasons", []))
                code = "PROTECTED_PATH" if "protected" in reasons.casefold() else "POLICY_BLOCKED"
                return "blocked", code
            if permission.get("requires_confirmation") and not any(
                output.get(key) for key in ("written", "executed", "sent", "saved", "confirmed")
            ):
                return "waiting_human", "CONFIRMATION_REQUIRED"
        return "success", ""

    @staticmethod
    def _exception_code(exc: Exception) -> str:
        if isinstance(exc, FileNotFoundError):
            return "FILE_NOT_FOUND"
        if isinstance(exc, PermissionError):
            return "PERMISSION_DENIED"
        message = str(exc).casefold()
        if "protected system" in message or "受保护" in message:
            return "PROTECTED_PATH"
        if "does not exist" in message or "not found" in message or "不存在" in message:
            return "FILE_NOT_FOUND"
        if "timeout" in message or "timed out" in message or "超时" in message:
            return "TIMEOUT"
        return "TOOL_EXECUTION_ERROR"


def build_default_tool_registry(
    index,
    web_research_agent: WebResearchAgent | None = None,
    workspace_root: Path | None = None,
    email_service: EmailMCPService | None = None,
    multimodal_service: MultimodalService | None = None,
) -> ToolRegistry:
    # 这里集中注册所有工具，意图路由器和 UI 都只需要依赖这一份工具清单。
    registry = ToolRegistry()
    email = email_service or EmailMCPService()
    search_client = web_research_agent.search_client if web_research_agent else WebSearchClient()
    write_safe_roots = [workspace_root] if workspace_root is not None else None

    if multimodal_service is not None:
        registry.register(ToolSpec(
            name="vision.answer_attachments", category="vision",
            description="使用云端视觉模型回答当前会话图片；敏感图片需要人工确认。",
            parameters=[
                ToolParameter("question", "string", True, "图片问题"),
                ToolParameter("asset_ids", "array", True, "受控图片资产 ID"),
            ],
            handler=lambda question, asset_ids: {
                "permission": multimodal_service.sensitive_permission(
                    multimodal_service.sensitive_assets(list(asset_ids))
                ),
                "answered": False,
            },
            approval_handler=lambda question, asset_ids: {
                "answer": multimodal_service.answer(
                    str(question), list(asset_ids), confirm_sensitive=True
                ),
                "answered": True,
            },
        ))
        registry.register(ToolSpec(
            name="vision.inspect_image", category="vision",
            description="校验并检查本地图片，可选执行 OCR；只读源文件，结果不包含 Base64。",
            parameters=[
                ToolParameter("path", "string", True, "本地图片路径"),
                ToolParameter("include_ocr", "boolean", False, "是否执行 OCR", True),
            ],
            handler=lambda path, include_ocr=True: multimodal_service.inspect_image(
                Path(path), bool(include_ocr)
            ),
        ))
        registry.register(ToolSpec(
            name="knowledge.index_image", category="knowledge",
            description="将图片或 PDF 页图写入独立多模态索引。",
            parameters=[ToolParameter("path", "string", True, "图片或 PDF 路径")],
            handler=lambda path: multimodal_service.index_file(Path(path)),
        ))
        registry.register(ToolSpec(
            name="knowledge.search_multimodal", category="knowledge",
            description="融合 OCR、文本和视觉向量检索图片资料；只读。",
            parameters=[
                ToolParameter("query", "string", False, "文本查询", ""),
                ToolParameter("image_path", "string", False, "可选查询图片路径", ""),
                ToolParameter("top_k", "integer", False, "返回证据数", 5),
            ],
            handler=lambda query="", image_path="", top_k=5: [item.to_dict() for item in multimodal_service.search(
                str(query), Path(image_path) if image_path else None, int(top_k)
            )],
        ))
        registry.register(ToolSpec(
            name="knowledge.answer_multimodal", category="knowledge",
            description="从多模态索引检索有限图片，并使用视觉模型生成带图片来源的综合回答。",
            parameters=[
                ToolParameter("query", "string", True, "需要基于图片知识库回答的问题"),
                ToolParameter("top_k", "integer", False, "检索候选数量", 4),
            ],
            handler=lambda query, top_k=4: multimodal_service.answer_from_index(str(query), int(top_k)),
            approval_handler=lambda query, top_k=4: {
                "answered": True,
                **multimodal_service.answer_from_index(
                    str(query), int(top_k), confirm_sensitive=True
                ),
            },
        ))

    # 邮件工具统一走 MCP 业务层；发送和保存草稿由业务层返回审批信息。
    registry.register(ToolSpec("email.list_messages", "email", "List recent email messages. Read-only.", [ToolParameter("limit", "integer", False, "Maximum messages", 20), ToolParameter("unread_only", "boolean", False, "Only unread", False)], lambda limit=20, unread_only=False: email.list_messages(int(limit), bool(unread_only))))
    registry.register(ToolSpec("email.read_thread", "email", "Read an email thread by thread_id. Read-only.", [ToolParameter("thread_id", "string", True, "Thread identifier")], lambda thread_id: email.read_thread(str(thread_id))))
    registry.register(ToolSpec("email.search", "email", "Search mailbox messages. Read-only.", [ToolParameter("query", "string", True, "Search text"), ToolParameter("limit", "integer", False, "Maximum messages", 20)], lambda query, limit=20: email.search(str(query), int(limit))))
    registry.register(ToolSpec("email.classify", "email", "Classify recent messages into reply, task, notification, or other.", [ToolParameter("limit", "integer", False, "Maximum messages", 20)], lambda limit=20: email.classify(int(limit))))
    registry.register(ToolSpec("email.summarize_thread", "email", "Summarize an email thread.", [ToolParameter("thread_id", "string", True, "Thread identifier")], lambda thread_id: email.summarize(str(thread_id))))
    registry.register(ToolSpec("email.create_reply_draft", "email", "Create a reply draft without changing the mailbox.", [ToolParameter("to", "array", True, "Recipients"), ToolParameter("subject", "string", True, "Subject"), ToolParameter("body", "string", True, "Draft body"), ToolParameter("template", "string", False, "Template", "")], lambda to, subject, body, template="": email.draft(to, str(subject), str(body), str(template))))
    registry.register(ToolSpec(
        "email.save_draft",
        "email",
        "Save a draft with optional local file attachments; requires human confirmation.",
        [
            ToolParameter("to", "array", True, "Recipients"),
            ToolParameter("subject", "string", True, "Subject"),
            ToolParameter("body", "string", True, "Body"),
            ToolParameter("attachment_paths", "array", False, "Absolute paths of local attachments", []),
        ],
        lambda to, subject, body, attachment_paths=None: email.save_draft(
            to, str(subject), str(body), attachment_paths or [], False
        ),
        approval_handler=lambda to, subject, body, attachment_paths=None: email.save_draft(
            to, str(subject), str(body), attachment_paths or [], True
        ),
    ))
    registry.register(ToolSpec(
        "email.send",
        "email",
        "Send an email with optional local file attachments; requires explicit human confirmation.",
        [
            ToolParameter("to", "array", True, "Recipients"),
            ToolParameter("subject", "string", True, "Subject"),
            ToolParameter("body", "string", True, "Body"),
            ToolParameter("attachment_paths", "array", False, "Absolute paths of local attachments", []),
        ],
        lambda to, subject, body, attachment_paths=None: email.send(
            to, str(subject), str(body), attachment_paths or [], False
        ),
        approval_handler=lambda to, subject, body, attachment_paths=None: email.send(
            to, str(subject), str(body), attachment_paths or [], True
        ),
    ))

    registry.register(
        ToolSpec(
            name="web.search",
            category="web",
            description="使用当前搜索配置执行网页搜索，返回清洗后的结果列表。",
            parameters=[
                ToolParameter("query", "string", True, "搜索主题或关键词"),
                ToolParameter("limit", "integer", False, "返回结果上限", 5),
            ],
            handler=lambda query, limit=5: [
                {"title": item.title, "url": item.url, "snippet": item.snippet}
                for item in search_client.search(str(query), limit=int(limit))
            ],
        )
    )
    registry.register(
        ToolSpec(
            name="web.read_page",
            category="web",
            description="用浏览器或 HTTP 读取单个网页正文，并返回标题与文本。",
            parameters=[ToolParameter("url", "string", True, "网页 URL")],
            handler=lambda url: search_client.fetch_page(str(url)).__dict__,
        )
    )
    registry.register(
        ToolSpec(
            name="web.research",
            category="web",
            description="围绕一个主题生成网页调研报告，复用搜索、抓取、入索引和证据检索链路。",
            parameters=[
                ToolParameter("topic", "string", True, "调研主题"),
                ToolParameter("max_results", "integer", False, "搜索结果上限", 5),
            ],
            handler=lambda topic, max_results=5: web_research_agent.research(str(topic), max_results=int(max_results)).report
            if web_research_agent
            else {"error": "Web research agent is not available"},
        )
    )

    registry.register(
        ToolSpec(
            name="files.scan_folder",
            category="files",
            description="扫描文件夹内的文件并返回类别、扩展名、大小和整理建议。",
            parameters=[ToolParameter("folder", "string", True, "文件夹路径")],
            handler=lambda folder: scan_folder(Path(folder)),
        )
    )
    registry.register(
        ToolSpec(
            name="files.classify_folder",
            category="files",
            description="对文件夹进行主题分类，输出分类结果与建议归档目录。",
            parameters=[ToolParameter("folder", "string", True, "文件夹路径")],
            handler=lambda folder: classify_folder(Path(folder)),
        )
    )
    registry.register(
        ToolSpec(
            name="files.plan_organization",
            category="files",
            description="生成文件整理计划，默认只输出移动建议，不实际改动文件。",
            parameters=[
                ToolParameter("folder", "string", True, "待整理文件夹"),
                ToolParameter("target_root", "string", False, "整理目标根目录", ""),
            ],
            handler=lambda folder, target_root="": build_organization_plan(
                Path(folder), Path(target_root) if target_root else None
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.apply_move",
            category="files",
            description="准备单个文件移动；实际执行必须经过应用内人工审批。",
            parameters=[
                ToolParameter("source", "string", True, "源文件路径"),
                ToolParameter("destination", "string", True, "目标路径"),
            ],
            handler=lambda source, destination: apply_file_move(
                Path(source), Path(destination), confirm=False, safe_roots=write_safe_roots
            ),
            approval_handler=lambda source, destination: apply_file_move(
                Path(source), Path(destination), confirm=True, safe_roots=write_safe_roots
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.build_index",
            category="files",
            description="扫描文件夹并将支持的文档写入本地索引。",
            parameters=[ToolParameter("folder", "string", True, "文件夹路径")],
            handler=lambda folder: build_supported_file_index(index, Path(folder)),
        )
    )
    if index is not None:
        registry.register(
            ToolSpec(
                name="knowledge.search",
                category="knowledge",
                description=(
                    "检索已经建立的本地知识库并返回带来源的相关证据。只读。"
                    "context_expansion：事实/错误码/步骤用 sentence_window，总结/比较/跨段推理用 parent，"
                    "无需扩展用 none。"
                ),
                parameters=[
                    ToolParameter("query", "string", True, "知识库查询"),
                    ToolParameter("top_k", "integer", False, "证据数量", 5),
                    ToolParameter("scope", "string", False, "单问题用 search，跨多文档汇总用 collection", "search"),
                    ToolParameter(
                        "context_expansion", "string", False,
                        "事实、错误码和步骤使用 sentence_window；总结、比较和跨段推理使用 parent；无需扩展用 none",
                        "sentence_window",
                    ),
                ],
                handler=lambda query, top_k=5, scope="search", context_expansion="sentence_window": [
                    evidence.__dict__ for evidence in (
                        index.search_collection(str(query)) if scope == "collection" else index.search(str(query), int(top_k))
                    )
                ],
            )
        )
    registry.register(
        ToolSpec(
            name="files.read_document",
            category="files",
            description="读取并解析单个本地文档，返回结构化内容。",
            parameters=[ToolParameter("path", "string", True, "文档路径或文件名")],
            handler=lambda path: read_document(
                path,
                search_roots=[workspace_root] if workspace_root is not None else None,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.resolve_document",
            category="files",
            description="在当前工作区中定位指定文档。",
            parameters=[ToolParameter("target", "string", True, "文档路径或文件名")],
            handler=lambda target: str(
                resolve_document_path(
                    target,
                    search_roots=[workspace_root] if workspace_root is not None else None,
                )
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.write_file",
            category="files",
            description="Write a local file in text, Markdown, Word docx, or PDF format. Risky paths require app approval.",
            parameters=[
                ToolParameter("path", "string", True, "Output file path"),
                ToolParameter("content", "string", False, "File content; empty creates an empty file", ""),
                ToolParameter("file_format", "string", False, "Optional format override, e.g. md, txt, docx, pdf", ""),
                ToolParameter("overwrite", "boolean", False, "Allow replacing an existing file", False),
            ],
            handler=lambda path, content, file_format="", overwrite=False: write_file(
                path,
                str(content),
                file_format=str(file_format) if file_format else None,
                overwrite=bool(overwrite),
                confirm=False,
                safe_roots=write_safe_roots,
            ),
            approval_handler=lambda path, content, file_format="", overwrite=False: write_file(
                path, str(content), file_format=str(file_format) if file_format else None,
                overwrite=bool(overwrite), confirm=True, safe_roots=write_safe_roots,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.write_markdown",
            category="files",
            description="Write a Markdown file. Existing files require overwrite=true; risky paths require app approval.",
            parameters=[
                ToolParameter("path", "string", True, "Output Markdown path"),
                ToolParameter("content", "string", True, "Markdown content"),
                ToolParameter("overwrite", "boolean", False, "Allow replacing an existing file", False),
            ],
            handler=lambda path, content, overwrite=False: write_markdown(
                path, str(content), overwrite=bool(overwrite), confirm=False, safe_roots=write_safe_roots
            ),
            approval_handler=lambda path, content, overwrite=False: write_markdown(
                path, str(content), overwrite=bool(overwrite), confirm=True, safe_roots=write_safe_roots
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.write_text",
            category="files",
            description="Write a UTF-8 plain text file.",
            parameters=[
                ToolParameter("path", "string", True, "Output text path"),
                ToolParameter("content", "string", True, "Text content"),
                ToolParameter("overwrite", "boolean", False, "Allow replacing an existing file", False),
            ],
            handler=lambda path, content, overwrite=False: write_text(
                path, str(content), overwrite=bool(overwrite), confirm=False, safe_roots=write_safe_roots
            ),
            approval_handler=lambda path, content, overwrite=False: write_text(
                path, str(content), overwrite=bool(overwrite), confirm=True, safe_roots=write_safe_roots
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.write_docx",
            category="files",
            description="Write a minimal Word docx document.",
            parameters=[
                ToolParameter("path", "string", True, "Output docx path"),
                ToolParameter("content", "string", True, "Document content"),
                ToolParameter("overwrite", "boolean", False, "Allow replacing an existing file", False),
            ],
            handler=lambda path, content, overwrite=False: write_docx(
                path, str(content), overwrite=bool(overwrite), confirm=False, safe_roots=write_safe_roots
            ),
            approval_handler=lambda path, content, overwrite=False: write_docx(
                path, str(content), overwrite=bool(overwrite), confirm=True, safe_roots=write_safe_roots
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="files.write_pdf",
            category="files",
            description="Write a simple PDF document. Requires PyMuPDF.",
            parameters=[
                ToolParameter("path", "string", True, "Output PDF path"),
                ToolParameter("content", "string", True, "Document content"),
                ToolParameter("overwrite", "boolean", False, "Allow replacing an existing file", False),
            ],
            handler=lambda path, content, overwrite=False: write_pdf(
                path, str(content), overwrite=bool(overwrite), confirm=False, safe_roots=write_safe_roots
            ),
            approval_handler=lambda path, content, overwrite=False: write_pdf(
                path, str(content), overwrite=bool(overwrite), confirm=True, safe_roots=write_safe_roots
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="permissions.assess_path_write",
            category="permissions",
            description="Assess the risk level for a file write path without writing anything.",
            parameters=[
                ToolParameter("path", "string", True, "Target path"),
                ToolParameter("operation", "string", False, "Operation name", "write"),
            ],
            handler=lambda path, operation="write": assess_path_write(path, str(operation)).to_dict(),
        )
    )
    registry.register(
        ToolSpec(
            name="permissions.assess_command",
            category="permissions",
            description="Assess the risk level for code or shell command execution without executing anything.",
            parameters=[
                ToolParameter("command", "string|array", True, "Command text or argument list"),
                ToolParameter("operation", "string", False, "Operation name", "execute"),
            ],
            handler=lambda command, operation="execute": assess_command(command, str(operation)).to_dict(),
        )
    )
    registry.register(
        ToolSpec(
            name="code.execute_python",
            category="code",
            description="Prepare Python code execution with timeout. Execution requires app approval and policy checks.",
            parameters=[
                ToolParameter("code", "string", True, "Python code to execute"),
                ToolParameter("timeout_seconds", "integer", False, "Execution timeout", 10),
                ToolParameter("cwd", "string", False, "Working directory", ""),
            ],
            handler=lambda code, timeout_seconds=10, cwd="": execute_python_code(
                str(code),
                timeout_seconds=int(timeout_seconds),
                confirm=False,
                cwd=Path(cwd) if cwd else None,
            ),
            approval_handler=lambda code, timeout_seconds=10, cwd="": execute_python_code(
                str(code), timeout_seconds=int(timeout_seconds), confirm=True,
                cwd=Path(cwd) if cwd else None,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="shell.execute_command",
            category="shell",
            description="执行由 LLM 选择的 Windows PowerShell 或 Linux shell 命令。只读白名单命令可自动执行；写操作和未知命令需要人工确认；破坏性命令被阻断；始终受超时限制。",
            parameters=[
                ToolParameter("command", "string|array", True, "Command text or argument list"),
                ToolParameter("timeout_seconds", "integer", False, "Execution timeout", None),
                ToolParameter("cwd", "string", False, "Working directory", ""),
            ],
            handler=lambda command, timeout_seconds=None, cwd="": execute_command(
                command,
                timeout_seconds=int(timeout_seconds) if timeout_seconds is not None else None,
                confirm=False,
                cwd=Path(cwd) if cwd else None,
            ),
            approval_handler=lambda command, timeout_seconds=None, cwd="": execute_command(
                command, timeout_seconds=int(timeout_seconds) if timeout_seconds is not None else None,
                confirm=True, cwd=Path(cwd) if cwd else None,
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop.get_active_window_title",
            category="desktop",
            description="返回当前前台窗口标题，用于低风险桌面观察。",
            parameters=[],
            handler=lambda: get_active_window_title(),
        )
    )
    registry.register(
        ToolSpec(
            name="desktop.open_path",
            category="desktop",
            description="在系统默认程序中打开文件或文件夹。",
            parameters=[ToolParameter("path", "string", True, "文件或文件夹路径")],
            handler=lambda path: desktop_open_path(Path(path)),
        )
    )
    registry.register(
        ToolSpec(
            name="desktop.open_url",
            category="desktop",
            description="用系统默认浏览器打开一个 URL。",
            parameters=[ToolParameter("url", "string", True, "网页地址")],
            handler=lambda url: desktop_open_url(str(url)),
        )
    )
    if workspace_root is not None:
        registry.register(
            ToolSpec(
                name="files.workspace_root",
                category="files",
                description="返回工具当前绑定的工作区根目录。",
                parameters=[],
                handler=lambda: str(workspace_root),
            )
        )

    return registry
