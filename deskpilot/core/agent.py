from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

from .api_clients import OpenAICompatibleClient
from .approval import PendingActionStore
from .config import LOG_DIR, load_config
from .encoding_utils import fix_mojibake
from .models import AgentStep, AnswerResult, Evidence
from .json_utils import parse_json_value
from .runtime import PlanAndExecuteRuntime, PlanStep
from ..context import AssembledContext, ContextBuilder, UsageCostTracker
from ..memory.memory_compactor import MemoryCompactor
from ..memory.memory_extractor import MemoryExtractor
from ..memory.memory_store import MemoryStore
from ..memory.session_store import SessionStore
from ..memory.turn_manager import MemoryTurnManager
from ..intent.router import IntentRouter
from ..intent.schemas import IntentDecision
from ..rag.vector_index import DocumentIndex
from ..rag.query_optimizer import QueryOptimizer
from ..rag.query_analyzer import QueryAnalyzer
from ..rag.retrieval import RetrievalFilters, RetrievalRequest
from ..rag.web_research import ResearchResult, WebResearchAgent
from ..tools.tool_registry import build_default_tool_registry
from ..multi_agent.planner import PlannerAgent
from ..multi_agent.plan_executor import PlanExecutor
from ..multi_agent.router import MultiAgentRouter


class DocumentQAAgent:
    """A small observable RAG agent for stage 1.

    The agent is intentionally conservative: it only answers from indexed
    evidence. It records each step so the desktop demo can explain how the
    result was produced.
    """

    def __init__(self, index: DocumentIndex):
        self.index = index
        config = load_config()
        self.client = OpenAICompatibleClient(config)
        self.session_store = SessionStore()
        self.memory_store = MemoryStore()
        self.memory_extractor = MemoryExtractor()
        self.memory_compactor = MemoryCompactor(self.session_store)
        self.context_assembler = ContextBuilder(
            model=config.llm_model,
            tokenizer_provider=os.getenv("CONTEXT_TOKENIZER_PROVIDER", "auto"),
        )
        self.query_optimizer = QueryOptimizer()
        self.query_analyzer = QueryAnalyzer(
            llm_call=lambda prompt: self.client.chat(
                [
                    {"role": "system", "content": "你是检索查询分析器，只输出请求指定的内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
        )
        self.usage_tracker = UsageCostTracker(LOG_DIR / "context_usage.jsonl")
        self.web_research_agent = WebResearchAgent(index)
        self._operation_lock = threading.RLock()
        self._turn_usage_start = self.usage_tracker.snapshot(self._combined_usage())
        self.workspace_root = Path.cwd().resolve()
        self.tool_registry = build_default_tool_registry(index, self.web_research_agent, workspace_root=self.workspace_root)
        self.pending_actions = PendingActionStore()
        self.runtime = PlanAndExecuteRuntime(default_max_retries=1)
        self.intent_router = IntentRouter(self.client)
        # Planner/Router 负责计划与复杂度判断；已迁移的邮件复合任务由
        # PlanExecutor + Supervisor 执行，其余任务暂走兼容执行链。
        self.multi_agent_router = MultiAgentRouter(
            PlannerAgent(
                llm_call=lambda prompt: self.client.chat(
                    [{"role": "system", "content": "你是 DeskPilot 的任务规划器，只输出 JSON。"},
                     {"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            )
        )

    def plan_task(
        self, question: str, conversation_context: str = "", *,
        include_workspace_files: bool = False, include_index_catalog: bool = False,
    ) -> dict[str, object]:
        """返回多智能体计划预览，供 UI、评测和后续 Supervisor 执行使用。"""
        workspace_files = self._workspace_document_candidates(question) if include_workspace_files else []
        indexed_documents = self._index_document_catalog(question) if include_index_catalog else []
        decision = self.multi_agent_router.planner.build_plan(
            question, workspace_files=workspace_files, conversation_context=conversation_context,
            indexed_documents=indexed_documents,
        )
        # Router 只依据 Planner 产出的计划做路由，避免再次理解问题造成决策漂移。
        selected_route = "multi_agent" if decision.route == "multi_agent" or decision.complexity_score >= 7 else "single_agent"
        reason = "Planner 识别到多节点或较高复杂度任务。" if selected_route == "multi_agent" else "Planner 识别到低复杂度单节点任务。"
        valid, error = self.multi_agent_router.planner.validate(decision)
        return {
            "route": selected_route,
            "score": decision.complexity_score,
            "reason": reason,
            "valid": valid,
            "error": error,
            "steps": [
                {
                    "id": step.step_id,
                    "agent": step.agent,
                    "depends_on": step.depends_on,
                    "allowed_tools": step.allowed_tools,
                    "requires_human": step.requires_human,
                    "arguments": step.arguments,
                }
                for step in decision.steps
            ],
        }

    def _index_document_catalog(self, question: str) -> list[dict[str, str]]:
        """只传索引目录的有限元信息；完整论文内容留在检索阶段。"""
        query = question.casefold()
        docs = sorted(
            self.index.documents.values(),
            key=lambda doc: (-sum(query.count(doc.title.casefold()[i:i + 2]) for i in range(max(0, len(doc.title) - 1))), doc.title),
        )
        result: list[dict[str, str]] = []
        size = 0
        for doc in docs:
            item = {"doc_id": doc.doc_id, "title": doc.title[:120], "source_type": doc.source_type}
            cost = len(json.dumps(item, ensure_ascii=False))
            if len(result) >= 60 or size + cost > 5000:
                break
            result.append(item)
            size += cost
        return result

    def _workspace_document_candidates(self, question: str) -> list[str]:
        """仅为需要定位本地文档的计划提供有界的文件名候选。"""
        from ..context.budget import estimate_tokens
        from ..tools.file_tools import SUPPORTED_DOC_EXTENSIONS

        excluded = {".conda", ".git", ".venv", "venv", "__pycache__", "node_modules", "index", "logs", "memory"}
        files: list[str] = []
        for directory, dirs, names in os.walk(self.workspace_root):
            dirs[:] = sorted(d for d in dirs if d.casefold() not in excluded and not d.startswith("."))
            for name in sorted(names):
                if Path(name).suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
                    files.append(str((Path(directory) / name).relative_to(self.workspace_root)))
                    if len(files) >= 3000:
                        break
            if len(files) >= 3000:
                break

        query = question.casefold()
        def relevance(path: str) -> tuple[int, int, str]:
            stem = Path(path).stem.casefold()
            # 文件名的字符片段可命中“开发计划文档”等自然语言描述；长度用于同分排序。
            fragments = {stem[i:i + 2] for i in range(max(0, len(stem) - 1))}
            return (-sum(fragment in query for fragment in fragments), len(path), path)

        result: list[str] = []
        size = 0
        tokens = 0
        for path in sorted(files, key=relevance):
            encoded = json.dumps(path, ensure_ascii=False)
            length = len(encoded) + 2
            cost = estimate_tokens(encoded + ",")
            if len(result) >= 60 or size + length > 6000 or tokens + cost > 2000:
                break
            result.append(path)
            size += length
            tokens += cost
        return result

    def list_tools(self, category: str | None = None) -> list[dict[str, object]]:
        return self.tool_registry.list_tools(category=category)

    def call_tool(self, name: str, **kwargs) -> object:
        return self.tool_registry.call(name, **kwargs).to_dict()

    def answer(self, question: str, top_k: int = 5, session_id: str | None = None) -> AnswerResult:
        """串行执行一次问答，隔离共享的 Client、Index trace 和 usage 快照。"""
        with self._operation_lock:
            return self._answer_unlocked(question, top_k=top_k, session_id=session_id)

    def _answer_unlocked(self, question: str, top_k: int = 5, session_id: str | None = None) -> AnswerResult:
        if hasattr(self, "usage_tracker"):
            self._turn_usage_start = self.usage_tracker.snapshot(self._combined_usage())
        session = self.session_store.get_or_create(session_id)
        user_message = self.session_store.append_message(session.session_id, "user", question)
        memory_context = self._load_memory_context(question, session.session_id)
        steps: list[AgentStep] = [
            AgentStep("understand_question", "success", f"收到问题：{question}"),
            AgentStep(
                "load_memory_context",
                "success",
                "已加载会话记忆："
                + "；".join(memory_context.debug_lines),
            ),
        ]
        router_context = self.context_assembler.for_role(memory_context, "router", complexity=3)
        self.intent_router.client = self.client
        steps.append(AgentStep("router_context", "success", "；".join(router_context.debug_lines)))
        index_hint = self.index.retrieval_hint(question)
        steps.append(AgentStep(
            "index_route_hint", "success",
            json.dumps({"candidate_count": len(index_hint), "candidates": index_hint}, ensure_ascii=False),
        ))
        decision = self.intent_router.route(
            question, self.tool_registry, router_context.text, index_hint=index_hint
        )
        if decision.mode == "plan_task":
            planner_context = self.context_assembler.for_role(memory_context, "planner", complexity=3)
            steps.append(AgentStep("planner_context", "success", "；".join(planner_context.debug_lines)))
            plan_preview = self.plan_task(
                question, planner_context.text, include_workspace_files=decision.needs_workspace_files,
                include_index_catalog=decision.needs_index_catalog,
            )
            planned_tools = {tool for item in plan_preview.get("steps", []) if isinstance(item, dict)
                             for tool in item.get("allowed_tools", [])}
            # 路由器未预告索引目录、但 Planner 明确选择了索引检索时，才补送有界目录重规划。
            if "knowledge.search" in planned_tools and not decision.needs_index_catalog and self.index.documents:
                plan_preview = self.plan_task(
                    question, planner_context.text + "\n已选用 knowledge.search，请从索引目录挑选实际文档 ID。",
                    include_workspace_files=decision.needs_workspace_files, include_index_catalog=True,
                )
            # 仅当文件产物计划无效或缺少可执行的前置资料/写入工具时，再尝试一次语义重规划。
            # 不绕过权限检查，也不在无资料时直接写入空文件。
            wants_output = decision.requires_file_output
            if wants_output:
                actions = [tool for item in plan_preview.get("steps", []) if isinstance(item, dict)
                           for tool in item.get("allowed_tools", [])]
                has_source = bool({"web.search", "knowledge.search", "web.research"}.intersection(actions))
                has_write = "files.write_file" in actions
                if not plan_preview.get("valid") or not (has_source and has_write):
                    feedback = ("上一次计划无法执行：" + str(plan_preview.get("error") or
                                "缺少资料获取或文件写入节点") +
                                "。请重新输出完整 JSON 计划：先由 knowledge 检索资料，"
                                "再由 commit 使用 files.write_file 写入；写入节点必须标记人工确认。"
                                "如源为已索引文档使用 knowledge.search，如需联网使用 web.search。"
                                "请给出纯检索主题及用户指定的文件路径，不要虚构检索结果。")
                    repaired = self.plan_task(
                        question, planner_context.text + "\n" + feedback,
                        include_workspace_files=decision.needs_workspace_files,
                        include_index_catalog=decision.needs_index_catalog or "knowledge.search" in planned_tools,
                    )
                    repaired_actions = [tool for item in repaired.get("steps", []) if isinstance(item, dict)
                                        for tool in item.get("allowed_tools", [])]
                    if repaired.get("valid") and "files.write_file" in repaired_actions and (
                        {"web.search", "knowledge.search", "web.research"}.intersection(repaired_actions)
                    ):
                        plan_preview = repaired
                        steps.append(AgentStep("repair_plan", "success", "原计划缺少可执行依赖，已重规划检索与文件提交节点。"))
            complexity = max(1, min(int(plan_preview.get("score", 3) or 3), 10))
            memory_context = self.context_assembler.for_role(
                memory_context, "answer", complexity=complexity, planner_state=plan_preview
            )
            steps.append(AgentStep("context_policy", "success", "；".join(memory_context.debug_lines)))
            planned_targets = self._planned_document_targets(plan_preview)
            has_commit = any(
                isinstance(item, dict) and any(
                    tool in {"email.send", "email.save_draft", "files.write_file"}
                    for tool in item.get("allowed_tools", [])
                )
                for item in plan_preview.get("steps", [])
            )
            if len(planned_targets) >= 2 and not has_commit:
                return self._answer_local_documents(
                    question, planned_targets, steps, session.session_id,
                    user_message.message_id, memory_context,
                )
            # 将 Router 和 Planner 的结构化结果分别展示，便于用户审计路由原因和执行计划。
            steps.append(
                AgentStep(
                    "multi_agent_router",
                    "success" if plan_preview.get("valid") else "failed",
                    json.dumps(
                        {
                            "route": plan_preview.get("route"),
                            "complexity_score": plan_preview.get("score"),
                            "reason": plan_preview.get("reason", ""),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            steps.append(
                AgentStep(
                    "planner_agent",
                    "success" if plan_preview.get("valid") else "failed",
                    json.dumps(
                        {
                            "valid": plan_preview.get("valid"),
                            "error": plan_preview.get("error", ""),
                            "steps": plan_preview.get("steps", []),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            if not plan_preview.get("valid"):
                return self._finalize_answer(
                    answer=f"任务计划校验失败：{plan_preview.get('error', '未知原因')}",
                    evidences=[],
                    steps=steps,
                    used_llm=False,
                    session_id=session.session_id,
                    user_message_id=user_message.message_id,
                    question=question,
                    memory_context=memory_context,
                )
            simple_index_query = self._planned_simple_index_query(
                question, plan_preview, decision, index_hint
            )
            if simple_index_query is not None:
                steps.append(AgentStep(
                    "repair_plan", "success",
                    "单次索引问答不需要 Planner；已将无副作用的空计划或单检索计划降级为 knowledge.search。",
                ))
                return self._answer_rag_request(
                    question=question,
                    query=simple_index_query,
                    top_k=top_k,
                    steps=steps,
                    session_id=session.session_id,
                    user_message_id=user_message.message_id,
                    memory_context=memory_context,
                )
            web_file_request = self._planned_web_file_request(plan_preview)
            if web_file_request is not None:
                return self._answer_web_file_request(
                    question, web_file_request, steps, session.session_id,
                    user_message.message_id, memory_context,
                )
            if any(
                isinstance(item, dict)
                and any(tool in {"email.send", "email.save_draft"} for tool in item.get("allowed_tools", []))
                for item in plan_preview.get("steps", [])
            ):
                return self._answer_planned_task(
                    question=question,
                    plan_preview=plan_preview,
                    steps=steps,
                    session_id=session.session_id,
                    user_message_id=user_message.message_id,
                    memory_context=memory_context,
                )
            report_request = self._planned_index_report(plan_preview)
            if report_request is not None:
                needs_file = decision.requires_file_output
                if not needs_file and not report_request["write_report"]:
                    needs_file = self.intent_router.requires_output_file(question)
                if needs_file and not report_request["write_report"]:
                    steps.append(AgentStep("repair_plan", "success", "检索计划漏掉文件提交，按确认的文件产物需求补齐写入步骤。"))
                    report_request["write_report"] = True
                if not report_request["write_report"]:
                    return self._answer_rag_request(
                        question=question,
                        query=str(report_request.get("query") or question),
                        top_k=top_k,
                        steps=steps,
                        session_id=session.session_id,
                        user_message_id=user_message.message_id,
                        memory_context=memory_context,
                        collection=True,
                        context_expansion="parent",
                        doc_ids=[str(item) for item in report_request.get("doc_ids", [])],
                    )
                return self._answer_index_report(
                    question, report_request, steps, session.session_id, user_message.message_id, memory_context
                )
            wants_file = decision.requires_file_output
            if not wants_file:
                wants_file = self.intent_router.requires_output_file(question)
            if wants_file and decision.needs_index_catalog:
                steps.append(AgentStep("repair_plan", "success", "计划遗漏索引检索或提交节点，使用有界索引目录补选文档。"))
                return self._answer_index_report(
                    question, {"query": str(decision.arguments.get("query") or question),
                               "doc_ids": [], "path": "", "write_report": True},
                    steps, session.session_id, user_message.message_id, memory_context,
                )
            # Router 可能正确识别为复杂只读汇总，却漏填 needs_index_catalog；Planner
            # 也可能只给 communication 空节点。只要计划无副作用、没有其他工具，且
            # 路由前探测已有强索引候选，就恢复为 collection RAG。
            plan_tools = {
                str(tool)
                for item in plan_preview.get("steps", []) if isinstance(item, dict)
                for tool in item.get("allowed_tools", [])
            }
            strong_index_candidates = any(
                item.get("strong_match") is True for item in index_hint
            )
            if (
                not wants_file
                and not has_commit
                and strong_index_candidates
                and not (plan_tools - {"knowledge.search"})
            ):
                steps.append(AgentStep(
                    "repair_plan", "success",
                    "只读索引计划缺少可执行检索节点；已根据强索引候选恢复为 collection RAG。",
                ))
                return self._answer_rag_request(
                    question=question,
                    query=str(decision.arguments.get("query") or question),
                    top_k=top_k,
                    steps=steps,
                    session_id=session.session_id,
                    user_message_id=user_message.message_id,
                    memory_context=memory_context,
                    collection=True,
                    context_expansion="parent",
                )
            if wants_file or has_commit or decision.needs_index_catalog:
                return self._finalize_answer(
                    answer="任务规划缺少可执行的资料获取或提交步骤；本次没有创建文件或生成可信的索引汇总。",
                    evidences=[], steps=steps, used_llm=False,
                    session_id=session.session_id, user_message_id=user_message.message_id,
                    question=question, memory_context=memory_context,
                )
            return self._finalize_answer(
                answer="Planner 已生成计划，但当前执行器没有找到受支持的可执行节点；本次未执行外部操作。",
                evidences=[], steps=steps, used_llm=False,
                session_id=session.session_id, user_message_id=user_message.message_id,
                question=question, memory_context=memory_context,
            )
        else:
            memory_context = self.context_assembler.for_role(memory_context, "answer", complexity=2)
        return self._route_with_intent(
            question, memory_context, steps, session.session_id, user_message.message_id, decision=decision
        )

    def _answer_rag_request(
        self,
        question: str,
        query: str,
        top_k: int,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
        collection: bool = False,
        context_expansion: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> AnswerResult:
        # Router 生成的 tool query 可能为了某个对象擅自加入专属术语。P1 始终保留
        # 用户原问题，并让 Query Analyzer 只为复杂任务增加互补查询，避免单路改写覆盖原意。
        original_title_doc_ids = self.index.matching_title_doc_ids(question)
        balanced_document_search = len(original_title_doc_ids) >= 2
        complex_retrieval = balanced_document_search or collection
        index_version = f"{len(self.index.documents)}:{len(self.index.chunks)}"
        analysis = self.query_analyzer.analyze(
            question,
            recent_context=memory_context.text[-1200:],
            index_version=index_version,
            complex_task=complex_retrieval,
        )
        # Router 可复用本轮语义判断选择扩展策略；执行层只做枚举校验，不按问句关键词路由。
        expansion_strategy = str(context_expansion or analysis.context_expansion)
        if expansion_strategy not in {"sentence_window", "parent", "none"}:
            expansion_strategy = analysis.context_expansion
        rewritten = self.query_optimizer.rewrite(analysis.standalone_question)
        queries = [rewritten.rewritten]
        if analysis.needs_multi_query:
            queries.extend(analysis.sub_questions)
        queries = list(dict.fromkeys(value for value in queries if value.strip()))[:4]
        hyde_text = self.query_analyzer.generate_hyde(analysis)
        steps.append(AgentStep(
            "query_analyzer",
            "success",
            json.dumps({
                "method": analysis.method,
                "cache_hit": analysis.cache_hit,
                "task_type": analysis.task_type,
                "standalone_question": analysis.standalone_question,
                "entities": analysis.entities,
                "must_terms": analysis.must_terms,
                "queries": queries,
                "hyde_enabled": bool(hyde_text),
                "context_expansion": expansion_strategy,
            }, ensure_ascii=False),
        ))
        steps.append(AgentStep("rewrite_query", "success", f"method={rewritten.method}; query={rewritten.rewritten}"))
        # 查询明确点名多份索引文档时，为每份文档保留证据配额。比较类问题若直接做
        # 全局 Top-K，标题相近或篇幅更长的单篇文档很容易占满所有候选。
        title_doc_ids = set(doc_ids or []) or original_title_doc_ids or self.index.matching_title_doc_ids(query)
        balanced_document_search = len(title_doc_ids) >= 2
        hybrid_enabled = os.getenv("RAG_HYBRID_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        if hybrid_enabled:
            evidence_limit = max(top_k, len(title_doc_ids) * 3) if balanced_document_search else top_k
            request = RetrievalRequest(
                original_query=question,
                queries=tuple(queries),
                filters=RetrievalFilters(doc_ids=tuple(sorted(title_doc_ids))),
                hyde_text=hyde_text,
                dense_k=max(20, evidence_limit * 5),
                sparse_k=max(20, evidence_limit * 5),
                top_k=evidence_limit,
                context_expansion=expansion_strategy,
            )
            evidences = self.index.search_hybrid(request)
        elif balanced_document_search:
            evidences = self.index.search_collection(
                rewritten.rewritten, doc_ids=sorted(title_doc_ids), max_docs=len(title_doc_ids),
                chunks_per_doc=max(2, min(3, top_k)), diversify_positions=False,
            )
        elif collection:
            evidences = self.index.search_collection(
                rewritten.rewritten,
                doc_ids=sorted(title_doc_ids) if title_doc_ids else None,
            )
        else:
            evidences = self.index.search(rewritten.rewritten, top_k=top_k)
        steps.append(
            AgentStep(
                "retrieve_evidence",
                "success" if evidences else "failed",
                f"从索引中检索到 {len(evidences)} 条相关片段；"
                f"strategy={'hybrid_rrf' if hybrid_enabled else ('balanced_documents' if balanced_document_search else ('collection' if collection else 'top_k'))}；"
                f"queries={len(queries)}；hyde={bool(hyde_text)}。",
            )
        )
        if self.index.last_trace:
            trace_data = self.index.last_trace.to_dict()
            trace_events = trace_data.get("events", [])
            event_by_name = {
                str(event.get("stage")): event for event in trace_events if isinstance(event, dict)
            }
            rerank_event = event_by_name.get("rerank", {})
            rerank_detail = rerank_event.get("detail", {}) if isinstance(rerank_event, dict) else {}
            steps.append(AgentStep(
                "rerank_evidence",
                "success",
                f"requested={rerank_detail.get('requested_provider', 'disabled')}；"
                f"effective={rerank_detail.get('effective_provider', 'disabled')}；"
                f"candidates={rerank_event.get('input_count', 0)}->{rerank_event.get('output_count', 0)}；"
                f"fallback={bool(rerank_detail.get('fallback'))}。",
            ))
            expansion_event = event_by_name.get("context_expansion", {})
            expansion_detail = expansion_event.get("detail", {}) if isinstance(expansion_event, dict) else {}
            steps.append(AgentStep(
                "expand_evidence", "success",
                f"strategy={expansion_detail.get('strategy', 'none')}；"
                f"candidates={expansion_event.get('input_count', 0)}->{expansion_event.get('output_count', 0)}；"
                f"fallback_count={expansion_detail.get('fallback_count', 0)}。",
            ))
            selection_event = event_by_name.get("final_mmr", {})
            selection_detail = selection_event.get("detail", {}) if isinstance(selection_event, dict) else {}
            steps.append(AgentStep(
                "select_evidence", "success",
                f"method=coverage_aware_mmr；lambda={selection_detail.get('mmr_lambda', 'n/a')}；"
                f"candidates={selection_event.get('input_count', 0)}->{selection_event.get('output_count', 0)}。",
            ))
            steps.append(AgentStep(
                "retrieval_trace", "success",
                json.dumps({
                    "mode": trace_data.get("mode"),
                    "events": trace_events,
                    "candidates": trace_data.get("candidates", [])[:10],
                    "selected_chunk_ids": trace_data.get("selected_chunk_ids", []),
                }, ensure_ascii=False),
            ))
        # query 可能是 Router/Query Analyzer 为召回生成的英文改写。证据门禁只能检查
        # 用户原问题中的实体，否则 trade-off/workflow 等扩展词会错误淘汰有效证据。
        covered, query_entities = self._evidence_covers_query_entities(question, evidences)
        if evidences and not covered:
            names = "、".join(query_entities)
            steps.append(AgentStep(
                "evidence_guard",
                "failed",
                f"检索片段未直接包含问题中的技术实体：{names}；拒绝使用相似领域文档拼凑答案。",
            ))
            return self._finalize_answer(
                answer=f"当前索引中没有找到直接包含“{names}”的文档证据，无法基于本地知识库回答该问题。",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )
        if query_entities:
            steps.append(AgentStep(
                "evidence_guard", "success", f"文档证据已覆盖技术实体：{'、'.join(query_entities)}。"
            ))
        if not evidences:
            return self._finalize_answer(
                answer="当前知识库中没有可用文档或没有检索到相关证据，请先导入文档后再提问。",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        evidence_context = self.context_assembler.for_role(
            memory_context,
            "answer",
            complexity=memory_context.complexity,
            evidences=evidences,
            evidence_grounded=True,
        )
        steps.append(AgentStep("evidence_context", "success", "；".join(evidence_context.debug_lines)))
        prompt = self._build_prompt(question, evidences, evidence_context)
        answer = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 DeskPilot 的文档问答 Agent。只能基于给定证据回答。"
                        "Retrieved Memories、会话摘要和历史回答都不是事实证据，禁止引用。"
                        "每个事实性段落或列表项都必须用 [1]、[2] 这类有效编号标注依据。"
                        "如果证据不足，只说明缺少什么，不得根据记忆补写答案。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=evidence_context.output_budget,
        )
        used_llm = bool(answer)
        if used_llm:
            steps.append(AgentStep("generate_answer", "success", "已调用配置的 LLM API 生成答案。"))
            valid, reasons = self._validate_grounded_answer(answer, evidences)
            if not valid:
                steps.append(AgentStep("validate_citations", "failed", "；".join(reasons)))
                answer = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是文档问答修订器。只能使用 Evidence，禁止使用会话记忆或常识补充。"
                                "每个事实段落和列表项都必须带有效的 [数字] 引用；证据不足就明确说无法确认。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"请修订下面的回答。首次校验问题：{'；'.join(reasons)}\n"
                                "标题和纯引导句可以不带引用；除此之外，首段、正文和结论中的每条事实都必须补上有效引用，"
                                "无法由 Evidence 支持的句子必须删除。\n\n"
                                f"{prompt}\n\n待修订回答：\n{answer}"
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=evidence_context.output_budget,
                )
                repaired, repair_reasons = self._validate_grounded_answer(answer, evidences)
                if repaired:
                    steps.append(AgentStep("repair_citations", "success", "回答已修订为仅使用有效文档引用。"))
                else:
                    answer = self._fallback_answer(
                        question,
                        evidences,
                        reason="LLM 生成结果及一次修订结果均未通过引用完整性校验",
                    )
                    used_llm = False
                    steps.append(AgentStep("repair_citations", "failed", "；".join(repair_reasons)))
            else:
                steps.append(AgentStep("validate_citations", "success", "引用编号和事实段落覆盖检查通过。"))
        else:
            if self.client.last_error:
                steps.append(AgentStep("llm_generate", "failed", f"LLM API 调用失败：{self.client.last_error}"))
            reason = (
                f"LLM API 调用失败：{self.client.last_error}"
                if self.client.last_error
                else "当前未配置可用的 LLM API"
            )
            answer = self._fallback_answer(question, evidences, reason=reason)
            steps.append(AgentStep("generate_answer", "success", "已使用本地高相关证据摘要生成降级答案。"))
        citation_support_enabled = os.getenv("RAG_CITATION_SUPPORT_ENABLED", "false").lower() in {
            "1", "true", "yes", "on",
        }
        if citation_support_enabled and used_llm:
            supported, support_detail = self._verify_citation_support(answer, evidences)
            steps.append(AgentStep(
                "verify_citation_support",
                "success" if supported else "failed",
                support_detail,
            ))
            if not supported:
                answer = self._fallback_answer(
                    question, evidences, reason="高可靠引用支持校验未通过",
                )
                used_llm = False
        answer = self._append_source_list(answer, evidences)
        steps.append(AgentStep(
            "attach_citations",
            "success",
            f"已附加 {len(self._cited_evidence_indexes(answer, len(evidences)))} 个可核对的文件/页码来源。",
        ))
        return self._finalize_answer(
            answer=answer,
            evidences=evidences,
            steps=steps,
            used_llm=used_llm,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=evidence_context,
        )

    def _load_memory_context(self, question: str, session_id: str) -> AssembledContext:
        workspace_memory = self.memory_store.workspace_text()
        session_summary = self.session_store.read_summary(session_id)
        recent_messages = self.session_store.recent_messages(session_id, limit=13)
        # answer() 已先持久化当前问题；Builder 会将 question 作为独立 Task packet，故不在历史对话重复注入。
        if recent_messages and recent_messages[-1].role == "user" and recent_messages[-1].content.strip() == question.strip():
            recent_messages = recent_messages[:-1]
        recent_messages = recent_messages[-12:]
        retrieved_memories = self.memory_store.search(question, session_id=session_id, top_k=6)
        return self.context_assembler.assemble(
            question=question,
            workspace_memory=workspace_memory,
            session_summary=session_summary,
            recent_messages=recent_messages,
            retrieved_memories=retrieved_memories,
        )

    def _route_with_intent(
        self,
        question: str,
        memory_context: AssembledContext,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        decision: IntentDecision,
    ) -> AnswerResult:
        steps.append(
            AgentStep(
                "route_intent",
                "success",
                f"mode={decision.mode}; tool={decision.tool_name or 'none'}; reason={decision.reason}",
            )
        )

        if decision.mode == "direct_answer":
            answer = self._direct_answer(question, memory_context)
            used_llm = bool(answer)
            if not answer:
                if self.client.config.llm_api_key and self.client.last_error:
                    steps.append(AgentStep("llm_api_call", "failed", self.client.last_error))
                answer = self._direct_fallback_answer(question, memory_context)
            elif self._needs_enumeration_review(answer):
                reviewed = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是回答一致性审核器。检查总数、编号列表、分类口径和例外是否自洽；"
                                "发现冲突时直接输出修正后的完整中文回答，没有冲突则原样输出。"
                            ),
                        },
                        {"role": "user", "content": f"问题：{question}\n\n待审核回答：\n{answer}"},
                    ],
                    temperature=0.0,
                    max_tokens=memory_context.output_budget or None,
                )
                if reviewed:
                    answer = reviewed
                    steps.append(AgentStep("review_enumeration", "success", "已复核枚举数量和统计口径。"))
            steps.append(
                AgentStep(
                    "direct_answer",
                    "success",
                    "已由意图路由器判断为通用问题并直接回答。" if used_llm else "未调用 LLM API，已使用本地 fallback 生成回答。",
                )
            )
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=used_llm,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        if decision.mode == "clarify":
            answer = self._build_clarification_answer(decision)
            steps.append(AgentStep("clarify", "success", f"需要补充信息：{', '.join(decision.missing_slots) or decision.tool_name or 'unknown'}"))
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        if decision.mode == "tool_call":
            # 对邮件查询做一次用户语义校正。LLM 偶尔会漏掉“未读”槽位，
            # 但这是用户请求中的明确约束，不能因为模型漏填而退化为查询全部邮件。
            decision.arguments = self._normalize_email_arguments(question, decision.tool_name, decision.arguments)
            return self._handle_intent_tool_call(
                question=question,
                decision=decision,
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        return self._finalize_answer(
            answer="意图路由结果无法执行；本次未调用工具。",
            evidences=[], steps=steps, used_llm=False,
            session_id=session_id, user_message_id=user_message_id,
            question=question, memory_context=memory_context,
        )

    @staticmethod
    def _needs_enumeration_review(answer: str) -> bool:
        numbered = re.findall(r"(?m)^\s*\d+[.、]\s+", answer)
        total_claim = re.search(r"(?:共有|共计|总共|一共|合计).{0,10}\d+", answer)
        return len(numbered) >= 3 and bool(total_claim)

    def _handle_intent_tool_call(
        self,
        question: str,
        decision: IntentDecision,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        tool_name = decision.tool_name
        arguments = dict(decision.arguments)

        if tool_name.startswith("files.write_") or tool_name == "files.write_file":
            file_write_request = self._normalize_file_write_request(question, arguments)
            steps.append(AgentStep("route_file_write", "success", f"意图路由到文件写入：{file_write_request.get('path', '')}"))
            return self._answer_file_write(
                question=question,
                file_write_request=file_write_request,
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        if tool_name == "knowledge.search":
            return self._answer_rag_request(
                question=question,
                query=str(arguments.get("query") or question),
                top_k=int(arguments.get("top_k", 5) or 5),
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
                collection=arguments.get("scope") == "collection",
                context_expansion=str(arguments.get("context_expansion") or "") or None,
            )

        if tool_name in {"files.resolve_document", "files.read_document"}:
            local_target = str(arguments.get("target") or arguments.get("path") or "").strip()
            if not local_target:
                return self._finalize_answer(
                    answer=self._build_clarification_answer(
                        IntentDecision(
                            mode="clarify",
                            reason="缺少本地文档路径。",
                            tool_name=tool_name,
                            missing_slots=["target"],
                        )
                    ),
                    evidences=[],
                    steps=steps,
                    used_llm=False,
                    session_id=session_id,
                    user_message_id=user_message_id,
                    question=question,
                    memory_context=memory_context,
                )
            steps.append(AgentStep("route_local_document", "success", f"意图路由到本地文档：{local_target}"))
            return self._answer_local_document(
                question=question,
                local_document_target=local_target,
                read_mode=str(arguments.get("read_mode") or "").strip(),
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        if tool_name == "web.research":
            topic = str(arguments.get("topic") or arguments.get("query") or question).strip()
            steps.append(AgentStep("route_web_research", "success", f"意图路由到网页调研：{topic}"))
            return self._answer_web_research_request(
                question=question,
                topic=topic,
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        if tool_name == "web.search":
            return self._answer_web_search_request(
                question=question,
                query=str(arguments.get("query") or question).strip(),
                limit=int(arguments.get("limit", 5) or 5),
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        if tool_name in {"shell.execute_command", "code.execute_python"}:
            return self._answer_execution_from_intent(
                question=question,
                tool_name=tool_name,
                arguments=arguments,
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )

        return self._answer_generic_tool_intent(
            question=question,
            tool_name=tool_name,
            arguments=arguments,
            steps=steps,
            session_id=session_id,
            user_message_id=user_message_id,
            memory_context=memory_context,
        )

    def _evidence_covers_query_entities(
        self, query: str, evidences: list[Evidence]
    ) -> tuple[bool, list[str]]:
        """技术实体必须真实出现在 Evidence 中，领域相近或低分向量命中不算覆盖。"""
        patterns = (
            r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+",
            r"[A-Z]{2,}[A-Za-z0-9]*",
            r"[A-Za-z]+[A-Z][A-Za-z0-9]*",
        )
        entities: list[str] = []
        for pattern in patterns:
            for entity in re.findall(pattern, query):
                if entity not in entities:
                    entities.append(entity)
        if not entities:
            return True, []
        haystack = " ".join(f"{item.source_label} {item.text}" for item in evidences).casefold()
        normalized_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
        matched = [
            entity
            for entity in entities
            if re.sub(r"[^a-z0-9]+", "", entity.casefold()) in normalized_haystack
        ]
        return bool(matched), entities

    def _normalize_email_arguments(
        self, question: str, tool_name: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        """把自然语言中的邮件筛选条件补回工具参数。"""
        normalized = dict(arguments)
        if tool_name == "email.list_messages":
            text = question.casefold()
            unread_markers = (
                "未读", "未查看", "未阅读", "没读", "没有读", "unread", "unseen"
            )
            if any(marker in text for marker in unread_markers):
                normalized["unread_only"] = True
        return normalized

    def _answer_research_email_request(
        self,
        question: str,
        request: dict[str, object],
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
        plan_preview: dict[str, object] | None = None,
    ) -> AnswerResult:
        """通过统一 PlanExecutor + Supervisor 执行资料、生成和审批准备节点。"""
        effective_plan = dict(plan_preview or {})
        if not effective_plan.get("steps"):
            effective_plan["steps"] = [
                {"id": "knowledge", "agent": "knowledge", "allowed_tools": ["web.research"]},
                {"id": "communication", "agent": "communication", "depends_on": ["knowledge"]},
                {"id": "commit", "agent": "communication", "depends_on": ["communication"],
                 "allowed_tools": [str(request.get("email_tool") or "email.send")], "requires_human": True},
            ]
        steps.append(AgentStep("plan_task", "success", "统一执行器开始执行 Planner 生成的 DAG。"))
        steps.append(AgentStep("plan_executor", "success", "已将 Planner 节点绑定到领域 handler。"))
        executor = PlanExecutor(
            tool_call=self.tool_registry.call,
            research=self.web_research_agent.research,
            llm_call=lambda prompt: self.client.chat(
                [
                    {"role": "system", "content": "你是邮件助手，只根据给定资料生成邮件正文。"},
                    {"role": "user", "content": prompt},
                ], temperature=0.2,
            ),
        )
        execution = executor.execute_email_plan(effective_plan, request)
        for result in execution.supervisor.results:
            legacy_name = None
            if result.agent == "knowledge":
                legacy_name = "research_topic"
            elif result.agent == "communication" and "body" in result.output:
                legacy_name = "compose_email_body"
            if legacy_name:
                steps.append(AgentStep(legacy_name, result.status, f"由 Supervisor 节点 {result.step_id} 执行。"))
            steps.append(AgentStep(
                f"supervisor:{result.step_id}", result.status,
                result.error or f"agent={result.agent}; tool_calls={result.tool_calls}; tokens={result.tokens}",
            ))
        steps.append(AgentStep(
            "supervisor", execution.supervisor.status,
            execution.supervisor.error or f"已处理 {len(execution.supervisor.results)} 个计划节点。",
        ))
        if execution.supervisor.status == "failed":
            return self._finalize_answer(
                answer=f"无法准备邮件：{execution.supervisor.error}",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )
        if execution.tool_result is None:
            return self._finalize_answer(
                answer="Supervisor 未生成可提交的邮件操作。", evidences=[], steps=steps,
                used_llm=False, session_id=session_id, user_message_id=user_message_id,
                question=question, memory_context=memory_context,
            )
        steps.append(AgentStep(
            "request_email_confirmation", "waiting_human",
            "Supervisor 已准备邮件提交，等待人工确认。",
        ))
        return self._finalize_generic_tool_result(
            question=question,
            tool_name=execution.tool_name,
            tool_result=execution.tool_result,
            call_arguments=execution.call_arguments,
            steps=steps,
            session_id=session_id,
            user_message_id=user_message_id,
            memory_context=memory_context,
        )

    def _answer_planned_task(
        self, question: str, plan_preview: dict[str, object], steps: list[AgentStep],
        session_id: str, user_message_id: str, memory_context: AssembledContext,
    ) -> AnswerResult:
        """按 Planner 输出的节点参数执行，不再由问题文本决定是否是天气邮件任务。"""
        plan_steps = plan_preview.get("steps", [])
        email_tools = {tool for item in plan_steps if isinstance(item, dict)
                       for tool in item.get("allowed_tools", []) if tool in {"email.send", "email.save_draft"}}
        if len(email_tools) > 1:
            return self._finalize_answer(
                answer="邮件计划同时包含发送和保存草稿，无法确定提交方式；未执行邮件操作。请明确选择。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        email_tool = next(iter(email_tools)) if email_tools else "email.send"
        arguments: dict[str, object] = {}
        for item in plan_steps if isinstance(plan_steps, list) else []:
            if isinstance(item, dict) and isinstance(item.get("arguments"), dict):
                arguments.update(item["arguments"])
        raw_recipient = arguments.get("to") or arguments.get("recipient") or arguments.get("recipients") or ""
        if isinstance(raw_recipient, list):
            recipients = [str(value).strip() for value in raw_recipient if str(value).strip()]
        else:
            recipients = [value.strip() for value in re.split(r"[,;，；]", str(raw_recipient)) if value.strip()]
        # Planner 负责规划，确定性槽位解析只用于补齐其遗漏，不决定任务类型。
        if not recipients:
            recipients = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", question)
        subject = str(arguments.get("subject") or "").strip()
        if not subject:
            subject_match = re.search(r"(?:标题|主题)\s*(?:为|是|：|:)?\s*([^，,；;。]+)", question)
            subject = subject_match.group(1).strip() if subject_match else ""
        request = str(arguments.get("request") or arguments.get("query") or arguments.get("topic") or "").strip()
        if not request:
            # 仅补齐知识检索槽位，不参与路由；去掉邮件提交描述以提高搜索查询质量。
            request = re.split(r"(?:整理成|生成|写成).{0,12}邮件|并发送|发送给", question, maxsplit=1)[0].strip(" ，。")
        if not recipients or not subject or not request:
            return self._finalize_answer(
                answer="Planner 已生成多智能体计划，但计划参数不完整，未执行外部操作。请补充收件人、标题或任务内容。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        attach_report = bool(arguments.get("attach_report") or arguments.get("as_attachment"))
        attachment_paths = arguments.get("attachment_paths") or []
        if isinstance(attachment_paths, str):
            attachment_paths = [attachment_paths]
        return self._answer_research_email_request(
            question=question,
            request={
                "to": recipients[0],
                "subject": subject,
                "request": request,
                "attach_report": attach_report,
                "attachment_paths": [str(item) for item in attachment_paths if str(item).strip()],
                "email_tool": email_tool,
            },
            steps=steps, session_id=session_id, user_message_id=user_message_id,
            memory_context=memory_context, plan_preview=plan_preview,
        )

    def _build_clarification_answer(self, decision: IntentDecision) -> str:
        slots = ", ".join(decision.missing_slots) if decision.missing_slots else "需要更多信息"
        if decision.tool_name:
            return f"我可以帮你执行 `{decision.tool_name}`，但还缺少这些信息：{slots}。"
        return f"我还需要更多信息才能继续：{slots}。"

    def _normalize_file_write_request(self, question: str, arguments: dict[str, object]) -> dict[str, object]:
        request = dict(arguments)
        if not request.get("path"):
            extracted = self._extract_file_write_request(question)
            if extracted and extracted.get("path"):
                request["path"] = extracted["path"]
        content = request.get("content") or request.get("content_request") or request.get("requested_content")
        if not content:
            content = self._extract_file_write_content(question)
        if content:
            request["content"] = content
        if "overwrite" not in request:
            request["overwrite"] = self._should_overwrite_for_file_write(question)
        return request

    def _answer_web_search_request(
        self,
        question: str,
        query: str,
        limit: int,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        steps.append(AgentStep("route_web_search", "success", f"意图路由到网页搜索：{query}"))
        result = self.tool_registry.call("web.search", query=query, limit=limit)
        if not result.ok:
            steps.append(AgentStep("web_search", "failed", result.error or "网页搜索失败"))
            return self._finalize_answer(
                answer=f"网页搜索失败：{result.error or '未知错误'}",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        output = result.output or []
        lines = [f"网页搜索结果：{query}", ""]
        if isinstance(output, list):
            for idx, item in enumerate(output[:limit], start=1):
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                lines.append(f"{idx}. {title}")
                if url:
                    lines.append(f"   {url}")
                if snippet:
                    lines.append(f"   {snippet}")
        else:
            lines.append(str(output))
        steps.append(AgentStep("web_search", "success", f"已返回 {min(len(output), limit) if isinstance(output, list) else 1} 条搜索结果"))
        return self._finalize_answer(
            answer="\n".join(lines).strip(),
            evidences=[],
            steps=steps,
            used_llm=False,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
        )

    def _answer_execution_from_intent(
        self,
        question: str,
        tool_name: str,
        arguments: dict[str, object],
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        command = arguments.get("command")
        timeout_seconds = arguments.get("timeout_seconds")
        cwd = str(arguments.get("cwd", "") or "")

        if tool_name == "shell.execute_command" and not command:
            test_request = self._extract_test_execution_request(question)
            if test_request:
                steps.append(AgentStep("route_test_execution", "success", f"意图路由到测试执行：{test_request['script']}"))
                return self._answer_test_execution(
                    question=question,
                    test_execution_request=test_request,
                    steps=steps,
                    session_id=session_id,
                    user_message_id=user_message_id,
                    memory_context=memory_context,
                )
            return self._finalize_answer(
                answer=self._build_clarification_answer(
                    IntentDecision(mode="clarify", reason="缺少 command 参数。", tool_name=tool_name, missing_slots=["command"])
                ),
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        if tool_name == "code.execute_python" and not str(arguments.get("code", "")).strip():
            return self._finalize_answer(
                answer=self._build_clarification_answer(
                    IntentDecision(mode="clarify", reason="缺少 code 参数。", tool_name=tool_name, missing_slots=["code"])
                ),
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        if tool_name == "shell.execute_command":
            result = self.tool_registry.call(
                tool_name,
                command=command,
                timeout_seconds=int(timeout_seconds) if timeout_seconds is not None else 10,
                cwd=cwd,
            )
        else:
            result = self.tool_registry.call(
                tool_name,
                code=str(arguments.get("code", "")),
                timeout_seconds=int(timeout_seconds) if timeout_seconds is not None else 10,
                cwd=cwd,
            )
        steps.append(AgentStep("execute_tool", "success" if result.ok else "failed", f"已调用工具：{tool_name}"))
        return self._finalize_generic_tool_result(
            question=question,
            tool_name=tool_name,
            tool_result=result,
            call_arguments={
                "command": command,
                "code": arguments.get("code", ""),
                "timeout_seconds": timeout_seconds,
                "cwd": cwd,
            },
            steps=steps,
            session_id=session_id,
            user_message_id=user_message_id,
            memory_context=memory_context,
        )

    def _answer_generic_tool_intent(
        self,
        question: str,
        tool_name: str,
        arguments: dict[str, object],
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        # confirm 不属于模型权限。即使旧 Router 或恶意模型返回该字段，也在工具边界前丢弃。
        safe_arguments = {key: value for key, value in arguments.items() if key != "confirm"}
        result = self.tool_registry.call(tool_name, **safe_arguments)
        steps.append(AgentStep("execute_tool", "success" if result.ok else "failed", f"已调用工具：{tool_name}"))
        return self._finalize_generic_tool_result(
            question=question,
            tool_name=tool_name,
            tool_result=result,
            call_arguments=safe_arguments,
            steps=steps,
            session_id=session_id,
            user_message_id=user_message_id,
            memory_context=memory_context,
        )

    def _finalize_generic_tool_result(
        self,
        question: str,
        tool_name: str,
        tool_result: object,
        call_arguments: dict[str, object] | None,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        ok = bool(getattr(tool_result, "ok", False))
        error = str(getattr(tool_result, "error", "") or "工具执行失败")
        # 失败时不能把 output 伪装成空字典，否则用户会误以为工具成功但没有数据。
        if not ok:
            steps.append(AgentStep("tool_result", "failed", error))
            return self._finalize_answer(
                answer=f"工具执行失败：{tool_name}\n\n原因：{error}",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        payload = getattr(tool_result, "output", None)
        answer = ""
        pending_action = None
        if isinstance(payload, dict) and payload.get("permission"):
            permission = payload.get("permission", {})
            if isinstance(permission, dict) and permission.get("requires_confirmation") and not permission.get("blocked"):
                attachments = payload.get("attachments", [])
                attachment_lines = []
                if isinstance(attachments, list):
                    attachment_lines = [
                        f"{item.get('name', '')} ({item.get('size', 0)} bytes)\n{item.get('path', '')}"
                        for item in attachments
                        if isinstance(item, dict)
                    ]
                description = f"执行工具：{tool_name}"
                if attachment_lines:
                    description += "\n附件：\n" + "\n".join(attachment_lines)
                pending_action = {
                    "tool_name": tool_name,
                    "kwargs": dict(call_arguments or {}),
                    "description": description,
                    "risk_level": permission.get("risk_level", "high"),
                    "reasons": permission.get("reasons", []),
                }
        if isinstance(payload, list):
            lines: list[str] = []
            for idx, item in enumerate(payload[:5], start=1):
                if not isinstance(item, dict):
                    lines.append(f"{idx}. {item}")
                    continue
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                lines.append(f"{idx}. {title}")
                if url:
                    lines.append(f"   {url}")
                if snippet:
                    lines.append(f"   {snippet}")
            answer = "\n".join(lines)
        elif isinstance(payload, dict):
            if tool_name == "email.list_messages" and isinstance(payload.get("messages"), list):
                # 邮件列表只展示用户真正关心的标题和正文，隐藏内部 UID、线程和协议字段。
                message_lines: list[str] = [f"最近找到 {payload.get('count', len(payload['messages']))} 封邮件："]
                for index, item in enumerate(payload["messages"], start=1):
                    if not isinstance(item, dict):
                        continue
                    subject = str(item.get("subject", "（无标题）")).strip() or "（无标题）"
                    body = str(item.get("body", "")).strip() or "（无正文）"
                    if len(body) > 1200:
                        body = body[:1200].rstrip() + "\n[正文预览已截断，可继续要求读取该邮件全文]"
                    message_lines.extend([f"\n{index}. 标题：{subject}", f"   内容：{body}"])
                answer = "\n".join(message_lines)
            elif tool_name in {"shell.execute_command", "code.execute_python"} and payload.get("executed"):
                answer = self._present_execution_result(question, payload)
            else:
                answer = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            answer = str(payload)

        steps.append(AgentStep("tool_result", "success", f"工具输出已返回：{tool_name}"))
        # 工具结果以 packet 进入 Executor 视图，避免后续步骤依赖未受预算控制的原始对象。
        memory_context = self.context_assembler.for_role(
            memory_context,
            "executor",
            complexity=memory_context.complexity,
            tool_outputs=[{"tool": tool_name, "ok": ok, "output": payload}],
        )
        steps.append(AgentStep("tool_context", "success", "；".join(memory_context.debug_lines)))

        return self._finalize_answer(
            answer=answer or f"工具已执行：{tool_name}",
            evidences=[],
            steps=steps,
            used_llm=False,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
            pending_action=pending_action,
        )

    def _present_execution_result(self, question: str, output: dict[str, object]) -> str:
        """将命令结果作为数据呈现；常见标量查询不向用户暴露内部 JSON。"""
        stdout = self._clip_tool_output(str(output.get("stdout", "")))
        stderr = self._clip_tool_output(str(output.get("stderr", "")))
        returncode = output.get("returncode")
        if returncode == 0 and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stdout.strip()) and re.search(
            r"(?:多少|几个|数量|计数|count)", question, re.IGNORECASE,
        ):
            return f"查询完成，结果是 **{stdout.strip()}**。"
        lines = [f"命令执行完成，退出码：{returncode}。"]
        if stdout:
            lines.extend(["", "stdout：", stdout])
        if stderr:
            lines.extend(["", "stderr：", stderr])
        if not stdout and not stderr:
            lines.extend(["", "命令没有输出 stdout/stderr。"])
        return "\n".join(lines)

    def _finalize_answer(
        self,
        answer: str,
        evidences: list[Evidence],
        steps: list[AgentStep],
        used_llm: bool,
        session_id: str,
        user_message_id: str,
        question: str,
        memory_context: AssembledContext,
        pending_action: dict | None = None,
    ) -> AnswerResult:
        answer = fix_mojibake(answer)
        if (
            hasattr(self, "usage_tracker")
            and hasattr(self, "_turn_usage_start")
        ):
            task_type = self._task_type_from_steps(steps)
            usage = self.usage_tracker.record_delta(
                self._turn_usage_start,
                self._combined_usage(),
                task_type,
                str(getattr(self.client.config, "llm_model", "unknown")),
            )
            steps.append(AgentStep(
                "context_cost",
                "success",
                (
                    f"task_type={task_type}; prompt={usage['prompt_tokens']}; completion={usage['completion_tokens']}; "
                    f"total={usage['total_tokens']}; samples={usage['samples']}; "
                    f"p50={usage['p50_total_tokens']}; p95={usage['p95_total_tokens']}"
                ),
            ))
        if pending_action:
            pending_action = self.pending_actions.create(session_id, pending_action)
        assistant_message = self.session_store.append_message(session_id, "assistant", answer)
        memory_result = MemoryTurnManager(
            self.memory_extractor, self.memory_store, self.session_store, self.memory_compactor,
        ).process(
            user_message=question, assistant_message=answer, session_id=session_id,
            source_message_ids=[user_message_id, assistant_message.message_id],
            grounded=bool(evidences), steps=steps, pending_action=pending_action,
        )
        steps.append(
            AgentStep(
                "extract_memory",
                "skipped" if memory_result.skipped else "success",
                (f"记忆 Gate 跳过本轮抽取：{memory_result.reason}。" if memory_result.skipped else
                 f"本轮抽取并保存 {memory_result.extracted} 条结构化记忆；gate={memory_result.reason}。"),
            )
        )
        steps.append(
            AgentStep(
                "compact_memory",
                "success",
                "会话达到阈值，已更新滚动摘要。" if memory_result.compacted else "当前会话未达到压缩阈值，暂不更新摘要。",
            )
        )
        # 将本轮可观测信息绑定到对应 assistant 消息。这样切换历史会话或导出时
        # 仍能恢复 Steps/Evidence，而不是只在当前进程的内存面板中短暂存在。
        self.session_store.update_message_metadata(
            session_id,
            assistant_message.message_id,
            {
                "steps": [
                    {"name": step.name, "status": step.status, "detail": step.detail[:12000]}
                    for step in steps
                ],
                "evidences": [
                    {
                        "source": item.source_label,
                        "score": float(item.score),
                        "text": item.text[:4000],
                    }
                    for item in evidences[:12]
                ],
            },
        )
        return AnswerResult(
            answer=answer,
            evidences=evidences,
            steps=steps,
            used_llm=used_llm,
            session_id=session_id,
            memory_context=memory_context.text,
            pending_action=pending_action,
        )

    def _task_type_from_steps(self, steps: list[AgentStep]) -> str:
        names = {step.name for step in steps}
        if "retrieve_evidence" in names:
            return "rag"
        if "route_web_research" in names or "web_search" in names:
            return "web"
        if "execute_tool" in names or "tool_result" in names:
            return "tool"
        return "direct"

    def _combined_usage(self) -> dict[str, int]:
        """汇总本轮可能参与的 LLM、Embedding、Memory 和 Web Research 客户端 usage。"""
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}
        clients = [
            getattr(self, "client", None),
            getattr(getattr(self, "index", None), "client", None),
            getattr(getattr(self, "memory_store", None), "client", None),
            getattr(getattr(self, "web_research_agent", None), "client", None),
        ]
        seen: set[int] = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            usage = getattr(client, "token_usage", {})
            for key in totals:
                totals[key] += int(usage.get(key, 0))
        return totals

    def approve_pending_action(self, pending_action: dict, session_id: str | None = None) -> AnswerResult:
        with self._operation_lock:
            return self._approve_pending_action_unlocked(pending_action, session_id=session_id)

    def _approve_pending_action_unlocked(
        self, pending_action: dict, session_id: str | None = None,
    ) -> AnswerResult:
        session = self.session_store.get_or_create(session_id)
        action_id = str(pending_action.get("action_id", ""))
        approved = self.pending_actions.consume(action_id, session.session_id)
        question = f"用户确认执行工具：{approved.tool_name}"
        user_message = self.session_store.append_message(session.session_id, "user", question)
        memory_context = self._load_memory_context(question, session.session_id)
        steps = [
            AgentStep("approve_pending_action", "success", f"用户已确认：{approved.description}"),
        ]
        tool_name = approved.tool_name
        kwargs = dict(approved.kwargs)
        result = self.tool_registry.call_approved(tool_name, **kwargs)
        if not result.ok:
            steps.append(AgentStep("execute_approved_action", "failed", result.error or "工具执行失败"))
            answer = f"已获得确认，但工具执行失败：{tool_name}\n\n原因：{result.error or '未知错误'}"
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session.session_id,
                user_message_id=user_message.message_id,
                question=question,
                memory_context=memory_context,
            )

        output = result.output or {}
        email_success = tool_name == "email.save_draft" and output.get("status") == "draft_saved"
        send_success = tool_name == "email.send" and output.get("status") == "sent"
        execution_success = bool(output.get("executed"))
        move_success = tool_name == "files.apply_move" and bool(output.get("confirmed"))
        if output.get("written") or email_success or send_success or execution_success or move_success:
            path = str(output.get("path", kwargs.get("path", "")))
            size = output.get("size", 0)
            steps.append(AgentStep("execute_approved_action", "success", f"已执行工具：{tool_name}"))
            if email_success:
                answer = f"已确认并保存邮件草稿，草稿箱：{output.get('mailbox', 'Drafts')}。"
            elif send_success:
                copy_status = "已同步到已发送文件夹。" if output.get("sent_copy_saved") else "邮件已发送，但已发送副本未能同步。"
                attachment_paths = output.get("attachment_paths", [])
                attachment_status = ""
                if isinstance(attachment_paths, list) and attachment_paths:
                    names = "、".join(Path(str(path)).name for path in attachment_paths)
                    attachment_status = f"\n\n已附加文件：{names}"
                answer = f"已确认并发送邮件。\n\n{copy_status}{attachment_status}"
            elif execution_success:
                answer = json.dumps({
                    "returncode": output.get("returncode"),
                    "timed_out": output.get("timed_out", False),
                    "duration_seconds": output.get("duration_seconds", 0),
                    "stdout": self._clip_tool_output(str(output.get("stdout", ""))),
                    "stderr": self._clip_tool_output(str(output.get("stderr", ""))),
                }, ensure_ascii=False, indent=2)
            elif move_success:
                answer = f"已确认并移动文件：\n{output.get('source', '')}\n-> {output.get('destination', '')}"
            else:
                answer = f"已确认并完成文件写入：{path}\n\n文件大小：{size} bytes。"
        else:
            steps.append(AgentStep("execute_approved_action", "failed", str(output.get("message", "工具未执行"))))
            answer = f"已获得确认，但工具没有完成执行：{tool_name}\n\n原因：{output.get('message', '未知原因')}"

        self.session_store.append_message(
            session.session_id,
            "tool",
            json.dumps({
                "tool_name": tool_name,
                "argument_names": sorted(kwargs),
                "result": {
                    "status": output.get("status", ""),
                    "path": output.get("path", ""),
                    "returncode": output.get("returncode"),
                    "timed_out": output.get("timed_out", False),
                    "written": output.get("written", False),
                    "executed": output.get("executed", False),
                },
            }, ensure_ascii=False),
            metadata={"tool_name": tool_name, "approved": True},
        )
        return self._finalize_answer(
            answer=answer,
            evidences=[],
            steps=steps,
            used_llm=False,
            session_id=session.session_id,
            user_message_id=user_message.message_id,
            question=question,
            memory_context=memory_context,
        )

    def cancel_pending_action(self, pending_action: dict, session_id: str | None = None) -> bool:
        """取消一次性审批请求，避免同一 action_id 之后被重放。"""
        session = self.session_store.get_or_create(session_id)
        return self.pending_actions.cancel(str(pending_action.get("action_id", "")), session.session_id)

    def _extract_test_execution_request(self, question: str) -> dict[str, object] | None:
        normalized = question.lower()
        execution_markers = ("执行", "运行", "跑一下", "跑一遍", "测试一下", "执行测试", "运行测试")
        if not any(marker in normalized for marker in execution_markers):
            return None
        match = re.search(r"([^\s，。；;、\"'`]+?\.py)", question, flags=re.IGNORECASE)
        if not match:
            return None
        script = self._normalize_execution_target(match.group(1))
        if not script:
            return None
        return {
            "script": script,
            "timeout_seconds": 120 if "test" in script.lower() or "测试" in normalized else 60,
        }

    def _normalize_execution_target(self, raw_target: str) -> str:
        target = str(raw_target or "").strip().strip('"').strip("'").strip("`“”‘’")
        target = target.strip("，。；;、,. ")
        prefixes = (
            "执行",
            "运行",
            "跑一下",
            "跑一遍",
            "测试一下",
            "执行测试",
            "运行测试",
            "当前目录下的",
            "当前目录下",
            "本目录下的",
            "本目录下",
        )
        changed = True
        while changed and target:
            changed = False
            for prefix in prefixes:
                if target.startswith(prefix) and len(target) > len(prefix):
                    target = target[len(prefix) :].strip()
                    changed = True
                    break
        return target.strip("，。；;、,. ")

    def _answer_test_execution(
        self,
        question: str,
        test_execution_request: dict[str, object],
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        script = str(test_execution_request["script"])
        timeout_seconds = int(test_execution_request.get("timeout_seconds", 120))
        script_path = (self.workspace_root / script).resolve(strict=False)
        steps.append(AgentStep("route_test_execution", "success", f"Detected Python test execution request: {script}"))

        def validate_target(values: dict[str, object]) -> dict[str, object]:
            if not self._is_path_inside_workspace(script_path) or script_path.suffix.lower() != ".py":
                raise ValueError(f"Only Python test files inside the workspace can be executed automatically: {script_path}")
            if not script_path.exists():
                raise FileNotFoundError(f"Test script does not exist: {script_path}")
            return {"script_path": script_path}

        def execute_test(values: dict[str, object]) -> dict[str, object]:
            result = self.tool_registry.call(
                "shell.execute_command",
                command=[sys.executable, str(script_path)],
                timeout_seconds=timeout_seconds,
                cwd=str(self.workspace_root),
            )
            if not result.ok:
                raise RuntimeError(result.error or "shell.execute_command failed")
            return {"execution_result": result}

        execution = self.runtime.run(
            [
                PlanStep("validate_test_target", validate_target, "Validated test path and extension.", max_retries=0),
                PlanStep("prepare_test", execute_test, "Prepared test command through the shell tool.", max_retries=1),
            ],
            {"script": script, "timeout_seconds": timeout_seconds},
        )
        steps.extend(execution.agent_steps)
        result = execution.values.get("execution_result")
        if execution.failed or result is None:
            answer = f"未执行测试：{script_path}\n\n原因：{execution.failure or 'plan did not produce an execution result'}"
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        output = result.output or {}
        permission = output.get("permission", {}) if isinstance(output, dict) else {}
        if isinstance(permission, dict) and permission.get("requires_confirmation") and not output.get("executed"):
            steps.append(AgentStep("request_test_confirmation", "waiting_human", "测试命令已准备，等待人工确认。"))
            return self._finalize_generic_tool_result(
                question=question,
                tool_name="shell.execute_command",
                tool_result=result,
                call_arguments={
                    "command": [sys.executable, str(script_path)],
                    "timeout_seconds": timeout_seconds,
                    "cwd": str(self.workspace_root),
                },
                steps=steps,
                session_id=session_id,
                user_message_id=user_message_id,
                memory_context=memory_context,
            )
        executed = bool(output.get("executed"))
        returncode = output.get("returncode")
        timed_out = bool(output.get("timed_out"))
        stdout = str(output.get("stdout", ""))
        stderr = str(output.get("stderr", ""))
        duration = output.get("duration_seconds", 0)
        status = "failed" if timed_out or returncode not in (0, None) else "success"
        steps.append(
            AgentStep(
                "summarize_test_result",
                status,
                f"executed={executed}, returncode={returncode}, duration={duration}s",
            )
        )
        tool_summary = self._format_test_execution_answer(
            script_path=script_path,
            executed=executed,
            timed_out=timed_out,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
        )
        self.session_store.append_message(
            session_id,
            "tool",
            tool_summary,
            metadata={
                "tool_name": "shell.execute_command",
                "script": str(script_path),
                "returncode": returncode,
                "timed_out": timed_out,
                "duration_seconds": duration,
            },
        )
        return self._finalize_answer(
            answer=tool_summary,
            evidences=[],
            steps=steps,
            used_llm=False,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
        )

    def _is_path_inside_workspace(self, path: Path) -> bool:
        try:
            path.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def _format_test_execution_answer(
        self,
        script_path: Path,
        executed: bool,
        timed_out: bool,
        returncode: object,
        stdout: str,
        stderr: str,
        duration: object,
    ) -> str:
        if not executed:
            return f"测试未执行：{script_path}"
        result_text = "通过" if returncode == 0 and not timed_out else "失败"
        lines = [
            f"测试执行结果：{result_text}",
            "",
            f"- 脚本：{script_path}",
            f"- returncode：{returncode}",
            f"- timed_out：{timed_out}",
            f"- duration_seconds：{duration}",
        ]
        if stdout.strip():
            lines.extend(["", "stdout：", self._clip_tool_output(stdout)])
        if stderr.strip():
            lines.extend(["", "stderr：", self._clip_tool_output(stderr)])
        if not stdout.strip() and not stderr.strip():
            lines.extend(["", "该测试没有输出 stdout/stderr。"])
        return "\n".join(lines)

    def _clip_tool_output(self, text: str, limit: int = 4000) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "\n... [output truncated]"

    def _planned_document_targets(self, plan_preview: dict[str, object]) -> list[str]:
        """从 Planner 结构化参数读取多个文档，不维护业务文档别名表。"""
        targets: list[str] = []
        for item in plan_preview.get("steps", []) if isinstance(plan_preview.get("steps"), list) else []:
            if not isinstance(item, dict):
                continue
            args = item.get("arguments", {})
            if not isinstance(args, dict):
                continue
            values = args.get("paths") or args.get("files") or args.get("documents") or []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                targets.extend(str(value).strip() for value in values if str(value).strip())
        return list(dict.fromkeys(targets))

    @staticmethod
    def _planned_simple_index_query(
        question: str,
        plan_preview: dict[str, object],
        decision: IntentDecision,
        index_hint: list[dict[str, object]],
    ) -> str | None:
        """把错误升级到 Planner 的单次只读知识问答恢复为普通 RAG。

        多文档 collection、文件写入、邮件等动作继续保留在 Planner 中；这里只处理
        没有副作用且最多涉及一份候选文档的空计划或单 knowledge.search 计划。
        """
        if not decision.needs_index_catalog or decision.requires_file_output:
            return None
        steps = plan_preview.get("steps", [])
        if not isinstance(steps, list):
            return None
        all_tools: set[str] = set()
        query = str(decision.arguments.get("query") or question).strip()
        explicit_doc_ids: list[str] = []
        scope = "search"
        for step in steps:
            if not isinstance(step, dict):
                continue
            tools = {str(tool) for tool in step.get("allowed_tools", [])}
            all_tools.update(tools)
            args = step.get("arguments", {}) if isinstance(step.get("arguments"), dict) else {}
            if "knowledge.search" in tools:
                query = str(args.get("query") or query).strip()
                scope = str(args.get("scope") or scope).strip().lower()
                values = args.get("doc_ids", [])
                if isinstance(values, list):
                    explicit_doc_ids.extend(str(value) for value in values if str(value).strip())
        if all_tools - {"knowledge.search"}:
            return None
        if scope == "collection" or len(set(explicit_doc_ids)) > 1:
            return None
        hinted_docs = {
            str(item.get("doc_id")) for item in index_hint
            if item.get("doc_id") and item.get("strong_match") is True
        }
        if not all_tools and len(hinted_docs) != 1:
            return None
        return query or question

    @staticmethod
    def _planned_web_file_request(plan_preview: dict[str, object]) -> dict[str, str] | None:
        """仅从 Planner 的依赖节点识别网页检索后写文件。"""
        knowledge: dict[str, object] | None = None
        commits: list[dict[str, object]] = []
        for step in plan_preview.get("steps", []):
            if not isinstance(step, dict):
                continue
            tools = step.get("allowed_tools", [])
            if "web.search" in tools:
                knowledge = step
            if "files.write_file" in tools:
                commits.append(step)
        if not knowledge or not commits:
            return None
        paths = []
        for commit in commits:
            arguments = commit.get("arguments") if isinstance(commit.get("arguments"), dict) else {}
            path = str(arguments.get("path") or "").strip()
            if path:
                paths.append(path)
        unique_paths = list(dict.fromkeys(paths))
        if len(unique_paths) > 1:
            return {"error": "任务中出现多个不同的目标文件：" + "、".join(unique_paths)}
        commit = next((item for item in commits if knowledge.get("id") in item.get("depends_on", [])), commits[-1])
        source = knowledge.get("arguments") if isinstance(knowledge.get("arguments"), dict) else {}
        destination = commit.get("arguments") if isinstance(commit.get("arguments"), dict) else {}
        return {"query": str(source.get("query") or "").strip(),
                "path": str(destination.get("path") or "").strip()}

    def _desktop_directory(self) -> Path:
        """优先读取 Windows 的实际桌面目录，兼容重定向到 OneDrive 的情况。"""
        if sys.platform == "win32":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                    value, _ = winreg.QueryValueEx(key, "Desktop")
                desktop = Path(os.path.expandvars(value)).expanduser()
                if desktop.is_dir():
                    return desktop.resolve()
            except (OSError, ValueError):
                pass
        for candidate in (Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"):
            if candidate.is_dir():
                return candidate.resolve()
        raise ValueError("无法确定当前用户的桌面目录，请明确提供目标文件的绝对路径。")

    def _planned_output_path(self, question: str, raw_path: str) -> Path:
        # 文件名优先取用户明确给出的目标，LLM 不得用项目目录替代用户指定的桌面。
        extracted = self._extract_file_write_request(question)
        user_path = str(extracted.get("path") or "") if extracted else ""
        raw = self._normalize_file_write_target(raw_path) if raw_path else ""
        # 用户明确说出的文件名优先于模型规划的候选，防止规划意外改名或改目录。
        target = Path(user_path or raw).expanduser() if user_path or raw else None
        if target is None or not target.suffix:
            raise ValueError("缺少目标文件名或扩展名，未执行文件写入。")
        desktop_requested = bool(re.search(r"(?:在|到|至|放在|放到|保存到)\s*(?:我的|用户的)?桌面", question))
        if desktop_requested:
            # 用户输入若已经包含同一个绝对路径，尊重原路径；其他情况按桌面位置重新绑定。
            if user_path and Path(user_path).is_absolute() and user_path in question:
                return Path(user_path).expanduser().resolve()
            return (self._desktop_directory() / target.name).resolve()
        if not target.is_absolute():
            target = self.workspace_root / target
        return target.resolve()

    def _answer_web_file_request(
        self, question: str, request: dict[str, str], steps: list[AgentStep],
        session_id: str, user_message_id: str, memory_context: AssembledContext,
    ) -> AnswerResult:
        try:
            if request.get("error"):
                raise ValueError(str(request["error"]) + "。请确认搜索结果最终应写入哪个文件，本次未创建文件。")
            target = self._planned_output_path(question, request.get("path", ""))
            query = request.get("query", "").strip()
            if not query:
                raise ValueError("计划没有提供网页搜索主题，未写入文件。")
        except ValueError as exc:
            return self._finalize_answer(
                answer=str(exc), evidences=[], steps=steps, used_llm=False,
                session_id=session_id, user_message_id=user_message_id,
                question=question, memory_context=memory_context,
            )
        if target.exists():
            return self._finalize_answer(
                answer=f"目标文件已存在，未覆盖：{target}。请指定新文件名；此搜索建文件任务默认不覆盖已有文件。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )

        def search(values: dict[str, object]) -> dict[str, object]:
            result = self.tool_registry.call("web.search", query=query, limit=5)
            if not result.ok:
                raise RuntimeError(result.error or "网页搜索失败")
            if not isinstance(result.output, list) or not result.output:
                raise RuntimeError("网页搜索没有返回结果；未创建文件。")
            return {"search_results": result.output}

        def format_results(values: dict[str, object]) -> dict[str, object]:
            lines = [f"网页搜索结果：{query}", ""]
            count = 0
            for item in values["search_results"]:
                if not isinstance(item, dict) or not str(item.get("url") or "").strip():
                    continue
                count += 1
                lines.extend([f"[{count}] {str(item.get('title') or '无标题').strip()}",
                              f"来源：{item['url']}", str(item.get("snippet") or "").strip(), ""])
            if len(lines) == 2:
                raise RuntimeError("搜索结果没有可核对的来源链接；未创建文件。")
            return {"file_content": "\n".join(lines).strip() + "\n"}

        execution = self.runtime.run([
            PlanStep("web_search", search, "已取得带来源的网页搜索结果。", max_retries=0),
            PlanStep("prepare_file_content", format_results, "已将真实搜索结果和来源整理成正文。", max_retries=0),
        ])
        steps.extend(execution.agent_steps)
        if execution.failed:
            return self._finalize_answer(
                answer=f"未写入文件：{execution.failure}", evidences=[], steps=steps,
                used_llm=False, session_id=session_id, user_message_id=user_message_id,
                question=question, memory_context=memory_context,
            )
        content = str(execution.values["file_content"])
        result = self.tool_registry.call("files.write_file", path=str(target), content=content, overwrite=False)
        output = result.output if result.ok and isinstance(result.output, dict) else {}
        pending = None
        if output.get("written"):
            answer = f"已将搜索结果写入文件：{target}\n\n内容附有对应来源链接。"
            status = "success"
        elif output.get("permission", {}).get("requires_confirmation"):
            permission = output["permission"]
            pending = {"tool_name": "files.write_file",
                       "kwargs": {"path": str(target), "content": content, "overwrite": False},
                       "description": f"写入搜索结果：{target}",
                       "risk_level": permission.get("risk_level", "high"),
                       "reasons": permission.get("reasons", [])}
            answer = f"搜索结果已准备好，写入 {target} 需要人工确认；确认前文件不会被创建。"
            status = "waiting_human"
        else:
            answer = f"搜索结果已准备好，但文件未写入：{result.error or output.get('message', '未知错误')}"
            status = "failed"
        steps.append(AgentStep("write_file", status, str(target)))
        return self._finalize_answer(
            answer=answer, evidences=[], steps=steps, used_llm=False,
            session_id=session_id, user_message_id=user_message_id,
            question=question, memory_context=memory_context, pending_action=pending,
        )

    @staticmethod
    def _planned_index_report(plan_preview: dict[str, object]) -> dict[str, object] | None:
        """以工具依赖关系识别索引检索 -> 文件提交，不依赖用户的固定措辞。"""
        knowledge: dict[str, object] | None = None
        commit: dict[str, object] | None = None
        for step in plan_preview.get("steps", []):
            if not isinstance(step, dict):
                continue
            tools = step.get("allowed_tools", [])
            if "knowledge.search" in tools:
                knowledge = step
            if "files.write_file" in tools:
                commit = step
        if knowledge is None:
            return None
        source = knowledge.get("arguments") if isinstance(knowledge.get("arguments"), dict) else {}
        output = commit.get("arguments") if commit and isinstance(commit.get("arguments"), dict) else {}
        return {"query": source.get("query", ""), "doc_ids": source.get("doc_ids", []),
                "path": output.get("path", ""), "write_report": commit is not None}

    def _select_index_doc_ids(self, query: str) -> list[str]:
        """Planner 漏选文档时，从有界索引目录补选；不能把所有网页都当作论文。"""
        catalog = self._index_document_catalog(query)
        if not catalog:
            return []
        reply = self.client.chat([
            {"role": "system", "content": "你是索引文档选择器，只输出 JSON。"},
            {"role": "user", "content": (
                f"任务主题：{query}\n索引目录：{json.dumps(catalog, ensure_ascii=False)}\n"
                '选最多 6 篇真正相关的文档；如果任务要求论文，优先学术论文而非框架介绍或博客。'
                '只输出目录中的 ID，格式：{"doc_ids":["..."]}；无法确定则空数组。'
            )},
        ], temperature=0.0, max_tokens=200)
        try:
            data = json.loads(reply.strip())
        except (ValueError, AttributeError):
            return []
        if not isinstance(data, dict) or not isinstance(data.get("doc_ids"), list):
            return []
        valid = {item["doc_id"] for item in catalog}
        return list(dict.fromkeys(str(item) for item in data["doc_ids"] if str(item) in valid))[:6]

    def _answer_index_report(
        self, question: str, request: dict[str, object], steps: list[AgentStep],
        session_id: str, user_message_id: str, memory_context: AssembledContext,
    ) -> AnswerResult:
        query = str(request.get("query") or question).strip()
        raw_ids = request.get("doc_ids")
        doc_ids = [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []
        if not doc_ids:
            doc_ids = self._select_index_doc_ids(query)
        if doc_ids and any(doc_id not in self.index.documents for doc_id in doc_ids):
            return self._finalize_answer(
                answer="计划选择的文档不在当前索引中，未生成报告。请刷新索引后重试。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        if not doc_ids:
            return self._finalize_answer(
                answer="无法从当前索引目录确定与主题相关的论文；本次没有生成或写入报告。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        # 总量限制在约 36 个片段；论文越少，每篇可读取的章节越多。
        chunks_per_doc = min(12, max(4, 36 // min(len(doc_ids), 6)))
        evidences = self.index.search_collection(query, doc_ids=doc_ids, max_docs=6, chunks_per_doc=chunks_per_doc)
        steps.append(AgentStep("retrieve_index_collection", "success" if evidences else "failed",
                               f"索引中选出 {len({item.doc_id for item in evidences})} 份文档、{len(evidences)} 条证据。"))
        if not evidences:
            return self._finalize_answer(
                answer="当前索引没有可用于该主题的论文证据，未生成或写入报告。请检查索引中的文档和可提取片段。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        sources = []
        references = []
        for number, evidence in enumerate(evidences, start=1):
            doc = self.index.documents[evidence.doc_id]
            sources.append(f"[{number}] {doc.title} | {evidence.source_label}\n{evidence.text[:600]}")
            references.append(f"[{number}] {doc.title}；来源：{doc.path}；片段：{evidence.source_label}")
        report = self.client.chat([
            {"role": "system", "content": (
                "你是基于索引证据撰写技术报告的助手。只输出 Markdown 报告正文，不描述工具、权限、"
                "Planner 或是否能够写文件；文件写入由独立执行器完成。每项技术结论标注提供的 [编号]。"
                "分别总结每篇选中的论文，至少覆盖方法及证据中可核实的实验或局限；"
                "每篇尽可能引用来自不同位置的多个片段，跨论文比较时不要引用无关网页。"
                "只能使用给定证据，证据不足的细节须明确说明，禁止编造论文方法或实验结果。"
                "不要自行编写参考来源章节，执行器会统一附上。"
            )},
            {"role": "user", "content": f"需求：{question}\n检索主题：{query}\n\n索引证据：\n" + "\n\n".join(sources)},
        ], max_tokens=3600)
        if not report:
            return self._finalize_answer(
                answer="已检索到论文证据，但报告正文生成失败，未写入文件。",
                evidences=evidences, steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        valid_numbers = set(range(1, len(evidences) + 1))
        content = re.sub(
            r"\[(\d+)\]", lambda match: match.group(0) if int(match.group(1)) in valid_numbers else "[来源待核实]",
            fix_mojibake(report).strip(),
        )
        if not re.search(r"\[(\d+)\]", content):
            content += "\n\n## 证据要点\n\n" + "\n".join(
                f"- [{number}] {self.index.documents[e.doc_id].title}：{e.text[:180].replace(chr(10), ' ')}"
                for number, e in enumerate(evidences, start=1)
            )
        content += "\n\n## 检索范围\n\n本报告基于索引中各论文不同位置的代表性片段，未逐页覆盖原文。\n"
        content += "\n## 证据片段索引\n\n" + "\n".join(
            f"- [{number}] {self.index.documents[e.doc_id].title}：{e.text[:130].replace(chr(10), ' ')}"
            for number, e in enumerate(evidences, start=1)
        )
        content += "\n\n## 参考来源\n\n" + "\n".join(references) + "\n"
        if not request.get("write_report", True):
            return self._finalize_answer(
                answer=content, evidences=evidences, steps=steps, used_llm=True,
                session_id=session_id, user_message_id=user_message_id,
                question=question, memory_context=memory_context,
            )
        raw_path = str(request.get("path") or "").strip()
        current_directory = bool(re.search(r"(?:当前|本)(?:工作)?(?:目录|文件夹)(?:下|中|里)?", question))
        if current_directory:
            # 当用户只说位置、没指定文件名时，不采用 Planner 猜出的子目录或绝对路径。
            explicit_filename = re.search(r"[^\s，,；;。/\\]+?\.(?:md|markdown|txt|docx|pdf)\b", question, re.IGNORECASE)
            raw_path = (self._normalize_file_write_target(explicit_filename.group(0).strip('"\'`“”‘’'))
                        if explicit_filename else "")
        elif raw_path in {"当前目录", "当前目录下", "本目录", ".", "./"}:
            raw_path = ""
        target = Path(raw_path).expanduser() if raw_path else Path("技术报告.md")
        if not target.suffix:
            target = target / "技术报告.md"
        if not target.is_absolute():
            target = self.workspace_root / target
        if not raw_path:
            original = target
            suffix = 1
            while target.exists():
                target = original.with_name(f"{original.stem}_{suffix}{original.suffix}")
                suffix += 1
        result = self.tool_registry.call("files.write_file", path=str(target), content=content, overwrite=False)
        output = result.output if result.ok and isinstance(result.output, dict) else {}
        waiting = bool(output.get("permission", {}).get("requires_confirmation") and not output.get("written"))
        steps.append(AgentStep("write_report", "waiting_human" if waiting else "success" if output.get("written") else "failed",
                               str(target) if output.get("written") or waiting else result.error or str(output.get("message", "未写入"))))
        pending = None
        if result.ok and isinstance(result.output, dict) and result.output.get("written"):
            message = f"已生成技术报告：{target}\n\n共引用 {len({item.doc_id for item in evidences})} 份索引文档；报告末尾附有对应来源。"
        elif result.ok and isinstance(result.output, dict) and result.output.get("permission", {}).get("requires_confirmation"):
            permission = result.output["permission"]
            pending = {"tool_name": "files.write_file", "kwargs": {"path": str(target), "content": content, "overwrite": False},
                       "description": f"写入技术报告：{target}", "risk_level": permission.get("risk_level", "high"),
                       "reasons": permission.get("reasons", [])}
            message = f"报告正文已生成，写入 {target} 需要人工确认。确认后才会创建文件。"
        else:
            message = f"报告正文已生成，但文件未写入：{result.error or result.output}"
        return self._finalize_answer(
            answer=message, evidences=evidences, steps=steps, used_llm=True,
            session_id=session_id, user_message_id=user_message_id,
            question=question, memory_context=memory_context, pending_action=pending,
        )

    def _extract_file_write_request(self, question: str) -> dict[str, object] | None:
        if not self._is_file_write_instruction(question):
            return None
        patterns = [
            r"[\"'`“”‘’]([^\"'`“”‘’]+?\.(?:md|markdown|txt|pdf|docx|csv|json|yaml|yml|py|js|ts|html|css))[\"'`“”‘’]",
            r"([^\s，。；;、\"'`“”‘’]+?\.(?:md|markdown|txt|pdf|docx|csv|json|yaml|yml|py|js|ts|html|css))",
        ]
        target = ""
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                target = self._normalize_file_write_target(match.group(1))
                target = self._apply_file_write_directory_hint(question, target)
                break
        if not target:
            return None
        return {
            "path": target,
            "content": self._extract_file_write_content(question),
            "overwrite": self._should_overwrite_for_file_write(question),
        }

    def _apply_file_write_directory_hint(self, question: str, target: str) -> str:
        if not target:
            return target
        target_path = Path(target)
        if target_path.is_absolute():
            return str(target_path)
        directory = self._extract_file_write_directory_hint(question)
        if not directory:
            return target
        return str((Path(directory) / target).resolve(strict=False))

    def _extract_file_write_directory_hint(self, question: str) -> str:
        action_words = (
            "\u521b\u5efa",
            "\u65b0\u5efa",
            "\u5efa\u7acb",
            "\u751f\u6210",
            "\u5199\u5165",
            "\u5199\u5230",
            "\u4fdd\u5b58\u4e3a",
            "\u53e6\u5b58\u4e3a",
            "\u5bfc\u51fa",
            "\u66f4\u65b0",
            "\u4fee\u6539",
            "\u66ff\u6362",
        )
        action_pattern = "|".join(re.escape(word) for word in action_words)
        patterns = [
            rf"(?:\u5728|\u5230|\u81f3)\s*([A-Za-z]:[\\/].*?)(?:\u76ee\u5f55\u4e0b|\u6587\u4ef6\u5939\u4e0b|\u4e0b\u9762|\u4e0b|\u4e2d|\u91cc\u9762|\u5185)\s*(?:{action_pattern})",
            rf"(?:\u5728|\u5230|\u81f3)\s*([A-Za-z]:[\\/].*?)\s+(?:{action_pattern})",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"').strip("'").strip("`“”‘’")
        return ""

    def _is_file_write_instruction(self, question: str) -> bool:
        normalized = question.lower()
        markers = (
            "创建",
            "新建",
            "建立",
            "生成",
            "写入",
            "写到",
            "保存为",
            "另存为",
            "导出",
            "更新",
            "修改",
            "替换",
        )
        return any(marker in normalized for marker in markers)

    def _should_overwrite_for_file_write(self, question: str) -> bool:
        normalized = question.lower()
        explicit_write_markers = (
            "写入",
            "写到",
            "保存为",
            "另存为",
            "导出",
            "更新",
            "修改",
            "替换",
        )
        return any(marker in normalized for marker in explicit_write_markers)

    def _normalize_file_write_target(self, raw_target: str) -> str:
        target = str(raw_target or "").strip().strip('"').strip("'").strip("`“”‘’")
        target = target.strip("，。；;、,. ")
        action_markers = (
            "保存为",
            "另存为",
            "创建一个",
            "创建一份",
            "创建",
            "新建一个",
            "新建一份",
            "新建",
            "建立一个",
            "建立一份",
            "建立",
            "生成一个",
            "生成一份",
            "生成",
            "写入一个",
            "写入一份",
            "写到",
            "导出为",
            "导出",
        )
        for marker in action_markers:
            position = target.rfind(marker)
            if position >= 0:
                candidate = target[position + len(marker) :].strip()
                if "." in candidate:
                    target = candidate
                    break
        location_prefixes = (
            "当前目录下的",
            "当前目录下",
            "当前文件夹下的",
            "当前文件夹下",
            "本目录下的",
            "本目录下",
            "工作区下的",
            "工作区下",
            "目录下的",
            "目录下",
            "一个",
            "一份",
            "名为",
            "叫做",
            "为",
            "在",
        )
        changed = True
        while changed and target:
            changed = False
            for prefix in location_prefixes:
                if target.startswith(prefix) and len(target) > len(prefix):
                    target = target[len(prefix) :].strip()
                    changed = True
                    break
        return target.strip("，。；;、,. ")

    def _extract_file_write_content(self, question: str) -> str:
        for marker in ("内容是", "内容为", "内容：", "内容:", "写入内容", "写入", "写到"):
            position = question.find(marker)
            if position >= 0:
                return question[position + len(marker) :].strip()
        return ""

    def _materialize_file_write_content(self, question: str, target_path: str, requested_content: str) -> tuple[str, bool, str]:
        requested_content = str(requested_content or "").strip()
        if not requested_content:
            return "", False, "empty"
        if not self._should_generate_file_write_content(requested_content):
            return requested_content, False, "literal"

        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 DeskPilot 的文件内容生成器。用户要把内容写入本地文件。"
                        "请只输出应该写入文件的正文，不要解释工具调用，不要包裹 Markdown 代码块。"
                        "如果用户请求的是作品全文，只能输出公有领域或用户有权使用的内容；"
                        "如果不能确定版权或事实准确性，请输出简短说明而不是编造。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"目标文件：{target_path}\n"
                        f"用户原始请求：{question}\n"
                        f"需要写入/生成的内容描述：{requested_content}\n\n"
                        "请生成要写入文件的正文。"
                    ),
                },
            ],
            temperature=0.2,
        )
        if response:
            return fix_mojibake(response).strip(), True, "llm"
        raise RuntimeError(
            "请求内容需要生成，但当前 LLM 未返回可写入正文；为避免把内容描述原样写入文件，本次停止写入。"
        )

    def _should_generate_file_write_content(self, requested_content: str) -> bool:
        normalized = requested_content.lower()
        generation_markers = (
            "全文",
            "完整内容",
            "整篇",
            "原文",
            "正文",
            "写一篇",
            "生成一篇",
            "生成一份",
        )
        return any(marker in normalized for marker in generation_markers)

    def _answer_file_write(
        self,
        question: str,
        file_write_request: dict[str, object],
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        target_path = str(file_write_request["path"])
        requested_content = str(file_write_request.get("content", ""))
        overwrite = bool(file_write_request.get("overwrite", False))
        # 位置描述不是可写入的正文；复杂任务应由 Planner 先取到工具结果。
        if re.fullmatch(
            r"(?:到|至)?(?:整个|这个|该)?文件(?:中|里|内)?|(?:到|至)?[^\s]+\.(?:txt|md|pdf|docx)(?:中|里|内)?",
            requested_content.strip(), flags=re.IGNORECASE,
        ):
            return self._finalize_answer(
                answer="未取得可写入的正文；文件位置说明不能作为内容。请先完成前置资料获取，未写入文件。",
                evidences=[], steps=steps, used_llm=False, session_id=session_id,
                user_message_id=user_message_id, question=question, memory_context=memory_context,
            )
        steps.append(AgentStep("route_file_write", "success", f"Detected file write request: {target_path}"))

        def materialize(values: dict[str, object]) -> dict[str, object]:
            content, generated_content, generation_method = self._materialize_file_write_content(
                question=question,
                target_path=target_path,
                requested_content=requested_content,
            )
            return {
                "content": content,
                "generated_content": generated_content,
                "generation_method": generation_method,
            }

        def call_write_tool(values: dict[str, object]) -> dict[str, object]:
            result = self.tool_registry.call(
                "files.write_file",
                path=target_path,
                content=str(values.get("content", "")),
                overwrite=overwrite,
            )
            return {"write_result": result}

        execution = self.runtime.run(
            [
                PlanStep("materialize_file_content", materialize, "Prepared content for the target file.", max_retries=0),
                PlanStep("write_file", call_write_tool, "Called files.write_file through the tool registry.", max_retries=0),
            ],
            {
                "target_path": target_path,
                "requested_content": requested_content,
                "overwrite": overwrite,
            },
        )
        steps.extend(execution.agent_steps)
        content = str(execution.values.get("content", requested_content))
        if execution.values.get("generated_content"):
            method = str(execution.values.get("generation_method", "generated"))
            steps.append(AgentStep("generate_file_content", "success", f"Generated file content: {method}"))
        result = execution.values.get("write_result")
        if execution.failed or result is None:
            answer = f"文件写入任务失败：{target_path}\n\n原因：{execution.failure or 'plan did not produce a write result'}"
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        if not result.ok:
            steps.append(AgentStep("write_file", "failed", result.error or "file write tool failed"))
            answer = f"创建文件失败：{target_path}。原因：{result.error or '未知错误'}"
            return self._finalize_answer(
                answer=answer,
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        output = result.output or {}
        if output.get("written"):
            path = str(output.get("path", target_path))
            size = output.get("size", 0)
            answer = f"已写入文件：{path}\n\n文件大小：{size} bytes。"
        else:
            permission = output.get("permission", {})
            reasons = permission.get("reasons", []) if isinstance(permission, dict) else []
            reason_text = "；".join(str(item) for item in reasons) if reasons else str(output.get("message", "文件未写入"))
            if isinstance(permission, dict) and permission.get("requires_confirmation"):
                pending_action = {
                    "tool_name": "files.write_file",
                    "kwargs": {
                        "path": target_path,
                        "content": content,
                        "overwrite": overwrite,
                    },
                    "description": f"写入文件：{target_path}",
                    "risk_level": permission.get("risk_level", "high"),
                    "reasons": reasons,
                }
                answer = (
                    f"该操作需要人工确认：{target_path}\n\n"
                    f"原因：{reason_text}\n\n"
                    "请在弹出的确认框中选择是否继续执行。"
                )
                return self._finalize_answer(
                    answer=answer,
                    evidences=[],
                    steps=steps,
                    used_llm=False,
                    session_id=session_id,
                    user_message_id=user_message_id,
                    question=question,
                    memory_context=memory_context,
                    pending_action=pending_action,
                )
            answer = (
                f"文件没有被创建：{target_path}\n\n"
                f"原因：{output.get('message', '文件未写入')}\n"
                f"权限判断：{reason_text}"
            )
        return self._finalize_answer(
            answer=answer,
            evidences=[],
            steps=steps,
            used_llm=False,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
        )

    def _answer_local_documents(
        self, question: str, targets: list[str], steps: list[AgentStep], session_id: str,
        user_message_id: str, memory_context: AssembledContext,
    ) -> AnswerResult:
        """批量读取本地文档，避免多文档请求被单文件槽位校验截断。"""
        contents: list[tuple[str, str]] = []
        evidences: list[Evidence] = []
        for target in targets:
            resolved = self.tool_registry.call("files.resolve_document", target=target)
            if not resolved.ok:
                return self._finalize_answer(
                    answer=f"无法定位本地文档：{target}。请确认文件名或检查文档别名。",
                    evidences=evidences, steps=steps, used_llm=False,
                    session_id=session_id, user_message_id=user_message_id,
                    question=question, memory_context=memory_context,
                )
            read = self.tool_registry.call("files.read_document", path=str(resolved.output))
            if not read.ok:
                return self._finalize_answer(
                    answer=f"已定位文档，但读取失败：{target}。原因：{read.error}",
                    evidences=evidences, steps=steps, used_llm=False,
                    session_id=session_id, user_message_id=user_message_id,
                    question=question, memory_context=memory_context,
                )
            payload = read.output or {}
            content = str(payload.get("content", ""))
            title = str(payload.get("title", target))
            contents.append((title, content))
            evidences.append(Evidence(
                chunk_id=f"local_{hashlib.sha256(str(resolved.output).encode('utf-8')).hexdigest()[:12]}",
                doc_id=str(resolved.output), source_label=f"file:{title}",
                text=content[:1800], score=1.0,
            ))
        prompt = "\n\n".join(f"文档：{title}\n内容：{content[:12000]}" for title, content in contents)
        answer = self.client.chat([
            {"role": "system", "content": "你是本地文档对比助手，只能根据给定文档内容回答，不得补充文档外事实。"},
            {"role": "user", "content": f"请分别提取各文档核心目标，再给出对比摘要。\n{prompt}"},
        ], temperature=0.2)
        if not answer:
            answer = "\n\n".join(f"## {title}\n{content[:800]}" for title, content in contents)
        steps.append(AgentStep("read_local_documents", "success", f"已读取 {len(contents)} 份本地文档。"))
        return self._finalize_answer(
            answer=fix_mojibake(answer), evidences=evidences, steps=steps,
            used_llm=bool(answer) and not bool(self.client.last_error), session_id=session_id,
            user_message_id=user_message_id, question=question,
            memory_context=memory_context,
        )

    def _answer_local_document(
        self,
        question: str,
        local_document_target: str,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
        read_mode: str = "",
    ) -> AnswerResult | None:
        steps.append(AgentStep("route_local_document", "success", f"检测到本地文档请求：{local_document_target}"))
        resolved = self.tool_registry.call("files.resolve_document", target=local_document_target)
        if not resolved.ok:
            steps.append(AgentStep("resolve_local_document", "failed", resolved.error or "无法定位文档"))
            return self._finalize_answer(
                answer=f"无法定位本地文档：{local_document_target}。请确认文件名是否正确，或把文件放到当前工作目录后重试。",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        resolved_path = str(resolved.output)
        steps.append(AgentStep("resolve_local_document", "success", f"已定位文档：{resolved_path}"))
        read_result = self.tool_registry.call("files.read_document", path=resolved_path)
        if not read_result.ok:
            steps.append(AgentStep("read_local_document", "failed", read_result.error or "无法读取文档"))
            return self._finalize_answer(
                answer=f"已找到文档，但读取失败：{resolved_path}。",
                evidences=[],
                steps=steps,
                used_llm=False,
                session_id=session_id,
                user_message_id=user_message_id,
                question=question,
                memory_context=memory_context,
            )

        payload = read_result.output or {}
        content = str(payload.get("content", "")).strip()
        title = str(payload.get("title", local_document_target)).strip() or local_document_target
        document_path = str(payload.get("path", resolved_path))
        steps.append(AgentStep("read_local_document", "success", f"已读取文档内容：{document_path}"))
        evidence_text = content if len(content) <= 1800 else content[:1800] + "..."
        doc_hash = hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:12]
        evidences = [
            Evidence(
                chunk_id=f"local_{doc_hash}",
                doc_id=f"local_{doc_hash}",
                source_label=f"file:{title}",
                text=evidence_text,
                score=1.0,
            )
        ]
        normalized_mode = self._local_document_read_mode(question, read_mode)
        if normalized_mode == "verbatim":
            if len(content) <= 12000:
                answer = f"文件 `{title}` 的内容如下：\n\n{content}"
            else:
                answer = (
                    f"文件 `{title}` 较长，以下显示前 12000 个字符：\n\n{content[:12000]}\n\n"
                    f"[内容已截断，文件总字符数：{len(content)}]"
                )
            steps.append(AgentStep("present_local_document", "success", "按用户要求展示文件原文。"))
            return self._finalize_answer(
                answer=answer, evidences=evidences, steps=steps, used_llm=False,
                session_id=session_id, user_message_id=user_message_id,
                question=question, memory_context=memory_context,
            )
        summary, used_llm = self._summarize_local_document(
            title=title,
            path=document_path,
            content=content,
            question=question,
            read_mode=normalized_mode,
        )
        step_name = "summarize_local_document" if normalized_mode == "summary" else "answer_local_document"
        detail = "已基于本地文档内容生成总结。" if normalized_mode == "summary" else "已严格根据本地文档回答问题。"
        steps.append(AgentStep(step_name, "success", detail))
        return self._finalize_answer(
            answer=summary,
            evidences=evidences,
            steps=steps,
            used_llm=used_llm,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
        )

    @staticmethod
    def _local_document_read_mode(question: str, requested: str) -> str:
        if requested in {"verbatim", "summary", "question_answer"}:
            return requested
        if re.search(r"(?:总结|概括|摘要|提炼|要点)", question):
            return "summary"
        if re.search(r"(?:内容是什么|有哪些内容|显示内容|查看内容|原文|全文)", question):
            return "verbatim"
        return "question_answer"

    def _summarize_local_document(
        self, title: str, path: str, content: str, question: str, read_mode: str = "summary",
    ) -> tuple[str, bool]:
        preview = content[:12000]
        task_instruction = (
            "请直接回答用户针对文件提出的问题；不要擅自改写成全文摘要。"
            if read_mode == "question_answer" else
            "请先给出 3-6 条要点总结，再给出一段简短整体概括；有章节时按章节组织。"
        )
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 DeskPilot 的本地文档总结助手。用户刚刚打开了一个本地文件，"
                        "你需要严格基于文件内容总结，不要引用外部网页，不要编造未出现的信息。"
                        "如果内容不足，直接说明内容有限。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"文件名：{title}\n"
                        f"文件路径：{path}\n"
                        f"用户问题：{question}\n\n"
                        f"任务要求：{task_instruction}\n"
                        "不要编造文件中没有的信息。\n\n"
                        f"文件内容：\n{preview}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if response:
            return fix_mojibake(response), True
        return self._fallback_local_document_summary(title, content), False

    def _fallback_local_document_summary(self, title: str, content: str) -> str:
        lines = [f"# {title}", "", "## 摘要", ""]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        for paragraph in paragraphs[:5]:
            snippet = paragraph.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            lines.append(f"- {snippet}")
        if len(lines) == 4:
            lines.append("- 文档内容较少，未提取到足够的段落。")
        lines.extend(["", "## 说明", "", "以上总结仅基于本地文件内容生成。"])
        return "\n".join(lines)

    def research(self, topic: str, session_id: str | None = None, max_results: int | None = None) -> ResearchResult:
        with self._operation_lock:
            return self._research_unlocked(topic, session_id=session_id, max_results=max_results)

    def _research_unlocked(
        self, topic: str, session_id: str | None = None, max_results: int | None = None,
    ) -> ResearchResult:
        session = self.session_store.get_or_create(session_id)
        user_message = self.session_store.append_message(session.session_id, "user", f"网页调研：{topic}")
        memory_context = self._load_memory_context(topic, session.session_id)
        result = self.web_research_agent.research(topic, max_results=max_results)
        result.session_id = session.session_id
        result.memory_context = memory_context.text
        result.steps.insert(
            1,
            AgentStep(
                "load_memory_context",
                "success",
                "已加载会话记忆："
                + "；".join(memory_context.debug_lines),
            ),
        )
        assistant_content = f"{result.report}\n\n报告文件：{result.artifact_path}"
        assistant_message = self.session_store.append_message(session.session_id, "assistant", assistant_content)
        artifact_memory = self.memory_store.add_memory(
            scope="session",
            memory_type="artifact",
            content=f"网页调研报告：{topic} -> {result.artifact_path}",
            source_session_id=session.session_id,
            source_message_ids=[user_message.message_id, assistant_message.message_id],
            confidence=0.86,
            tags=["research_report", "web_research"],
        )
        if artifact_memory:
            self.session_store.append_session_item(session.session_id, artifact_memory.memory_type, artifact_memory.to_dict())
        memory_result = MemoryTurnManager(
            self.memory_extractor, self.memory_store, self.session_store, self.memory_compactor,
        ).process(
            user_message=f"网页调研：{topic}", assistant_message=assistant_content,
            session_id=session.session_id,
            source_message_ids=[user_message.message_id, assistant_message.message_id],
            grounded=bool(result.evidences), steps=result.steps,
        )
        saved_count = memory_result.extracted + (1 if artifact_memory else 0)
        result.steps.append(AgentStep("extract_memory", "success", f"调研任务抽取并保存 {saved_count} 条结构化记忆。"))
        result.steps.append(
            AgentStep(
                "compact_memory",
                "success",
                "会话达到阈值，已更新滚动摘要。" if memory_result.compacted else "当前会话未达到压缩阈值，暂不更新摘要。",
            )
        )
        return result

    def _answer_web_research_request(
        self,
        question: str,
        topic: str,
        steps: list[AgentStep],
        session_id: str,
        user_message_id: str,
        memory_context: AssembledContext,
    ) -> AnswerResult:
        steps.append(AgentStep("route_web_research", "success", f"Detected web research request: {topic}"))
        result = self.web_research_agent.research(topic)
        steps.extend(result.steps)
        answer = result.report
        if result.artifact_path:
            answer = f"{answer}\n\n报告文件：{result.artifact_path}"
        return self._finalize_answer(
            answer=answer,
            evidences=result.evidences,
            steps=steps,
            used_llm=result.used_llm,
            session_id=session_id,
            user_message_id=user_message_id,
            question=question,
            memory_context=memory_context,
        )

    def _direct_answer(self, question: str, memory_context: AssembledContext) -> str:
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 DeskPilot 的通用问答助手。直接回答用户问题，保持简洁、准确、中文优先。"
                        "会话记忆只能用于理解用户项目、偏好和历史决策，不能作为通用知识事实来源。"
                        "版本规则是最高优先级：没有版本号的精确产品名默认指该名称的初始正式版本，不能自动替换成最新一代；"
                        "如果该名称也常被泛指整个系列，回答第一句必须明确‘以下按初始版本回答’，再简要说明后续版本差异。"
                        "禁止把后续版本才引入的架构、训练方法或能力归到初始版本上。"
                        "对无法确认的具体结构或训练细节应明确说明不确定。"
                        "通用知识问题只回答用户询问的主题，不要主动扩展到 DeskPilot、当前项目或其技术实现；"
                        "除非用户明确询问当前项目，并且上下文中存在可核验的项目资料，否则不得声称 DeskPilot 使用了"
                        "某个框架、模型、算法或基础设施。"
                        "绝不声称已经调用工具、搜索网页、修改文件或发送邮件；只有真实工具结果才能声明执行成功。"
                        "如果用户是在补充上一轮工具参数，应说明需要重新执行工具流程，不能模拟执行结果。"
                    ),
                },
                {"role": "user", "content": memory_context.text or question},
            ],
            temperature=0.3,
            max_tokens=memory_context.output_budget or None,
        )

    def _direct_fallback_answer(self, question: str, memory_context: AssembledContext) -> str:
        if self.client.config.llm_api_key and self.client.last_error:
            return (
                "LLM API Key 已读取到，但本次 API 调用失败，所以暂时只能使用本地 fallback。\n\n"
                f"错误原因：{self.client.last_error}\n\n"
                "请检查网络连通性、模型名、base_url，以及服务商账号额度/权限。"
            )
        if memory_context.retrieved_memories:
            lines = ["当前未配置可用 LLM API。我先根据已保存的会话记忆给出可核对的信息：", ""]
            for item in memory_context.retrieved_memories[:5]:
                lines.append(f"- [{item.memory_type}] {item.content}")
            lines.extend(["", "如果需要更自然的综合回答，请在 `.env` 中填写 `DASHSCOPE_API_KEY`。"])
            return "\n".join(lines)
        return (
            "当前问题不需要检索文档，但没有可用 LLM API，也没有检索到相关会话记忆。"
            "请在 `.env` 中填写 `DASHSCOPE_API_KEY` 后重试，或先继续对话让系统沉淀记忆。"
        )

    def _build_prompt(self, question: str, evidences: list[Evidence], memory_context: AssembledContext) -> str:
        return (
            f"统一运行时上下文：\n{memory_context.text or question}\n\n"
            "请只根据 Evidence 回答。每个事实性段落或列表项都要标注对应的 [数字] 来源；"
            "不得引用 Retrieved Memories，也不得在承认证据不足后继续根据记忆给出肯定结论。"
        )

    def _validate_grounded_answer(self, answer: str, evidences: list[Evidence]) -> tuple[bool, list[str]]:
        """执行确定性引用检查，防止记忆或无效编号冒充文档证据。"""
        reasons: list[str] = []
        if re.search(r"\[(?:来源\s*:\s*)?Retrieved Memories\]", answer, flags=re.IGNORECASE):
            reasons.append("回答引用了 Retrieved Memories，而不是文档证据")
        cited = self._cited_evidence_indexes(answer, len(evidences))
        if not cited:
            reasons.append("回答没有合法的 [数字] 文档引用")
        all_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        invalid = sorted(number for number in all_numbers if number < 1 or number > len(evidences))
        if invalid:
            reasons.append(f"存在越界引用：{invalid}")
        factual_lines = []
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if len(line) < 18 or line.startswith(("#", ">")):
                continue
            if self._is_citation_exempt_line(line):
                continue
            if re.match(r"^\[\d+\]\s+", line):
                continue
            factual_lines.append(line)
        uncovered = [line for line in factual_lines if not re.search(r"\[\d+\]", line)]
        if uncovered:
            preview = " | ".join(line[:100] for line in uncovered[:2])
            reasons.append(f"有 {len(uncovered)} 个事实段落或列表项缺少引用：{preview}")
        return not reasons, reasons

    def _verify_citation_support(self, answer: str, evidences: list[Evidence]) -> tuple[bool, str]:
        """可选的高可靠校验：判断被引用结论是否真的由对应 Evidence 支持。

        默认关闭，避免普通问答增加一次模型调用。校验器异常时保留确定性引用检查的
        结果，不把网络或 JSON 格式故障误判为回答不可信。
        """
        evidence_text = "\n\n".join(
            f"[{index}] {item.source_label}\n{item.text[:1800]}"
            for index, item in enumerate(evidences[:8], start=1)
        )
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是引用支持审计器。逐条核对回答中的事实和引用证据，只输出 JSON："
                        '{"supported":true|false,"unsupported_claims":["..."]}。'
                        "不得使用常识、历史会话或外部知识。"
                    ),
                },
                {"role": "user", "content": f"Evidence:\n{evidence_text}\n\n回答：\n{answer[:6000]}"},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        if not response:
            return True, f"校验器调用失败，已沿用确定性引用检查：{self.client.last_error or '无返回'}"
        try:
            payload = parse_json_value(response, dict)
            if not isinstance(payload, dict):
                raise ValueError("引用校验器没有返回合法 JSON 对象")
            supported = bool(payload.get("supported"))
            unsupported = payload.get("unsupported_claims", [])
            detail = "引用事实支持检查通过。" if supported else (
                "存在未被证据支持的结论：" + "；".join(str(value) for value in unsupported[:5])
            )
            return supported, detail
        except (ValueError, TypeError, AttributeError) as exc:
            return True, f"校验器返回无法解析，已沿用确定性引用检查：{type(exc).__name__}"

    def _is_citation_exempt_line(self, line: str) -> bool:
        """识别标题和组织句；这些文本不承载独立事实，不要求单独引用。"""
        if line in {"引用来源：", "参考来源："} or line.startswith("```"):
            return True
        if re.fullmatch(r"\*\*[^*]+\*\*", line):
            return True
        if line.endswith(("：", ":")) and re.search(
            r"(?:如下|以下|包含|包括|分为|归纳|概括|环节|部分|方面|步骤|流程|架构|组成)", line
        ):
            return True
        return False

    def _cited_evidence_indexes(self, answer: str, evidence_count: int) -> list[int]:
        return sorted({
            int(value)
            for value in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(value) <= evidence_count
        })

    def _append_source_list(self, answer: str, evidences: list[Evidence]) -> str:
        """把数字引用展开成用户可直接核对的文件名和页码。"""
        # 来源清单由程序根据 Evidence 确定性生成，不信任模型自行拼接的文件名或 URL。
        body = re.split(r"(?m)^#{0,3}\s*(?:引用来源|参考来源)\s*[：:]\s*$", answer, maxsplit=1)[0].rstrip()
        indexes = self._cited_evidence_indexes(body, len(evidences))
        if not indexes:
            indexes = list(range(1, min(3, len(evidences)) + 1))
        lines = [body, "", "引用来源："]
        lines.extend(f"[{index}] {evidences[index - 1].source_label}" for index in indexes)
        return "\n".join(lines)

    def _fallback_answer(
        self,
        question: str,
        evidences: list[Evidence],
        reason: str = "当前未配置或未能调用可用的 LLM API",
    ) -> str:
        lines = [
            "基于当前检索到的文档片段，可以先给出以下摘要式回答：",
            "",
        ]
        top_score = max((evidence.score for evidence in evidences), default=0.0)
        selected = [
            (idx, evidence)
            for idx, evidence in enumerate(evidences, start=1)
            if evidence.score >= top_score * 0.75
        ][:3]
        if not selected and evidences:
            selected = [(1, evidences[0])]
        for idx, evidence in selected:
            snippet = self._relevant_evidence_excerpt(question, evidence.text)
            snippet = re.sub(r"^\s*\[\d+\]\s*", "", snippet)
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            lines.append(f"- [{idx}] {snippet}")
        lines.extend(
            [
                "",
                f"说明：{reason}，系统已改用本地证据摘要。可在 Steps 中查看 API 调用或引用校验详情。",
            ]
        )
        return "\n".join(lines)

    def _relevant_evidence_excerpt(self, question: str, text: str) -> str:
        """离线降级时展示最相关的句子，而不是机械截取 chunk 开头。"""
        rewritten = self.query_optimizer.rewrite(question).rewritten.casefold()
        terms = {
            term
            for term in re.findall(r"[a-z][a-z0-9+_.-]{2,}|[\u4e00-\u9fff]{2,}", rewritten)
            if len(term) >= 3
        }
        aspect_weights = {
            "architecture": 4,
            "workflow": 4,
            "structure": 3,
            "modules": 3,
            "process": 2,
            "pipeline": 1,
            "principle": 3,
            "mechanism": 3,
        }
        normalized = " ".join(text.split())
        sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|(?=\d+\.\d+\.\s)", normalized) if part.strip()]
        if not sentences:
            return normalized
        substantive = [index for index, sentence in enumerate(sentences) if len(sentence) >= 60]
        candidates = substantive or list(range(len(sentences)))
        scores = [
            sum(aspect_weights.get(term, 1) for term in terms if term in sentence.casefold())
            for sentence in sentences
        ]
        best = max(candidates, key=lambda index: scores[index])
        # 同时带上后一条句子，避免只显示章节标题而缺少正文。
        return " ".join(sentences[best:best + 2])
