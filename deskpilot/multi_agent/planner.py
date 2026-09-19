from __future__ import annotations

import json
import re
from collections.abc import Callable

from .schemas import TaskPlan, TaskPlanStep


class PlannerAgent:
    """P0 规则优先 Planner，后续可替换为 LLM + JSON Schema。"""

    def __init__(self, llm_call: Callable[[str], str] | None = None):
        # 正常路径使用 LLM 进行语义规划；无 Key 或调用失败时才进入保守回退。
        self.llm_call = llm_call

    def build_plan(
        self, question: str, workspace_files: list[str] | None = None, conversation_context: str = "",
        indexed_documents: list[dict[str, str]] | None = None,
    ) -> TaskPlan:
        text = question.strip()
        if self.llm_call:
            planned = self._build_plan_with_llm(text, workspace_files or [], conversation_context, indexed_documents or [])
            if planned is not None:
                return planned

        # 离线模式无法理解任意自然语言，只提供可预测的降级能力。
        lower = text.casefold()
        steps: list[TaskPlanStep] = []
        needs_knowledge = any(key in lower for key in ("搜索", "调研", "网页", "天气", "文档", "pdf", "ppt", "资料", "索引", "论文"))
        needs_communication = any(key in lower for key in ("邮件", "报告", "总结", "写入", "生成", "发送", "创建"))
        # 无模型时只为显式文件产物提供有限降级，正式路径依然由语义 Planner 选工具。
        file_output = bool(re.search(r"\.(?:md|txt|docx|pdf)(?![A-Za-z0-9_.])", text, re.I)) or (
            "报告" in lower and any(word in lower for word in ("输出", "保存", "写入", "生成", "创建"))
        )
        needs_side_effect = file_output or ("保存" in lower and "草稿" in lower) or any(
            key in lower for key in ("发送", "移动", "重命名")
        )
        # 降级路径只抽取显式搜索主题；不能将整条文件操作命令当作搜索词。
        search_topic = ""
        quoted = re.search(r"搜索\s*[‘'\"“]([^’'\"”]+)[’'\"”]", text)
        if quoted:
            search_topic = quoted.group(1).strip()
        elif "搜索" in text:
            search_topic = re.split(r"[，,。；;]|然后|把|并将", text.split("搜索", 1)[1], maxsplit=1)[0].strip()

        if needs_knowledge and needs_communication:
            indexed = bool(indexed_documents)
            source_tools = ["knowledge.search"] if indexed else ["web.search"]
            steps.append(TaskPlanStep(
                "knowledge", "knowledge", "收集并整理任务所需资料",
                allowed_tools=source_tools,
                arguments={"query": search_topic} if search_topic and not indexed else {},
            ))
            steps.append(TaskPlanStep(
                "communication", "communication", "根据资料生成邮件或文档内容",
                depends_on=["knowledge"], allowed_tools=[],
            ))
            if needs_side_effect:
                steps.append(TaskPlanStep(
                    "commit", "communication", "提交外部副作用动作",
                    depends_on=["communication"],
                    allowed_tools=["files.write_file"] if file_output else [
                        "email.send" if "发送" in lower else "email.save_draft"
                    ],
                    requires_human=True,
                ))
        elif needs_knowledge:
            steps.append(TaskPlanStep(
                "knowledge", "knowledge", "读取或检索资料",
                allowed_tools=["knowledge.search"] if indexed_documents else ["web.search", "files.read_document"],
            ))
        elif needs_communication:
            steps.append(TaskPlanStep(
                "communication", "communication", "生成用户要求的内容",
                allowed_tools=["files.write_file"] if file_output else ["email.create_reply_draft"],
                requires_human=needs_side_effect,
            ))

        route = "multi_agent" if len({step.agent for step in steps}) > 1 or len(steps) > 2 else "single_agent"
        score = 2 * len({step.agent for step in steps}) + 2 * sum(bool(step.depends_on) for step in steps) + len(steps) + 2 * int(needs_side_effect)
        return TaskPlan(goal=text, route=route, steps=steps, complexity_score=score)

    def _build_plan_with_llm(
        self, question: str, workspace_files: list[str], conversation_context: str = "",
        indexed_documents: list[dict[str, str]] | None = None,
    ) -> TaskPlan | None:
        prompt = f"""
你是任务 Planner。请根据用户请求生成严格 JSON，不要输出 Markdown：
{{"goal":"...","route":"single_agent|multi_agent","complexity_score":0,
 "steps":[{{"step_id":"knowledge|communication|commit","agent":"knowledge|communication",
 "description":"...","depends_on":[],"allowed_tools":[],"requires_human":false,"arguments":{{}}}}]}}

规则：knowledge 负责搜索和读取资料，communication 负责生成内容，commit 负责发送邮件、保存草稿或写文件；
所有 commit 节点必须 requires_human=true。简单问答可以 steps=[] 且 route=single_agent。
如果用户用自然语言描述文档，必须从“工作区文件清单”选择真实文件路径，并写入 knowledge.arguments.paths，不能把描述原样当作路径。
若用户要求汇总已建立索引的多篇资料并写报告：knowledge.allowed_tools=["knowledge.search"]，knowledge.arguments.query 填纯检索主题，knowledge.arguments.doc_ids 从“索引文档目录”选择相关 ID；commit.allowed_tools=["files.write_file"]，commit.arguments.path 填明确指定的报告文件路径；只说“当前目录”但未指定文件名时留空，由执行器生成技术报告.md。不得把知识库文档当作工作区路径。
若只要求汇总多篇已索引文档，同样选相关 doc_ids，但不要加入 commit 节点。不要把索引之外的文档编造为已索引。
“索引文档目录”给出类型和标题；如果用户要论文，优先选与主题直接相关的学术论文，不要把无关的 Agent 博客或框架文档选进论文报告。用户要求写报告到当前目录时，必须同时有 knowledge.search 与 commit.files.write_file，不能只给出知识回答。
若用户要求先联网搜索、再把搜索结果写进文件（即使先说创建文件、后说搜索也一样）：knowledge.allowed_tools=["web.search"]，knowledge.arguments.query 填纯搜索主题；commit.allowed_tools=["files.write_file"]，commit.arguments.path 填目标文件名或明确路径，depends_on=["knowledge"]。不要将“写入到整个文件中”等写入动作说明当成文件内容；写入正文来自前置搜索结果。若明确要求写到桌面，输出路径应在用户的桌面目录下，不能写到当前工作目录。
如果任务涉及邮件，必须在 commit.arguments 中填写 to、subject、request；没有明确标题时生成合理标题，没有明确正文时将用户要求作为 request。
邮件提交只能选一个工具：只保存草稿使用 email.save_draft；明确要求发送使用 email.send。生成草稿不等于发送。
如果用户要求先整理文档并作为附件发送，knowledge.allowed_tools 必须包含 web.research，commit.arguments 必须设置 attach_report=true；
附件路径是前置节点运行时产生的动态结果，不要虚构 attachment_paths。
request/query/topic 只能填写要搜索或调研的主题，不得包含“在网上搜索、整理成文档、作为附件、发送给某人”等操作指令。
读取源文件时不要输出清单之外的路径；新建的报告输出路径可按用户要求填写。不要省略已知的收件人、标题或查询内容。
如果当前输入是在补充上一轮未完成任务的参数，请结合最近会话重建完整计划；绝不输出“模拟调用”或声称工具已成功。
示例：用户说“读取项目说明和评测说明”，应从清单选择对应的 .md 文件；用户说“发给 a@b.com，标题为 test”，应输出 to=["a@b.com"] 和 subject="test"。
工作区文件清单：{json.dumps(workspace_files, ensure_ascii=False)}
索引文档目录：{json.dumps(indexed_documents or [], ensure_ascii=False)}
Planner 独立运行时上下文（其中已包含当前任务）：{conversation_context or question}
""".strip()
        try:
            response = self.llm_call(prompt)
            match = re.search(r"\{.*\}", response or "", flags=re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            allowed_agents = {"knowledge", "communication"}
            steps: list[TaskPlanStep] = []
            for raw in data.get("steps", []):
                if not isinstance(raw, dict) or raw.get("agent") not in allowed_agents:
                    return None
                tools = [str(tool) for tool in raw.get("allowed_tools", [])]
                risky = {"email.send", "email.save_draft", "files.write_file"}
                requires_human = bool(raw.get("requires_human", False)) or bool(risky.intersection(tools))
                steps.append(TaskPlanStep(
                    step_id=str(raw.get("step_id", "")),
                    agent=str(raw["agent"]),
                    description=str(raw.get("description", "")),
                    depends_on=[str(dep) for dep in raw.get("depends_on", [])],
                    allowed_tools=tools,
                    requires_human=requires_human,
                    arguments=dict(raw.get("arguments", {})) if isinstance(raw.get("arguments", {}), dict) else {},
                ))
            return TaskPlan(
                goal=str(data.get("goal", question)),
                route="multi_agent" if str(data.get("route")) == "multi_agent" else "single_agent",
                steps=steps,
                complexity_score=int(data.get("complexity_score", len(steps))),
                metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
            )
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            return None

    def validate(self, plan: TaskPlan) -> tuple[bool, str]:
        """执行前校验节点、依赖、预算和高风险动作。"""
        if len(plan.steps) > plan.max_agents + 1:
            return False, "计划节点数超过预算"
        ids = {step.step_id for step in plan.steps}
        if len(ids) != len(plan.steps):
            return False, "计划包含重复节点"
        risky_tools = {"email.send", "email.save_draft", "files.write_file"}
        for step in plan.steps:
            if any(dep not in ids for dep in step.depends_on):
                return False, f"节点 {step.step_id} 存在缺失依赖"
            if any(tool in risky_tools for tool in step.allowed_tools) and not step.requires_human:
                return False, f"高风险节点 {step.step_id} 未标记人工确认"

        # 拓扑删除法用于检测循环依赖。
        remaining = {step.step_id: set(step.depends_on) for step in plan.steps}
        while remaining:
            ready = [key for key, deps in remaining.items() if not deps]
            if not ready:
                return False, "计划存在循环依赖"
            for key in ready:
                remaining.pop(key)
            for deps in remaining.values():
                deps.difference_update(ready)
        return True, ""
