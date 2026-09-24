from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import AgentStep, Chunk, Document
from deskpilot.intent.router import IntentRouter
from deskpilot.mcp.email_mcp import _html_to_text
from deskpilot.memory.policy import MemoryPolicy
from deskpilot.multi_agent.plan_executor import PlanExecutor
from deskpilot.multi_agent.planner import PlannerAgent
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.permissions import assess_command
from deskpilot.tools.tool_registry import ToolResult


class Debug3RegressionTests(unittest.TestCase):
    def test_readonly_powershell_count_pipeline_needs_no_confirmation(self) -> None:
        decision = assess_command("(Get-ChildItem -Filter *.py -File | Measure-Object).Count", cwd=Path.cwd())
        self.assertTrue(decision.allowed_without_confirmation)
        self.assertFalse(decision.requires_confirmation)
        self.assertEqual(decision.risk_level, "low")

    def test_write_command_hidden_after_readonly_pipeline_is_not_allowed(self) -> None:
        decision = assess_command("Get-ChildItem | Remove-Item", cwd=Path.cwd())
        self.assertFalse(decision.allowed_without_confirmation)

    def test_planner_normalizes_legacy_read_file_tool_name(self) -> None:
        planner = PlannerAgent(llm_call=lambda prompt: (
            '{"goal":"mail","route":"multi_agent","complexity_score":5,"steps":['
            '{"step_id":"read_file","agent":"knowledge","allowed_tools":["files.read_file"],'
            '"arguments":{"paths":["test.txt"]}},'
            '{"step_id":"send_email","agent":"communication","depends_on":["read_file"],'
            '"allowed_tools":["email.send"],"requires_human":true,'
            '"arguments":{"to":["a@example.com"],"subject":"test","request":"介绍",'
            '"attachment_paths":["test.txt"]}}]}'
        ))
        plan = planner.build_plan("读取 test.txt 并作为附件发送")
        self.assertEqual(plan.steps[0].allowed_tools, ["files.read_document"])
        self.assertTrue(planner.validate(plan)[0])

    def test_plan_executor_supports_arbitrary_ids_and_existing_attachment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deskpilot_attachment_") as folder:
            attachment = Path(folder) / "test.txt"
            attachment.write_text("床前明月光，疑是地上霜。", encoding="utf-8")
            calls: list[tuple[str, dict[str, object]]] = []

            def call(name: str, **kwargs: object) -> ToolResult:
                calls.append((name, kwargs))
                if name == "files.read_document":
                    return ToolResult(True, name, {
                        "path": str(attachment), "content": attachment.read_text(encoding="utf-8"),
                    })
                return ToolResult(True, name, {"permission": {"requires_confirmation": True}})

            executor = PlanExecutor(
                tool_call=call,
                research=lambda topic: SimpleNamespace(report="", artifact_path=""),
                llm_call=lambda prompt: "这是一首借月色表达思乡之情的诗。",
            )
            result = executor.execute_email_plan({"steps": [
                {"id": "read_file", "agent": "knowledge", "allowed_tools": ["files.read_document"],
                 "arguments": {"paths": [str(attachment)]}},
                {"id": "send_email", "agent": "communication", "depends_on": ["read_file"],
                 "allowed_tools": ["email.send"]},
            ]}, {
                "to": "a@example.com", "subject": "静夜思", "request": "介绍诗歌",
                "email_tool": "email.send", "attachment_paths": [str(attachment)],
            })
            self.assertEqual(result.supervisor.status, "waiting_human")
            self.assertEqual([item.step_id for item in result.supervisor.results], [
                "read_file", "communication", "send_email",
            ])
            self.assertEqual(result.call_arguments["attachment_paths"], [str(attachment)])
            self.assertEqual(calls[-1][0], "email.send")

    def test_plan_recognizers_use_tools_not_fixed_node_ids(self) -> None:
        plan = {"steps": [
            {"id": "knowledge_1", "allowed_tools": ["web.search"],
             "arguments": {"query": "上海天气"}},
            {"id": "commit_2", "depends_on": ["knowledge_1"],
             "allowed_tools": ["files.write_file"], "arguments": {"path": "weather.txt"}},
        ]}
        self.assertEqual(DocumentQAAgent._planned_web_file_request(plan), {
            "query": "上海天气", "path": "weather.txt",
        })
        index_plan = {"steps": [{
            "id": "knowledge_9", "allowed_tools": ["knowledge.search"],
            "arguments": {"query": "Falcon", "doc_ids": ["a", "b"]},
        }]}
        self.assertEqual(DocumentQAAgent._planned_index_report(index_plan)["doc_ids"], ["a", "b"])

    def test_conflicting_output_paths_request_clarification(self) -> None:
        plan = {"steps": [
            {"id": "knowledge_1", "allowed_tools": ["web.search"], "arguments": {"query": "天气"}},
            {"id": "commit_1", "allowed_tools": ["files.write_file"],
             "arguments": {"path": "weather2.txt"}},
            {"id": "commit_2", "depends_on": ["knowledge_1"], "allowed_tools": ["files.write_file"],
             "arguments": {"path": "weather.txt"}},
        ]}
        request = DocumentQAAgent._planned_web_file_request(plan)
        self.assertIn("多个不同的目标文件", request["error"])

    def test_explicit_knowledge_scope_cannot_be_direct_answer(self) -> None:
        client = SimpleNamespace(chat=lambda *args, **kwargs: (
            '{"mode":"direct_answer","reason":"general","knowledge_scope":"general"}'
        ))
        decision = IntentRouter(client).route(
            "根据当前知识库中的内容列出常用模型",
            [{"name": "knowledge.search", "parameters": [
                {"name": "query", "type": "string", "required": True},
            ]}],
            index_hint=[],
        )
        self.assertEqual(decision.mode, "plan_task")
        self.assertTrue(decision.needs_index_catalog)
        self.assertEqual(decision.knowledge_scope, "local_index")

    def test_document_acronym_alias_matches_old_index_without_reindex(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deskpilot_alias_") as folder:
            index = DocumentIndex(Path(folder) / "index.json")
            document = Document(
                "doc_ram", str(Path(folder) / "ram.md"), "recognize_anything_workflow.md", "md",
                "The Recognize Anything Model (RAM) uses an image encoder and recognition decoder.",
            )
            index.documents[document.doc_id] = document
            index.chunks["chunk_ram"] = Chunk(
                "chunk_ram", document.doc_id, document.content,
                "recognize_anything_workflow.md#Architecture", 0,
            )
            hints = index.retrieval_hint("RAM 的工作流程是什么样的")
            self.assertEqual(hints[0]["doc_id"], "doc_ram")

    def test_file_read_mode_and_email_privacy_gate(self) -> None:
        self.assertEqual(DocumentQAAgent._local_document_read_mode("内容是什么", ""), "verbatim")
        self.assertEqual(DocumentQAAgent._local_document_read_mode("请总结内容", ""), "summary")
        should_extract, reason = MemoryPolicy().should_extract(
            "查看两封未读邮件", grounded=False,
            steps=[AgentStep("execute_tool", "success", "email.list_messages")],
        )
        self.assertFalse(should_extract)
        self.assertEqual(reason, "low_value_direct_turn")

    def test_html_mail_comments_and_tracking_links_are_removed(self) -> None:
        text = _html_to_text(
            '<!-- hidden --><p>正文</p><div>https://example.com/path?' + ('x' * 350) + '</div>'
        )
        self.assertEqual(text, "正文")


if __name__ == "__main__":
    unittest.main()
