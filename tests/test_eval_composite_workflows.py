from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import OpenAICompatibleClient, local_hash_embedding
from deskpilot.core.models import AgentStep
from deskpilot.core.runtime import PlanAndExecuteRuntime
from deskpilot.intent.validator import SlotValidator
from deskpilot.rag.web_research import ResearchResult
from deskpilot.tools.file_tools import resolve_document_path
from deskpilot.tools.tool_registry import ToolResult
from eval.run_eval import DATASET, load_cases, make_agent, prepare_case, run_case
from eval.scoring import score_case


class CompositeWorkflowTests(unittest.TestCase):
    def test_dict_tool_schema_accepts_extra_slots(self) -> None:
        schema = {"name": "files.write_file", "parameters": [
            {"name": "path", "type": "string", "required": True},
        ]}
        result = SlotValidator().validate(schema, {"path": "example.txt", "content": "text"})
        self.assertTrue(result.ok)
        self.assertEqual(result.arguments, {"path": "example.txt", "content": "text"})

    def test_indexed_fixture_resolves_as_local_document(self) -> None:
        case = next(item for item in load_cases(DATASET) if item["id"] == "rag_001")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent = make_agent(root)
            with patch.object(agent.index, "add_file") as add:
                prepare_case(case, root, agent)
            expected = root / "workspace" / "eval" / "dataset" / "fixtures" / "docs" / "deskpilot_intro.md"
            self.assertEqual(add.call_args.args[0], expected)
            previous = Path.cwd()
            try:
                os.chdir(agent.workspace_root)
                self.assertEqual(resolve_document_path("deskpilot_intro.md", [agent.workspace_root]), expected)
                self.assertEqual(resolve_document_path("eval/dataset/fixtures/docs/deskpilot_intro.md", [agent.workspace_root]), expected)
            finally:
                os.chdir(previous)

    def test_new_composite_cases_are_isolated_in_offline_mode(self) -> None:
        cases = [case for case in load_cases(DATASET) if case["id"].startswith("composite_")]
        self.assertEqual(len(cases), 9)
        for case in cases:
            self.assertEqual(run_case(case, "offline")["status"], "skipped")

    def test_offline_does_not_use_configured_api_and_restores_environment(self) -> None:
        case = next(item for item in load_cases(DATASET) if item["id"] == "rag_001")
        before = os.environ.get("TOOL_SAFE_ROOTS")
        with patch.object(OpenAICompatibleClient, "_post_json", side_effect=AssertionError("network called")) as post:
            result = run_case(case, "offline")
        post.assert_not_called()
        self.assertNotEqual(result["status"], "error")
        self.assertEqual(result["token_usage"]["reported_calls"], 0)
        self.assertEqual(os.environ.get("TOOL_SAFE_ROOTS"), before)

    def test_draft_and_send_use_distinct_tools_after_research(self) -> None:
        for tool in ("email.save_draft", "email.send"):
            with self.subTest(tool=tool):
                agent = DocumentQAAgent.__new__(DocumentQAAgent)
                agent.runtime = PlanAndExecuteRuntime()
                agent.web_research_agent = SimpleNamespace(research=lambda topic: ResearchResult(
                    topic, "上海天气：明日多云。来源：https://example.org/weather", [], [], [], "", False,
                ))
                agent.client = SimpleNamespace(chat=lambda *args, **kwargs: "明日多云，详情见来源。")
                called = []

                def call(name, **kwargs):
                    called.append((name, kwargs))
                    return ToolResult(True, name, {"permission": {"requires_confirmation": True}})

                agent.tool_registry = SimpleNamespace(call=call)
                agent._finalize_generic_tool_result = lambda **kwargs: kwargs
                result = agent._answer_planned_task(
                    "先查上海天气再保存草稿" if tool.endswith("save_draft") else "先查上海天气再发送",
                    {"steps": [{"id": "knowledge", "arguments": {"query": "上海天气"}},
                               {"id": "commit", "allowed_tools": [tool], "arguments": {
                                   "to": ["test@example.com"], "subject": "上海天气", "request": "上海天气"}}]},
                    [], "s", "m", object(),
                )
                self.assertEqual(called[0][0], tool)
                self.assertEqual(result["tool_name"], tool)
                self.assertNotIn("confirm", called[0][1])
                self.assertIn("多云", called[0][1]["body"])
                # Supervisor 接管执行后会在确认前写入节点级审计步骤；这里验证关键步骤
                # 的先后关系，而不是把列表末尾位置绑定到旧执行器的内部实现。
                names = [step.name for step in result["steps"]]
                required = [
                    "plan_executor", "research_topic", "compose_email_body",
                    "supervisor:commit", "supervisor", "request_email_confirmation",
                ]
                positions = [names.index(name) for name in required]
                self.assertEqual(positions, sorted(positions))

    def test_composite_scoring_needs_actual_pending_tool(self) -> None:
        case = {"id": "draft", "subset": "Composite-Workflow", "expected": {
            "pending_tool": "email.save_draft", "requires_confirmation": True,
            "pending_arguments_contains": {"body": "多云"},
        }}
        result = SimpleNamespace(answer="准备好了", steps=[AgentStep("plan_task", "success", "")],
                                 pending_action={"tool_name": "email.send", "kwargs": {"body": "多云"}})
        self.assertEqual(score_case(case, result)["status"], "failed")
        result.pending_action["tool_name"] = "email.save_draft"
        self.assertEqual(score_case(case, result)["status"], "passed")

    def test_mocked_api_mode_executes_composite_draft_end_to_end(self) -> None:
        cases = {case["id"]: case for case in load_cases(DATASET)}
        for case_id, tool, subject in (("composite_006", "email.save_draft", "上海天气"),
                                       ("composite_007", "email.send", "上海天气"),
                                       ("multi_001", "email.send", "上海天气"),
                                       ("multi_002", "email.save_draft", "test3"),
                                       ("multi_exec_001", "email.send", "天气提醒")):
            with self.subTest(case_id=case_id):
                def chat(client, messages, **kwargs):
                    system = str(messages[0].get("content", ""))
                    if "intent router" in system.lower():
                        return '{"mode":"plan_task","needs_index_catalog":false,"requires_file_output":false}'
                    if "任务规划器" in system:
                        return ('{"goal":"draft","route":"multi_agent","complexity_score":8,"steps":['
                                '{"step_id":"knowledge","agent":"knowledge","allowed_tools":["web.research"],'
                                '"arguments":{"query":"上海明天的天气"}},'
                                '{"step_id":"commit","agent":"communication","depends_on":["knowledge"],'
                                '"allowed_tools":["' + tool + '"],"requires_human":true,'
                                '"arguments":{"to":["test@example.com"],"subject":"' + subject + '",'
                                '"request":"上海明天的天气"}}]}')
                    if "邮件助手" in system:
                        return "明日多云，18-24度。来源：https://example.org/weather"
                    return ""

                with patch.object(OpenAICompatibleClient, "chat", chat):
                    scored = run_case(cases[case_id], "api")
                self.assertEqual(scored["status"], "passed", scored.get("checks"))
                self.assertEqual(scored["checks"]["pending_tool_ok"], True)

    def test_mocked_search_to_file_and_failure_boundaries(self) -> None:
        cases = {item["id"]: item for item in load_cases(DATASET)}

        def chat(client, messages, **kwargs):
            system = str(messages[0].get("content", ""))
            prompt = str(messages[-1].get("content", ""))
            if "intent router" in system.lower():
                return '{"mode":"plan_task","requires_file_output":true}'
            if "任务规划器" in system:
                target = "existing.txt" if "existing.txt" in prompt else "weather.txt"
                return ('{"goal":"file","route":"multi_agent","complexity_score":8,"steps":['
                        '{"step_id":"knowledge","agent":"knowledge","allowed_tools":["web.search"],'
                        '"arguments":{"query":"上海明天的天气"}},'
                        '{"step_id":"commit","agent":"communication","depends_on":["knowledge"],'
                        '"allowed_tools":["files.write_file"],"requires_human":true,'
                        '"arguments":{"path":"' + target + '"}}]}')
            return ""

        with patch.object(OpenAICompatibleClient, "chat", chat):
            for case_id in ("composite_003", "composite_005", "composite_008", "composite_009"):
                with self.subTest(case_id=case_id):
                    result = run_case(cases[case_id], "api")
                    self.assertEqual(result["status"], "passed", result.get("checks"))

    def test_indexed_fixture_supports_rag_with_mocked_model(self) -> None:
        case = next(item for item in load_cases(DATASET) if item["id"] == "rag_001")

        def chat(client, messages, **kwargs):
            if "intent router" in str(messages[0].get("content", "")).lower():
                return ('{"mode":"tool_call","tool_name":"knowledge.search",'
                        '"arguments":{"query":"RAG 索引流程 Chunk embedding"}}')
            return "文档先解析并切分为 Chunk，再生成 embedding 建立索引。"

        with patch.object(OpenAICompatibleClient, "chat", chat), patch.object(
            OpenAICompatibleClient, "embed", side_effect=lambda texts: [local_hash_embedding(t) for t in texts]
        ):
            result = run_case(case, "api")
        self.assertEqual(result["status"], "passed", result.get("checks"))
        self.assertGreater(result["checks"]["evidence_count"], 0)

    def test_optional_local_pdf_cases_have_valid_schema(self) -> None:
        local = load_cases(DATASET.parent / "local_multimodal_cases.jsonl")
        self.assertEqual(len(local), 3)
        for case in local:
            self.assertTrue(case["runtime"]["requires_local_files"])
            self.assertTrue(case["runtime"]["requires_llm"])
            self.assertEqual(run_case(case, "offline")["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
