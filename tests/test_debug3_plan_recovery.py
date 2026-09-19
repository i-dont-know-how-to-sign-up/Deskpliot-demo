from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import Chunk, Document
from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.intent.schemas import IntentDecision
from deskpilot.multi_agent.planner import PlannerAgent
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.write_tools import write_file


REPORT = "请总结一下当前索引中的多模态论文，并输出一份技术报告放在当前目录下，报告中需要有引用来源"
WEATHER = "在桌面创建一个weather.txt文件，然后搜索‘上海明天的天气’，把搜索结果写入到weather.txt中"


class Debug3PlanRecoveryTests(unittest.TestCase):
    def test_fallback_plans_have_safe_dependency_chain(self) -> None:
        planner = PlannerAgent(llm_call=lambda _: "")
        report = planner.build_plan(REPORT, indexed_documents=[{"doc_id": "paper", "title": "多模态论文.pdf"}])
        weather = planner.build_plan(WEATHER)
        for plan in (report, weather):
            self.assertEqual(planner.validate(plan), (True, ""))
            self.assertIn("files.write_file", plan.steps[-1].allowed_tools)
            self.assertTrue(plan.steps[-1].requires_human)
            self.assertIn("knowledge", plan.steps[-1].depends_on + plan.steps[1].depends_on)
        self.assertEqual(report.steps[0].allowed_tools, ["knowledge.search"])
        self.assertEqual(weather.steps[0].allowed_tools, ["web.search"])
        self.assertEqual(weather.steps[0].arguments["query"], "上海明天的天气")

    def test_both_debug_requests_reach_execution_when_planner_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = DocumentIndex(root / "index.json")
            index.documents["paper"] = Document("paper", str(root / "paper.pdf"), "多模态论文.pdf", "pdf", "paper")
            agent = DocumentQAAgent(index)
            agent.workspace_root = root
            cases = ((REPORT, IntentDecision(mode="plan_task", needs_index_catalog=True,
                                             requires_file_output=True), "_answer_index_report"),
                     (WEATHER, IntentDecision(mode="plan_task", requires_file_output=True),
                      "_answer_web_file_request"))
            for question, decision, handler in cases:
                with self.subTest(question=question), patch.object(agent.client, "chat", return_value=""), patch.object(
                    agent.intent_router, "route", return_value=decision
                ), patch.object(agent, handler, return_value="dispatched") as dispatched:
                    result = agent.answer(question)
                    self.assertEqual(result, "dispatched", [step.name for step in getattr(result, "steps", [])])
                    self.assertEqual(dispatched.call_count, 1)

    def test_weather_search_prepares_desktop_write_without_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            desktop = root / "Desktop"
            desktop.mkdir()
            agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
            agent.workspace_root = root / "workspace"
            agent.workspace_root.mkdir()
            calls = []

            def call(name, **kwargs):
                calls.append((name, kwargs))
                if name == "web.search":
                    output = [{"title": "上海天气", "url": "https://example.org/weather", "snippet": "明日多云"}]
                else:
                    output = write_file(**kwargs, safe_roots=[agent.workspace_root])
                return type("ToolResult", (), {"ok": True, "output": output, "error": ""})()

            with patch.object(agent.client, "chat", return_value=""), patch.object(
                agent.intent_router, "route", return_value=IntentDecision(mode="plan_task", requires_file_output=True)
            ), patch.object(agent, "_desktop_directory", return_value=desktop), patch.object(
                agent.tool_registry, "call", side_effect=call
            ):
                response = agent.answer(WEATHER)
            self.assertEqual([name for name, _ in calls], ["web.search", "files.write_file"])
            self.assertEqual(calls[0][1]["query"], "上海明天的天气")
            self.assertEqual(calls[1][1]["path"], str(desktop / "weather.txt"))
            self.assertIn("https://example.org/weather", calls[1][1]["content"])
            self.assertNotIn("写入到weather.txt中", calls[1][1]["content"])
            self.assertEqual(response.pending_action["tool_name"], "files.write_file")
            self.assertFalse((desktop / "weather.txt").exists())

    def test_report_is_written_with_two_indexed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = DocumentIndex(root / "index.json")
            for doc_id, title, content in (("one", "BLIP.pdf", "BLIP aligns images and text."),
                                           ("two", "BLIP-2.pdf", "BLIP-2 connects vision to language.")):
                index.documents[doc_id] = Document(doc_id, str(root / title), title, "pdf", content)
                chunk = Chunk(doc_id + "_0", doc_id, content, title + "#chunk-0", 0)
                chunk.embedding = local_hash_embedding(content)
                index.chunks[chunk.chunk_id] = chunk
            agent = DocumentQAAgent(index)
            agent.workspace_root = root

            def chat(messages, **kwargs):
                if "索引文档选择器" in str(messages):
                    return '{"doc_ids":["one","two"]}'
                if "技术报告" in str(messages) and "索引证据" in str(messages):
                    return "# 技术报告\n\nBLIP 图文对齐 [1]；BLIP-2 使用视觉语言模型 [2]。"
                return ""  # 模拟 Planner 的 LLM 输出不可解析

            def call(name, **kwargs):
                self.assertEqual(name, "files.write_file")
                output = write_file(**kwargs, safe_roots=[root])
                return type("ToolResult", (), {"ok": True, "output": output, "error": ""})()

            with patch.object(agent.client, "chat", side_effect=chat), patch.object(
                agent.intent_router, "route", return_value=IntentDecision(
                    mode="plan_task", needs_index_catalog=True, requires_file_output=True)
            ), patch.object(agent.tool_registry, "call", side_effect=call):
                response = agent.answer(REPORT)
            report = root / "技术报告.md"
            self.assertTrue(report.exists(), response.answer)
            contents = report.read_text(encoding="utf-8")
            self.assertIn("BLIP.pdf", contents)
            self.assertIn("BLIP-2.pdf", contents)
            self.assertIn("[1]", contents)


if __name__ == "__main__":
    unittest.main()
