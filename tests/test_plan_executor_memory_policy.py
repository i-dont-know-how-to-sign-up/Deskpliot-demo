from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.policy import MemoryPolicy
from deskpilot.memory.turn_manager import MemoryTurnManager
from deskpilot.multi_agent.plan_executor import PlanExecutor
from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
from deskpilot.multi_agent.supervisor import SupervisorAgent
from deskpilot.tools.tool_registry import ToolResult


class PlanExecutorTests(unittest.TestCase):
    def test_email_plan_runs_through_supervisor_and_waits_for_approval(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def call_tool(name: str, **kwargs: object) -> ToolResult:
            calls.append((name, kwargs))
            return ToolResult(True, name, {"permission": {"requires_confirmation": True}})

        executor = PlanExecutor(
            tool_call=call_tool,
            research=lambda topic: SimpleNamespace(
                report="上海明日多云。来源：https://example.org/weather",
                artifact_path="",
            ),
            llm_call=lambda prompt: "上海明日多云，请注意天气变化。",
        )
        preview = {
            "goal": "查询天气并准备邮件",
            "steps": [
                {"id": "knowledge", "allowed_tools": ["web.research"]},
                {"id": "communication", "depends_on": ["knowledge"]},
                {"id": "commit", "allowed_tools": ["email.send"]},
            ],
        }
        execution = executor.execute_email_plan(preview, {
            "to": "receiver@example.com",
            "subject": "天气",
            "request": "查询上海明日天气",
            "email_tool": "email.send",
        })

        self.assertEqual(execution.supervisor.status, "waiting_human")
        self.assertEqual([item.step_id for item in execution.supervisor.results], [
            "knowledge", "communication", "commit",
        ])
        self.assertEqual(execution.tool_name, "email.send")
        self.assertEqual(execution.call_arguments["body"], "上海明日多云，请注意天气变化。")
        self.assertNotIn("confirm", execution.call_arguments)
        self.assertEqual(calls[0][0], "email.send")
        self.assertNotIn("confirm", calls[0][1])

    def test_supervisor_retries_and_enforces_tool_budget(self) -> None:
        attempts = 0

        def flaky(_values: dict[str, object]) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary")
            return {"ok": True, "_tool_calls": 2}

        plan = TaskPlan(
            "retry", "multi_agent",
            [TaskPlanStep("knowledge", "knowledge", "retry", handler=flaky, max_retries=1)],
            max_tool_calls=1,
        )
        result = SupervisorAgent().execute(plan, parallel=False)
        self.assertEqual(attempts, 2)
        self.assertEqual(result.status, "failed")
        self.assertIn("预算", result.error)

    def test_supervisor_enforces_elapsed_time_budget(self) -> None:
        plan = TaskPlan(
            "timeout", "multi_agent",
            [TaskPlanStep("knowledge", "knowledge", "slow", handler=lambda values: {"ok": True})],
            max_duration_seconds=1,
        )
        with patch("deskpilot.multi_agent.supervisor.time.monotonic", side_effect=[0.0, 2.0]):
            result = SupervisorAgent().execute(plan, parallel=False)
        self.assertEqual(result.status, "failed")
        self.assertIn("执行时间", result.error)


class MemoryPolicyTests(unittest.TestCase):
    def test_simple_direct_turn_skips_extractor_but_still_checks_compaction(self) -> None:
        extractor = Mock()
        store = Mock()
        store.list_memories.return_value = []
        session_store = Mock()
        compactor = Mock()
        compactor.compact_if_needed.return_value = (False, "")
        manager = MemoryTurnManager(extractor, store, session_store, compactor)

        result = manager.process(
            user_message="什么是 RAG？",
            assistant_message="RAG 是检索增强生成。",
            session_id="sess_1",
            source_message_ids=["u1", "a1"],
            grounded=False,
            steps=[],
        )

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, "low_value_direct_turn")
        extractor.extract.assert_not_called()
        compactor.compact_if_needed.assert_called_once()

    def test_preference_turn_is_extracted(self) -> None:
        extractor = Mock()
        extractor.extract.return_value = [{
            "scope": "user", "memory_type": "preference",
            "content": "用户偏好：回答先给结论", "confidence": 0.9,
            "tags": ["topic:answer_style"],
        }]
        memory = SimpleNamespace(memory_type="preference", to_dict=lambda: {"content": "回答先给结论"})
        store = Mock()
        store.add_memory.return_value = memory
        store.list_memories.return_value = []
        session_store = Mock()
        compactor = Mock()
        compactor.compact_if_needed.return_value = (False, "")
        manager = MemoryTurnManager(extractor, store, session_store, compactor)

        result = manager.process(
            user_message="请记住，我希望以后回答先给结论。",
            assistant_message="已记住。",
            session_id="sess_1",
            source_message_ids=["u1", "a1"],
            grounded=False,
            steps=[],
        )

        self.assertFalse(result.skipped)
        self.assertEqual(result.reason, "explicit_user_state")
        self.assertEqual(result.extracted, 1)
        extractor.extract.assert_called_once()
        session_store.append_session_item.assert_called_once()

    def test_default_ttl_depends_on_memory_type_and_scope(self) -> None:
        policy = MemoryPolicy()
        self.assertIsNone(policy.default_expires_at("user", "preference"))
        self.assertIsNone(policy.default_expires_at("workspace", "decision"))
        self.assertIsNone(policy.default_expires_at("session", "task", ["completed"]))

        now = datetime.now(timezone.utc)
        session_task = datetime.fromisoformat(policy.default_expires_at("session", "task"))
        workspace_fact = datetime.fromisoformat(policy.default_expires_at("workspace", "fact"))
        self.assertGreater((session_task - now).days, 28)
        self.assertLess((session_task - now).days, 31)
        self.assertGreater((workspace_fact - now).days, 363)

    def test_topic_tag_supersedes_only_same_topic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deskpilot_memory_policy_") as folder:
            root = Path(folder)
            store = MemoryStore(
                root / "indexes" / "memory.sqlite",
                root / "workspace",
                vector_provider="sqlite",
            )
            old_style = store.add_memory(
                "user", "preference", "回答采用详细风格", confidence=0.95,
                tags=["topic:answer_style"],
            )
            unrelated = store.add_memory(
                "user", "preference", "界面采用深色主题", confidence=0.95,
                tags=["topic:ui_theme"],
            )
            new_style = store.add_memory(
                "user", "preference", "回答采用简洁风格", confidence=0.95,
                tags=["topic:answer_style"],
            )

            self.assertEqual(store.get_memory(old_style.memory_id).status, "superseded")
            self.assertEqual(store.get_memory(new_style.memory_id).status, "active")
            self.assertEqual(store.get_memory(unrelated.memory_id).status, "active")


if __name__ == "__main__":
    unittest.main()
