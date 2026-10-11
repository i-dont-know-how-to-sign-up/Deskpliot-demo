from __future__ import annotations

from pathlib import Path
import pytest

from eval.reliability_scenarios import SCENARIOS, run_scenario
from deskpilot.tasks.service import TaskService
from deskpilot.tasks.store import TaskStore
from deskpilot.tasks.control import get_execution_control


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reliability_contracts(scenario: str, tmp_path: Path) -> None:
    assert run_scenario(scenario, tmp_path)["passed"]


def test_durable_queue_reopens_and_attributes_usage(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    first = TaskService(store)
    first.register("fixture", lambda values: values)
    task_id = first.submit("fixture", {"question": "fixture task"}, session_id="session")
    reopened = TaskService(TaskStore(store.path))

    def run(values):
        control = get_execution_control()
        assert control.task_id == task_id
        control.record_tool("fixture")
        control.record_attempt()
        control.record_usage({"total_tokens": 7})
        return {"answer": "fixture completed"}

    reopened.register("fixture", run)
    assert reopened.run_once()
    assert store.get_task(task_id)["status"] == "succeeded"
    assert reopened.descriptor(task_id)["usage"]["total_tokens"] == 7
    assert store.load_output(task_id)["answer"] == "fixture completed"
    assert not reopened.run_once()
    reopened.close()


def test_queued_pause_and_cancel_do_not_dispatch(tmp_path: Path) -> None:
    service = TaskService(TaskStore(tmp_path / "tasks.sqlite3"))
    calls = []
    service.register("fixture", lambda values: calls.append(values))
    task_id = service.submit("fixture", {})
    service.request(task_id, "pause")
    assert service.store.get_task(task_id)["status"] == "paused"
    assert not service.run_once()
    service.request(task_id, "cancel")
    assert service.store.get_task(task_id)["status"] == "cancelled"
    assert not calls
    service.close()


def test_pdf_progress_reopens_and_tracks_reused_pages(tmp_path: Path) -> None:
    from deskpilot.tasks.control import ExecutionControl
    service = TaskService(TaskStore(tmp_path / "tasks.sqlite3"))
    service.register("fixture", lambda values: values)
    task_id = service.submit("fixture", {})
    control = ExecutionControl(task_id=task_id)
    service.initialize_metrics(control)
    control._event("pdf_started", total_pages=3)
    control._event("pdf_checkpoint", page=1)
    service.persist_metrics(control)
    assert service.descriptor(task_id)["pdf_progress"] == {"total": 3, "completed": 1, "reused": 0}
    resumed = ExecutionControl(task_id=task_id)
    service.initialize_metrics(resumed)
    resumed._event("pdf_started", total_pages=3)
    resumed._event("pdf_checkpoint", page=1, reused=True)
    resumed._event("pdf_checkpoint", page=2)
    service._active[task_id] = resumed
    assert service.descriptor(task_id)["pdf_progress"] == {"total": 3, "completed": 2, "reused": 1}
    service.persist_metrics(resumed)
    # 活跃与持久观测不应把同一批次的复用次数叠加两遍。
    assert service.descriptor(task_id)["pdf_progress"]["reused"] == 1
    service._active.clear()
    assert TaskService(service.store).descriptor(task_id)["pdf_progress"]["reused"] == 1
    service.close()


def _isolated_agent(tmp_path: Path, monkeypatch):
    import deskpilot.core.agent as module
    from deskpilot.memory.session_store import SessionStore
    from deskpilot.memory.memory_store import MemoryStore
    from deskpilot.multimodal.service import MultimodalService
    from deskpilot.rag.vector_index import DocumentIndex
    for key in ("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY", "VISION_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("ALLOW_LOCAL_FALLBACK", "true")
    monkeypatch.setenv("OCR_PROVIDER", "disabled")
    monkeypatch.setenv("MEMORY_VECTOR_PROVIDER", "sqlite")
    monkeypatch.setattr(module, "SessionStore", lambda: SessionStore(tmp_path / "sessions"))
    monkeypatch.setattr(module, "MemoryStore", lambda: MemoryStore(tmp_path / "memory.sqlite3", tmp_path / "memory"))
    monkeypatch.setattr(module, "MultimodalService", lambda: MultimodalService(tmp_path / "multimodal"))
    return module.DocumentQAAgent(DocumentIndex(tmp_path / "index.json"))


def test_demo_queue_approval_reopens_and_completes_original_task(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.intent.schemas import IntentDecision
    from deskpilot.mcp.email_mcp import EmailMCPService, MockEmailProvider
    from deskpilot.tools.tool_registry import build_default_tool_registry
    agent = _isolated_agent(tmp_path, monkeypatch)
    session = agent.session_store.create_session()
    provider = MockEmailProvider()
    agent.tool_registry = build_default_tool_registry(agent.index, email_service=EmailMCPService(provider))
    monkeypatch.setattr(agent.intent_router, "route", lambda *a, **kw: IntentDecision(
        mode="tool_call", tool_name="email.save_draft", arguments={
            "to": ["recipient@example.com"], "subject": "fixture", "body": "fixture body"}))
    task_id = agent.task_service.submit("core.answer", {
        "question": "保存这封测试草稿", "session_id": session.session_id}, session_id=session.session_id)
    delivered = []
    agent.task_service.on_result = lambda task, result: delivered.append(result)
    assert agent.task_service.run_once()
    assert delivered and delivered[0].pending_action
    assert agent.task_ledger.store.get_task(task_id)["status"] == "waiting_human"
    # 模拟重建 Agent：不保留内存审批令牌，恢复同一 SQLite 和会话目录。
    reopened = _isolated_agent(tmp_path, monkeypatch)
    reopened.tool_registry = build_default_tool_registry(reopened.index, email_service=EmailMCPService(provider))
    restored = reopened.pending_actions.list_pending(task_id=task_id)[0]
    result = reopened.approve_pending_action(reopened.pending_actions.public(restored), session_id=session.session_id)
    assert "保存邮件草稿" in result.answer
    assert reopened.task_ledger.store.get_task(task_id)["status"] == "succeeded"
    assert len(provider.drafts) == 1
    with pytest.raises(ValueError):
        reopened.approve_pending_action(delivered[0].pending_action, session_id=session.session_id)
    agent.task_service.close()
    reopened.task_service.close()


def test_vision_attempt_usage_is_attributed(monkeypatch, tmp_path: Path) -> None:
    import io
    import json
    import urllib.request
    from PIL import Image
    from deskpilot.multimodal.providers.vision_llm import VisionLLMProvider
    from deskpilot.tasks.control import ExecutionControl, execution_scope
    image = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "white").save(image)
    class Response(io.BytesIO):
        pass
    body = json.dumps({"choices": [{"message": {"content": "fixture [图片1]"}}],
                       "usage": {"total_tokens": 12}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response(body))
    provider = VisionLLMProvider()
    provider.api_key = "fixture"
    provider.allow_cloud = True
    control = ExecutionControl(task_id="vision", max_tokens=8000)
    with execution_scope(control, node_id="answer", attempt_id="vision_attempt"):
        assert provider.answer("fixture", [image])
    assert control.usage["api_attempts"] == 1
    assert control.usage["total_tokens"] == 12
    assert control.usage["reserved_tokens"] == 0
    assert all(event["node_id"] == "answer" for event in control.events)
