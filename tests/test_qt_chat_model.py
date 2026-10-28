from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtCore import QObject

from deskpilot.qt_app import ChatMessageModel, DeskPilotBridge, _step_dict


def test_chat_model_updates_only_last_message_content() -> None:
    model = ChatMessageModel()
    model.reset_messages(
        [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": ""},
        ]
    )
    first_index = model.index(0, 0)
    last_index = model.index(1, 0)

    model.update_content(1, "较长回答的第一部分")

    assert model.rowCount() == 2
    assert model.data(first_index, model.ContentRole) == "问题"
    assert model.data(last_index, model.MessageRole) == "assistant"
    assert model.data(last_index, model.ContentRole) == "较长回答的第一部分"


def test_chat_model_reset_replaces_session_transcript() -> None:
    model = ChatMessageModel()
    model.reset_messages([{"role": "assistant", "content": "旧会话"}])
    model.reset_messages([{"role": "assistant", "content": "新会话"}])
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), model.ContentRole) == "新会话"


def test_step_detail_is_bounded_for_qml_layout() -> None:
    step = type("Step", (), {
        "name": "retrieval_trace", "status": "success", "detail": "x" * 20000,
    })()
    item = _step_dict(step)
    assert item["name"] == "retrieval_trace"
    assert len(item["detail"]) < 7000
    assert "UI detail truncated" in item["detail"]


def test_bridge_busy_gate_rejects_overlapping_operations() -> None:
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge._busy = False
    bridge._status = "就绪"
    assert bridge._begin_operation("正在处理") is True
    assert bridge.busy is True
    assert bridge._begin_operation("不应启动") is False
    bridge._end_operation()
    assert bridge.busy is False


def test_resumed_task_reloads_persisted_transcript_instead_of_overwriting(monkeypatch) -> None:
    from types import SimpleNamespace
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge.current_session_id = "session"
    bridge._stream_timer = SimpleNamespace(stop=lambda: None)
    bridge._stream_buffer = ""
    bridge._stream_closed = False
    calls = []
    for name in ("refreshTasks", "_load_session_messages", "_end_operation"):
        monkeypatch.setattr(bridge, name, lambda name=name: calls.append(name))
    monkeypatch.setattr(bridge, "_set_result_panels", lambda result: calls.append("panels"))
    monkeypatch.setattr(bridge, "_set_status", lambda status: calls.append(status))
    bridge._handle_task_result("task", SimpleNamespace(session_id="session", pending_action=None))
    assert calls == ["refreshTasks", "_load_session_messages", "panels", "回答完成", "_end_operation"]
