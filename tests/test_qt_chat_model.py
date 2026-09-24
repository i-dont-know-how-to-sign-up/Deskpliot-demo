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
