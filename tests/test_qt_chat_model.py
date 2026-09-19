from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.qt_app import ChatMessageModel


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
