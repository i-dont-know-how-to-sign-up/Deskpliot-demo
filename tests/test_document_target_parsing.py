from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.tools.file_tools import resolve_document_path


def test_action_prefix_is_removed_from_document_target() -> None:
    filename = "开发日志.md"
    question = "请读取" + filename
    assert resolve_document_path(question).name == filename


def test_multiple_document_targets_come_from_structured_plan() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    targets = agent._planned_document_targets({
        "steps": [{
            "allowed_tools": ["files.read_document"],
            "arguments": {"paths": ["开发日志.md", "开发计划文档.md"]},
        }],
    })
    assert targets == ["开发日志.md", "开发计划文档.md"]


if __name__ == "__main__":
    test_action_prefix_is_removed_from_document_target()
    test_multiple_document_targets_come_from_structured_plan()
    print("document target parsing passed")
