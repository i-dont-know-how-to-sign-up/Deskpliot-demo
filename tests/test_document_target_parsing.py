from __future__ import annotations

import sys
from tempfile import TemporaryDirectory
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.tools.file_tools import resolve_document_path


def test_action_prefix_is_removed_from_document_target() -> None:
    filename = "开发日志.md"
    question = "请读取" + filename
    # 测试必须自带夹具，不能依赖不会提交到仓库的本地开发日志。
    with TemporaryDirectory() as temp_dir:
        document = Path(temp_dir) / filename
        document.write_text("测试文档", encoding="utf-8")

        resolved = resolve_document_path(question, search_roots=[Path(temp_dir)])

    assert resolved == document.resolve()


def test_multiple_document_targets_come_from_structured_plan() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    targets = agent._planned_document_targets({
        "steps": [{
            "allowed_tools": ["files.read_document"],
            "arguments": {"paths": ["开发日志.md", "开发计划文档.md"]},
        }],
    })
    assert targets == ["开发日志.md", "开发计划文档.md"]


def test_report_filename_is_not_polluted_by_action_description() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    raw = "再将带引用来源的结论保存到当前目录的技术报告.md"

    assert agent._normalize_file_write_target(raw) == "技术报告.md"


if __name__ == "__main__":
    test_action_prefix_is_removed_from_document_target()
    test_multiple_document_targets_come_from_structured_plan()
    test_report_filename_is_not_polluted_by_action_description()
    print("document target parsing passed")
