from __future__ import annotations

from pathlib import Path

import pytest

from deskpilot.tasks import TaskStore
from deskpilot.tasks.compensation import compensate_file_effect, compensate_file_write
from deskpilot.tasks.control import ExecutionControl, execution_scope
from deskpilot.tasks.effects import effects_for_task
from deskpilot.tools.write_tools import write_file
from deskpilot.tools.file_tools import apply_file_move


def committed_write(store: TaskStore, target: Path, content: str, *, overwrite: bool) -> str:
    from deskpilot.tasks.effects import begin_effect, finish_effect
    from deskpilot.tools.tool_registry import ToolResult

    control = ExecutionControl(task_id="task-file", node_id="write")
    control.task_store = store
    arguments = {"path": str(target), "content": content, "overwrite": overwrite}
    with execution_scope(control):
        operation_id = begin_effect(control, "files.write_file", arguments)
        output = write_file(target, content, overwrite=overwrite, confirm=True,
                            safe_roots=[target.parent])
        finish_effect(control, operation_id, ToolResult(True, "files.write_file", output))
    assert output["written"]
    return operation_id


@pytest.mark.parametrize("overwrite", [False, True])
def test_file_compensation_requires_approval_and_preserves_original(tmp_path: Path, overwrite: bool) -> None:
    store = TaskStore(tmp_path / "task.sqlite")
    target = tmp_path / "report.txt"
    if overwrite:
        target.write_text("original", encoding="utf-8")
    operation_id = committed_write(store, target, "generated", overwrite=overwrite)
    effect = effects_for_task(store, "task-file")[0]
    assert effect["operation_id"] == operation_id
    assert effect["state"] == "committed"
    prepared = compensate_file_write(store, operation_id, safe_roots=[tmp_path])
    assert prepared["permission"]["requires_confirmation"]
    assert target.read_text(encoding="utf-8") == "generated"
    done = compensate_file_write(store, operation_id, confirm=True, safe_roots=[tmp_path])
    assert done["restored"] is overwrite
    assert target.read_text(encoding="utf-8") == "original" if overwrite else not target.exists()
    assert effects_for_task(store, "task-file")[0]["state"] == "compensated"
    with pytest.raises(ValueError, match="可补偿状态"):
        compensate_file_write(store, operation_id, confirm=True, safe_roots=[tmp_path])


def test_file_compensation_refuses_modified_target(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "task.sqlite")
    target = tmp_path / "report.txt"
    target.write_text("original", encoding="utf-8")
    operation_id = committed_write(store, target, "generated", overwrite=True)
    target.write_text("user edit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="已被修改"):
        compensate_file_write(store, operation_id, confirm=True, safe_roots=[tmp_path])
    assert target.read_text(encoding="utf-8") == "user edit"
    assert effects_for_task(store, "task-file")[0]["state"] == "committed"


def test_move_compensation_requires_unchanged_paths(tmp_path: Path) -> None:
    from deskpilot.tasks.effects import begin_effect, finish_effect
    from deskpilot.tools.tool_registry import ToolResult

    store = TaskStore(tmp_path / "task.sqlite")
    source, destination = tmp_path / "source.txt", tmp_path / "dest.txt"
    source.write_text("payload", encoding="utf-8")
    control = ExecutionControl(task_id="task-move", node_id="move")
    control.task_store = store
    with execution_scope(control):
        operation_id = begin_effect(control, "files.apply_move", {
            "source": str(source), "destination": str(destination),
        })
        output = apply_file_move(source, destination, confirm=True, safe_roots=[tmp_path])
        finish_effect(control, operation_id, ToolResult(True, "files.apply_move", output))
    assert not source.exists() and destination.read_text(encoding="utf-8") == "payload"
    prepared = compensate_file_effect(store, operation_id, safe_roots=[tmp_path])
    assert prepared["permission"]["requires_confirmation"]
    destination.write_text("user edit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="已变化"):
        compensate_file_effect(store, operation_id, confirm=True, safe_roots=[tmp_path])
    assert destination.read_text(encoding="utf-8") == "user edit"
    destination.write_text("payload", encoding="utf-8")
    result = compensate_file_effect(store, operation_id, confirm=True, safe_roots=[tmp_path])
    assert result["compensated"] and source.read_text(encoding="utf-8") == "payload"
    assert not destination.exists()


def test_interrupted_compensation_requires_manual_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = TaskStore(tmp_path / "task.sqlite")
    target = tmp_path / "report.txt"
    operation_id = committed_write(store, target, "generated", overwrite=False)
    original_unlink = Path.unlink

    def interrupted_unlink(path: Path, *args, **kwargs):
        if path == target:
            raise OSError("injected interruption")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_unlink)
    with pytest.raises(OSError, match="injected interruption"):
        compensate_file_write(store, operation_id, confirm=True, safe_roots=[tmp_path])
    assert target.read_text(encoding="utf-8") == "generated"
    assert effects_for_task(store, "task-file")[0]["state"] == "compensating"
    with pytest.raises(ValueError):
        compensate_file_write(store, operation_id, confirm=True, safe_roots=[tmp_path])
