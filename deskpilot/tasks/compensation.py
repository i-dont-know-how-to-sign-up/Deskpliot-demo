from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..core.sqlite_utils import sqlite_transaction
from ..tools.permissions import assess_path_write
from ..tools.write_tools import _sha256, _stage_path


def _claim_compensation(store: Any, operation_id: str) -> None:
    # 先落盘补偿状态；进程中断后留待人工核对，不能自动重放回滚。
    with sqlite_transaction(store.path) as connection:
        changed = connection.execute(
            "UPDATE runtime_tool_effects SET state='compensating' WHERE operation_id=? AND state='committed'",
            (operation_id,),
        ).rowcount
    if changed != 1:
        raise RuntimeError("Compensation state changed; manual review required")


def file_effect(store: Any, operation_id: str) -> dict[str, Any] | None:
    with sqlite_transaction(store.path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_tool_effects'"
        ).fetchone()
        if not exists:
            return None
        row = connection.execute(
            "SELECT * FROM runtime_tool_effects WHERE operation_id=?", (operation_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["result"] = json.loads(result.pop("result_json") or "{}")
    return result


def compensate_file_write(
    store: Any, operation_id: str, *, confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    effect = file_effect(store, operation_id)
    if not effect or effect["tool_name"] not in {"files.write_file", "files.write_markdown", "files.write_text",
                                                "files.write_docx", "files.write_pdf"}:
        raise ValueError("没有可补偿的文件写入记录")
    if effect["state"] != "committed" or not effect["result"].get("compensatable"):
        raise ValueError("该文件操作未处于可补偿状态")
    detail = effect["result"]
    target = Path(str(detail["path"]))
    decision = assess_path_write(target, operation="write_file", additional_safe_roots=safe_roots)
    if decision.blocked:
        raise PermissionError("补偿目标受保护，不能修改")
    if not confirm:
        return {"ok": True, "compensated": False, "path": str(target), "permission": {
            "requires_confirmation": True, "blocked": False, "risk_level": "high",
            "reasons": ["补偿会删除本任务新建文件或恢复覆盖前版本；确认前会重新校验目标哈希。"],
        }}
    if not target.is_file() or _sha256(target) != detail.get("after_sha256"):
        raise RuntimeError("目标文件已被修改或删除，拒绝自动补偿")
    before_hash = str(detail.get("before_sha256") or "")
    backup_path = str(detail.get("backup_path") or "")
    backup = Path(backup_path) if backup_path else None
    if before_hash:
        expected_root = (store.path.parent / "effect_backups").resolve()
        if backup is None or backup.resolve().parent != expected_root or not backup.is_file():
            raise RuntimeError("原文件备份缺失或位置不可信，拒绝补偿")
        if _sha256(backup) != before_hash:
            raise RuntimeError("原文件备份哈希不匹配，拒绝补偿")
        _claim_compensation(store, operation_id)
        stage = _stage_path(target)
        try:
            shutil.copy2(backup, stage)
            if _sha256(stage) != before_hash or _sha256(target) != detail["after_sha256"]:
                raise RuntimeError("补偿前文件版本发生变化")
            os.replace(stage, target)
        finally:
            stage.unlink(missing_ok=True)
    else:
        # 新建文件只在内容仍为本次任务提交的版本时删除。
        if _sha256(target) != detail["after_sha256"]:
            raise RuntimeError("补偿前文件版本发生变化")
        _claim_compensation(store, operation_id)
        target.unlink()
    with sqlite_transaction(store.path) as connection:
        changed = connection.execute(
            "UPDATE runtime_tool_effects SET state='compensated' WHERE operation_id=? AND state='compensating'",
            (operation_id,),
        ).rowcount
    if changed != 1:
        raise RuntimeError("补偿已执行，但日志状态发生竞争；需人工核对")
    return {"ok": True, "compensated": True, "path": str(target),
            "restored": bool(before_hash), "operation_id": operation_id}


def compensate_file_effect(
    store: Any, operation_id: str, *, confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    effect = file_effect(store, operation_id)
    if effect and effect["tool_name"] == "files.apply_move":
        return _compensate_file_move(store, effect, confirm=confirm, safe_roots=safe_roots)
    return compensate_file_write(store, operation_id, confirm=confirm, safe_roots=safe_roots)


def _compensate_file_move(
    store: Any, effect: dict[str, Any], *, confirm: bool,
    safe_roots: list[str | Path] | None,
) -> dict[str, Any]:
    if effect["state"] != "committed" or not effect["result"].get("compensatable"):
        raise ValueError("该文件移动未处于可补偿状态")
    detail = effect["result"]
    source, destination = Path(detail["source"]), Path(detail["destination"])
    from ..tools.permissions import assess_file_move
    decision = assess_file_move(destination, source, additional_safe_roots=safe_roots)
    if decision.blocked:
        raise PermissionError("补偿路径受保护，不能逆向移动")
    if not confirm:
        return {"ok": True, "compensated": False, "path": str(destination), "permission": {
            "requires_confirmation": True, "blocked": False, "risk_level": "high",
            "reasons": ["补偿将文件移回原路径；源和目标文件版本必须与提交日志一致。"],
        }}
    if source.exists() or not destination.is_file() or _sha256(destination) != detail.get("after_sha256"):
        raise RuntimeError("移动路径或文件版本已变化，拒绝自动补偿")
    _claim_compensation(store, effect["operation_id"])
    source.parent.mkdir(parents=True, exist_ok=True)
    stage = _stage_path(source)
    linked = False
    try:
        shutil.copy2(destination, stage)
        if _sha256(stage) != detail["before_sha256"] or _sha256(destination) != detail["after_sha256"]:
            raise RuntimeError("逆向移动前文件版本发生变化")
        os.link(stage, source)
        linked = True
        destination.unlink()
    finally:
        stage.unlink(missing_ok=True)
    if not linked:
        raise RuntimeError("逆向移动未完成")
    with sqlite_transaction(store.path) as connection:
        changed = connection.execute(
            "UPDATE runtime_tool_effects SET state='compensated' WHERE operation_id=? AND state='compensating'",
            (effect["operation_id"],),
        ).rowcount
    if changed != 1:
        raise RuntimeError("逆向移动已完成，但日志状态发生竞争；需人工核对")
    return {"ok": True, "compensated": True, "source": str(source),
            "destination": str(destination), "operation_id": effect["operation_id"]}
