from __future__ import annotations

import email
from email import policy
from typing import Any

from ..core.sqlite_utils import sqlite_transaction
from .compensation import file_effect
from .models import json_data


def reconcile_email_effect(store: Any, provider: Any, operation_id: str) -> dict[str, Any]:
    effect = file_effect(store, operation_id)
    if not effect or effect["tool_name"] != "email.send" or effect["state"] not in {"dispatched", "unknown", "committed"}:
        raise ValueError("没有可核对的邮件发送记录")
    message_id = str(effect["result"].get("message_id") or "")
    if not message_id:
        snapshot = store.path.parent / "email_effects" / f"{operation_id}.eml"
        if snapshot.is_file():
            message_id = str(email.message_from_bytes(snapshot.read_bytes(), policy=policy.default).get("Message-ID") or "")
    if not message_id:
        return {"status": "manual_review", "reason": "缺少 Message-ID，不能自动核对或补副本"}
    finder = getattr(provider, "find_sent_copy", None)
    if not callable(finder):
        return {"status": "manual_review", "reason": "当前邮箱后端不支持 Message-ID 核对"}
    found = bool(finder(message_id))
    if effect["state"] in {"unknown", "dispatched"} and not found:
        return {"status": "unknown", "message_id": message_id,
                "reason": "已发送箱未找到副本；这不能证明 SMTP 未投递，禁止自动重发"}
    detail = dict(effect["result"])
    if effect["state"] in {"unknown", "dispatched"}:
        detail.update(message_id=message_id, sent_copy_saved=True, reconciled_via_imap=True)
        new_state = "committed"
        action = "found_sent_copy"
    elif found:
        detail["sent_copy_saved"] = True
        new_state = "committed"
        action = "already_present"
    elif detail.get("sent_copy_saved") is False:
        repair = getattr(provider, "repair_sent_copy", None)
        if not callable(repair):
            return {"status": "manual_review", "reason": "当前邮箱后端不支持已发送副本补写"}
        repair(operation_id, message_id, store.path)
        detail["sent_copy_saved"] = True
        new_state = "committed"
        action = "sent_copy_repaired"
    else:
        return {"status": "manual_review", "reason": "日志未标记副本缺失，未执行补写"}
    with sqlite_transaction(store.path) as connection:
        changed = connection.execute(
            "UPDATE runtime_tool_effects SET state=?,result_json=? WHERE operation_id=? AND state=?",
            (new_state, json_data(detail), operation_id, effect["state"]),
        ).rowcount
    if changed != 1:
        raise RuntimeError("邮件核对时副作用状态已变化，请刷新任务")
    return {"status": "verified", "action": action, "message_id": message_id,
            "sent_copy_saved": True}
