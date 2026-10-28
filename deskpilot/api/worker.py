from __future__ import annotations

import os
from typing import Any

from .service import PrivateTaskService
from ..sandbox import DockerSandboxRunner, SandboxUnavailable


def register_api_handlers(agent: Any) -> None:
    if os.getenv("DESKPILOT_RELEASE_MODE") != "1" or os.getenv("DESKPILOT_API_WORKER") != "1":
        raise RuntimeError("Private API handlers require release-mode dedicated worker")

    def direct(payload: dict[str, Any]) -> dict[str, str]:
        question = str(payload["question"])
        answer = agent.client.chat([
            {"role": "system", "content": (
                "你是 DeskPilot 的私有 API 直接问答助手。只回答通用问题。"
                "你没有工具和文件访问能力，不得声称执行过搜索、读写文件、邮件或命令。"
                "当请求需要外部资料或本地操作时，明确说明当前 API 未开放这些能力。"
            )},
            {"role": "user", "content": question},
        ], temperature=0.2, max_tokens=800)
        if not answer.strip():
            raise RuntimeError("Direct-answer model returned no content")
        return {"answer": answer, "session_id": str(payload["session_id"])}

    def decision(payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        action_id = str(payload["action_id"])
        item = agent.pending_actions.inspect(action_id, session_id)
        if item is None or item.task_id != payload["parent_task_id"]:
            raise ValueError("Approval no longer belongs to this task")
        if item.tool_name not in {"email.send", "email.save_draft"}:
            raise PermissionError("API approval tool is not enabled")
        if PrivateTaskService._digest(item.fingerprint) != payload["preview_digest"]:
            raise ValueError("Approval preview changed")
        if payload["decision"] == "reject":
            if not agent.cancel_pending_action({"action_id": action_id}, session_id=session_id):
                raise ValueError("Approval was already consumed")
            return {"rejected": True}
        result = agent.approve_pending_action({"action_id": action_id}, session_id=session_id)
        if any(step.status in {"failed", "unknown"} for step in result.steps):
            raise RuntimeError("Approved action did not complete; inspect its original task")
        return {"approved": True, "answer": result.answer}

    def sandbox_python(payload: dict[str, Any]) -> dict[str, Any]:
        if os.getenv("DESKPILOT_ENABLE_DOCKER_SANDBOX") != "1":
            raise SandboxUnavailable("Docker sandbox is disabled")
        image = os.getenv("DESKPILOT_DOCKER_IMAGE", "")
        if not image:
            raise SandboxUnavailable("DESKPILOT_DOCKER_IMAGE must be a pinned image digest")
        inputs = {name: value.encode("utf-8") for name, value in payload["inputs"].items()}
        result = DockerSandboxRunner(image=image).run_python(str(payload["code"]), inputs)
        if not result["ok"]:
            reason = "timeout" if result["timed_out"] else ("output limit" if result["output_limited"] else "nonzero exit")
            raise RuntimeError(f"Docker sandbox execution failed ({reason}, returncode={result['returncode']}): "
                               f"{result['stderr'][:512]}")
        return {"answer": result["stdout"], "sandbox": result}

    agent.task_service.handlers.clear()
    agent.task_service.register("core.api_direct", direct)
    agent.task_service.register("core.api_decision", decision)
    if os.getenv("DESKPILOT_ENABLE_DOCKER_SANDBOX") == "1":
        agent.task_service.register("core.api_sandbox_python", sandbox_python)
