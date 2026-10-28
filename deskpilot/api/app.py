from __future__ import annotations

import asyncio
import json
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..tasks.models import TERMINAL_STATES
from .security import AuthenticationError, AuthorizationError, TokenCatalog
from .service import APIServiceError, PrivateTaskService


class SubmitTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=120)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(min_length=8, max_length=128)
    preview_digest: str = Field(min_length=64, max_length=64)


class SandboxSubmit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=8192)
    inputs: dict[str, str] = Field(default_factory=dict)


def create_app(service: PrivateTaskService, tokens: TokenCatalog) -> FastAPI:
    app = FastAPI(title="DeskPilot Private API", version="0.1.0", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > 16_384):
                return JSONResponse({"error": "request_too_large"}, status_code=413)
            if len(await request.body()) > 16_384:
                return JSONResponse({"error": "request_too_large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(APIServiceError)
    async def service_error(_: Request, exc: APIServiceError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"error": exc.code}, status_code=exc.status, headers=headers)

    @app.exception_handler(AuthenticationError)
    async def auth_error(_: Request, __: AuthenticationError):
        return JSONResponse({"error": "unauthorized"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(AuthorizationError)
    async def scope_error(_: Request, __: AuthorizationError):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    @app.get("/health/live")
    def live():
        return {"status": "live"}

    @app.get("/health/ready")
    def ready():
        with service.store._connection() as connection:
            connection.execute("SELECT 1").fetchone()
        if not service.worker_ready():
            return JSONResponse({"status": "not_ready", "reason": "worker_unavailable"}, status_code=503)
        return {"status": "ready", "worker": "separate_process"}

    @app.post("/v1/tasks", status_code=202)
    def submit(body: SubmitTask, authorization: str | None = Header(default=None),
               idempotency_key: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "submit")
        if not idempotency_key:
            raise APIServiceError(400, "idempotency_key_required")
        return service.submit(principal.principal_id, idempotency_key, body.question, body.session_id)

    @app.post("/v1/sandbox/tasks", status_code=202)
    def submit_sandbox(body: SandboxSubmit, authorization: str | None = Header(default=None),
                       idempotency_key: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "execute")
        if not idempotency_key:
            raise APIServiceError(400, "idempotency_key_required")
        return service.submit_sandbox(principal.principal_id, idempotency_key, body.code, body.inputs)

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "read")
        return service.task_view(principal.principal_id, task_id)

    @app.post("/v1/tasks/{task_id}/cancel", status_code=202)
    def cancel(task_id: str, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "control")
        return service.control(principal.principal_id, task_id, "cancel")

    @app.post("/v1/tasks/{task_id}/resume", status_code=202)
    def resume(task_id: str, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "control")
        return service.control(principal.principal_id, task_id, "resume")

    @app.post("/v1/tasks/{task_id}/approve", status_code=202)
    def approve(task_id: str, body: ApprovalDecision, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "approve")
        return service.decide(principal.principal_id, task_id, body.action_id, "approve", body.preview_digest)

    @app.post("/v1/tasks/{task_id}/reject", status_code=202)
    def reject(task_id: str, body: ApprovalDecision, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "approve")
        return service.decide(principal.principal_id, task_id, body.action_id, "reject", body.preview_digest)

    @app.get("/v1/tasks/{task_id}/events")
    async def events(task_id: str, request: Request, authorization: str | None = Header(default=None),
                     last_event_id: str | None = Header(default=None), after: int = 0):
        principal = tokens.authenticate(authorization, "read")
        service.owned_task(principal.principal_id, task_id)
        try:
            cursor = int(last_event_id) if last_event_id is not None else after
        except ValueError as exc:
            raise APIServiceError(400, "invalid_event_cursor") from exc
        if cursor < 0:
            raise APIServiceError(400, "invalid_event_cursor")

        async def stream():
            nonlocal cursor
            deadline = asyncio.get_running_loop().time() + 600
            while asyncio.get_running_loop().time() < deadline:
                if await request.is_disconnected():
                    break
                try:
                    tokens.authenticate(authorization, "read")
                    rows = await asyncio.to_thread(service.events, principal.principal_id, task_id, cursor)
                except (AuthenticationError, AuthorizationError, APIServiceError):
                    break
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: task\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                task = await asyncio.to_thread(service.owned_task, principal.principal_id, task_id)
                if task["status"] in TERMINAL_STATES and not rows:
                    break
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str, authorization: str | None = Header(default=None)):
        principal = tokens.authenticate(authorization, "read")
        answer = service.answer_artifact(principal.principal_id, artifact_id)
        return Response(answer, media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="answer.txt"',
                                 "Cache-Control": "no-store"})

    return app
