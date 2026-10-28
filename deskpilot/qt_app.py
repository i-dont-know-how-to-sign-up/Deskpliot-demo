from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from .core.agent import DocumentQAAgent
from .core.config import ROOT_DIR, WORKSPACE_DIR, ensure_dirs
from .release.profile import release_mode
from .rag.vector_index import DocumentIndex

try:
    from PySide6.QtCore import (
        QAbstractListModel, QBuffer, QIODevice, QModelIndex, QObject, Property, Qt, Signal, Slot, QUrl, QTimer,
    )
    from PySide6.QtWidgets import QApplication, QFileDialog
    from PySide6.QtGui import QFont
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
except ImportError as exc:  # pragma: no cover - 给未安装 GUI 依赖时提供清晰提示
    raise RuntimeError(
        f"当前解释器未安装 PySide6：{sys.executable}\n"
        "请使用项目环境运行，或在当前解释器中执行: python -m pip install PySide6"
    ) from exc


def _step_dict(step: Any) -> dict[str, str]:
    detail = str(step.detail)
    # retrieval_trace 可能包含数万字符。UI 保留足够的审计信息并限制布局成本，
    # 完整内容仍会持久化到会话 metadata 和导出文件中。
    if len(detail) > 6000:
        detail = detail[:6000] + "\n... [UI detail truncated; export session for the persisted trace]"
    return {"name": str(step.name), "status": str(step.status), "detail": detail}


def _evidence_dict(item: Any) -> dict[str, Any]:
    return {
        "source": str(item.source_label),
        "score": f"{float(item.score):.4f}",
        "text": str(item.text),
    }


class ChatMessageModel(QAbstractListModel):
    """支持单条增量更新的聊天模型，避免流式输出时反复重建整个列表。"""

    MessageRole = Qt.UserRole + 1
    ContentRole = Qt.UserRole + 2
    AttachmentsRole = Qt.UserRole + 3

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[dict[str, Any]] = []

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API 命名
        return {
            self.MessageRole: b"messageRole", self.ContentRole: b"messageContent",
            self.AttachmentsRole: b"messageAttachments",
        }

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        if role == self.MessageRole:
            return item.get("role", "assistant")
        if role in {self.ContentRole, Qt.DisplayRole}:
            return item.get("content", "")
        if role == self.AttachmentsRole:
            return item.get("attachments", [])
        return None

    def reset_messages(self, messages: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._items = [dict(item) for item in messages]
        self.endResetModel()

    def update_content(self, row: int, content: str) -> None:
        if not 0 <= row < len(self._items) or self._items[row].get("content") == content:
            return
        self._items[row]["content"] = content
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.ContentRole])


class DeskPilotBridge(QObject):
    """Qt Quick 与现有 Agent 核心之间的最小桥接层。"""

    sessionsChanged = Signal()
    stepsChanged = Signal()
    evidencesChanged = Signal()
    memoriesChanged = Signal()
    statsChanged = Signal()
    statusChanged = Signal()
    busyChanged = Signal()
    contextChanged = Signal()
    tasksChanged = Signal()
    pendingAttachmentsChanged = Signal()
    releaseInfoChanged = Signal()
    confirmRequested = Signal(str, str, str)
    errorRaised = Signal(str)
    _resultReady = Signal(object)
    _streamEvent = Signal(object)
    _indexReady = Signal(str)
    _indexProgress = Signal(str)
    _errorReady = Signal(str)
    _taskResultReady = Signal(str, object)
    _taskErrorReady = Signal(str, str)
    _mailReconcileReady = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.index = DocumentIndex()
        self.agent = DocumentQAAgent(self.index)
        self.current_session_id = self.agent.session_store.get_or_create().session_id
        self._messages: list[dict[str, str]] = []
        self._message_model = ChatMessageModel(self)
        self._sessions: list[dict[str, Any]] = []
        self._steps: list[dict[str, str]] = []
        self._evidences: list[dict[str, Any]] = []
        self._memories: list[dict[str, Any]] = []
        self._stats: dict[str, Any] = {}
        self._status = "就绪"
        self._busy = False
        self._context = ""
        self._live_stream_received = False
        self._stream_buffer = ""
        self._stream_closed = False
        self._pending_action: dict[str, Any] | None = None
        self._pending_attachments: list[dict[str, Any]] = []
        self._tasks: list[dict[str, Any]] = []
        self._visible_task_id = ""
        self._release_info: dict[str, Any] = {}
        # 所有后台任务都通过 signal 回到 Qt 主线程，再更新 QML 属性。
        self._resultReady.connect(self._start_stream)
        self._streamEvent.connect(self._handle_stream_event)
        self._indexReady.connect(self._index_finished)
        self._indexProgress.connect(self._index_progress)
        self._errorReady.connect(self._background_failed)
        self._taskResultReady.connect(self._handle_task_result)
        self._taskErrorReady.connect(self._handle_task_error)
        self._mailReconcileReady.connect(self._handle_mail_reconcile)
        self.agent.task_service.on_result = lambda task_id, result: self._taskResultReady.emit(task_id, result)
        self.agent.task_service.on_error = lambda task_id, error: self._taskErrorReady.emit(task_id, error)
        self.agent.task_event_callback = lambda event: self._streamEvent.emit(event)
        self._task_timer = QTimer(self)
        self._task_timer.setInterval(750)
        self._task_timer.timeout.connect(self.refreshTasks)
        self._task_timer.start()
        self._stream_timer = QTimer(self)
        self._stream_timer.setSingleShot(True)
        self._stream_timer.setInterval(40)
        self._stream_timer.timeout.connect(self._flush_stream)
        self._refresh_all()
        self._load_session_messages()
        self._refresh_release_info()

    def _get(self, name: str) -> Any:
        return getattr(self, f"_{name}")

    messages = Property(QObject, lambda self: self._message_model, constant=True)
    sessions = Property("QVariantList", lambda self: self._get("sessions"), notify=sessionsChanged)
    steps = Property("QVariantList", lambda self: self._get("steps"), notify=stepsChanged)
    evidences = Property("QVariantList", lambda self: self._get("evidences"), notify=evidencesChanged)
    memories = Property("QVariantList", lambda self: self._get("memories"), notify=memoriesChanged)
    stats = Property("QVariantMap", lambda self: self._get("stats"), notify=statsChanged)
    status = Property("QString", lambda self: self._get("status"), notify=statusChanged)
    busy = Property(bool, lambda self: self._busy, notify=busyChanged)
    context = Property("QString", lambda self: self._get("context"), notify=contextChanged)
    tasks = Property("QVariantList", lambda self: self._tasks, notify=tasksChanged)
    pendingAttachments = Property(
        "QVariantList", lambda self: self._pending_attachments, notify=pendingAttachmentsChanged
    )
    releaseMode = Property(bool, lambda self: release_mode(), constant=True)
    releaseInfo = Property("QVariantMap", lambda self: getattr(self, "_release_info", {}), notify=releaseInfoChanged)

    def _refresh_release_info(self) -> None:
        if not release_mode():
            return
        from .release.profile import profile_home, read_settings
        from .release.autostart import autostart_enabled
        from .release.secrets import SecretStore

        home = profile_home()
        settings = read_settings(home)
        vault_error = False
        try:
            vault = SecretStore(home / "secrets.dpapi")
            if vault.path.exists():
                vault.get("DASHSCOPE_API_KEY")
        except (ImportError, OSError, RuntimeError, ValueError):
            vault_error = True
        self._release_info = {
            "home": str(home),
            "model": settings.get("QWEN_MODEL", "qwen3.7-plus"),
            "visionModel": settings.get("VISION_MODEL", "qwen-vl-max-latest"),
            "visionBaseUrl": settings.get("VISION_BASE_URL", settings.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            "cloudImages": settings.get("ALLOW_CLOUD_IMAGE_UPLOAD", "false") == "true",
            "autostart": autostart_enabled() if sys.platform == "win32" else False,
            "emailProvider": settings.get("EMAIL_PROVIDER", "mock"),
            "emailUsername": settings.get("EMAIL_USERNAME", ""),
            "vaultError": vault_error,
        }
        self.releaseInfoChanged.emit()

    @Slot(str, str, bool)
    def saveReleaseSettings(self, model: str, api_key: str, cloud_images: bool) -> None:
        if not release_mode() or self._busy:
            return
        from .release.profile import profile_home, read_settings, save_settings
        from .release.secrets import SecretStore

        model = model.strip()
        if not model:
            self.errorRaised.emit("模型名称不能为空。")
            return
        try:
            home = profile_home()
            settings = read_settings(home)
            settings.update({"QWEN_MODEL": model, "ALLOW_CLOUD_IMAGE_UPLOAD": "true" if cloud_images else "false"})
            if api_key.strip():
                SecretStore(home / "secrets.dpapi").set("DASHSCOPE_API_KEY", api_key.strip())
            save_settings(settings, home)
            self._refresh_release_info()
            self._set_status("设置已保存；重启后生效")
        except (OSError, RuntimeError, ValueError) as exc:
            self.errorRaised.emit(str(exc))

    @Slot(bool)
    def setWorkerAutostart(self, enabled: bool) -> None:
        if not release_mode():
            return
        from .release.autostart import set_autostart

        try:
            set_autostart(enabled)
            self._refresh_release_info()
            self._set_status("后台 worker 登录自启动已更新")
        except (OSError, RuntimeError, ValueError) as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str, str, str, result=bool)
    def saveVisionSettings(self, model: str, base_url: str, api_key: str) -> bool:
        if not release_mode():
            self.errorRaised.emit("视觉模型设置仅在发布模式中可用。")
            return False
        if self._busy:
            self.errorRaised.emit("当前任务仍在运行，请等待任务完成后再保存视觉模型设置。")
            return False
        from urllib.parse import urlsplit
        from .release.profile import profile_home, read_settings, save_settings
        from .release.secrets import SecretStore

        model = model.strip()
        base_url = base_url.strip().rstrip("/")
        parsed = urlsplit(base_url)
        if not model or not base_url or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            self.errorRaised.emit("请填写视觉模型名称和 HTTPS API 基础地址，地址不能包含账号、查询参数或片段。")
            return False
        try:
            home = profile_home()
            settings = read_settings(home)
            settings.update({"VISION_MODEL": model, "VISION_BASE_URL": base_url})
            if api_key.strip():
                SecretStore(home / "secrets.dpapi").set("VISION_API_KEY", api_key.strip())
            save_settings(settings, home)
            self._refresh_release_info()
            self._set_status("视觉模型设置已保存；重启后生效")
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.errorRaised.emit(f"视觉模型设置保存失败：{exc}")
            return False

    @Slot(str, str, str, result=bool)
    def saveEmailSettings(self, provider: str, username: str, password: str) -> bool:
        if not release_mode():
            self.errorRaised.emit("邮箱设置仅在发布模式中可用。")
            return False
        if self._busy:
            self.errorRaised.emit("当前任务仍在运行，请等待任务完成后再保存邮箱设置。")
            return False
        from .release.profile import profile_home, read_settings, save_settings
        from .release.secrets import SecretStore

        provider = provider.strip().lower()
        if provider not in {"mock", "qq", "163", "outlook"}:
            self.errorRaised.emit("不支持的邮箱服务商。")
            return False
        if provider != "mock" and (not username.strip() or "@" not in username):
            self.errorRaised.emit("请填写有效的邮箱地址。")
            return False
        try:
            home = profile_home()
            settings = read_settings(home)
            account = username.strip()
            if provider != "mock" and not password.strip():
                saved_password = SecretStore(home / "secrets.dpapi").get("EMAIL_PASSWORD")
                if not saved_password or settings.get("EMAIL_PROVIDER") != provider or settings.get("EMAIL_USERNAME") != account:
                    raise ValueError("新邮箱账号必须填写客户端授权码；留空仅适用于原邮箱账号不变且已有可读取的授权码。")
            settings.update({"EMAIL_PROVIDER": provider, "EMAIL_USERNAME": account})
            if password.strip():
                SecretStore(home / "secrets.dpapi").set("EMAIL_PASSWORD", password.strip())
            save_settings(settings, home)
            self._refresh_release_info()
            self._set_status("邮箱设置已保存；重启后生效")
            return True
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            self._set_status("邮箱设置保存失败")
            self.errorRaised.emit(f"邮箱设置保存失败：{detail}")
            return False

    @Slot()
    def resetReleaseCredentials(self) -> None:
        if not release_mode() or self._busy:
            return
        from .release.profile import profile_home
        from .release.secrets import SecretStore

        try:
            previous = SecretStore(profile_home() / "secrets.dpapi").reset()
            detail = f"；旧文件保留为 {previous.name}" if previous else ""
            self._set_status("凭据已重置，需重新填写模型 Key 和邮箱授权码" + detail)
        except Exception as exc:
            self.errorRaised.emit(str(exc) or type(exc).__name__)

    @Slot()
    def exportDiagnostics(self) -> None:
        if not release_mode():
            return
        from .release.diagnostics import export_diagnostics
        from .release.profile import profile_home

        path, _ = QFileDialog.getSaveFileName(None, "导出诊断信息", str(profile_home() / "diagnostics.json"), "JSON (*.json)")
        if path:
            try:
                export_diagnostics(Path(path))
                self._set_status("诊断信息已导出")
            except (OSError, RuntimeError, ValueError) as exc:
                self.errorRaised.emit(str(exc))

    @Slot(str)
    def saveFeedback(self, description: str) -> None:
        if not release_mode():
            return
        from .release.feedback import save_local_feedback

        try:
            save_local_feedback(description)
            self._set_status("反馈已保存在本机，未自动上传")
        except (OSError, ValueError) as exc:
            self.errorRaised.emit(str(exc))

    def _set_status(self, value: str) -> None:
        self._status = value
        self.statusChanged.emit()

    def _begin_operation(self, status: str) -> bool:
        """共享 Agent 和索引采用单任务串行策略，避免跨线程状态串扰。"""
        if self._busy:
            return False
        self._busy = True
        self.busyChanged.emit()
        self._set_status(status)
        return True

    def _end_operation(self) -> None:
        if self._busy:
            self._busy = False
            self.busyChanged.emit()

    def _background_failed(self, message: str) -> None:
        if self._messages and self._messages[-1].get("role") == "assistant":
            self._update_last_message(f"操作失败：{message}")
        self._end_operation()
        self._set_status("操作失败")
        self.errorRaised.emit(message)

    def _refresh_all(self) -> None:
        self._refresh_sessions()
        self._refresh_stats()
        self._refresh_memories()
        self.refreshTasks()

    def _refresh_sessions(self) -> None:
        self._sessions = [
            {
                "id": item.session_id,
                "title": item.title,
                "count": item.message_count,
                "active": item.session_id == self.current_session_id,
                "pinned": item.pinned,
            }
            for item in self.agent.session_store.list_sessions()
        ]
        self.sessionsChanged.emit()

    def _refresh_stats(self) -> None:
        current = self.index.stats()
        self._stats = {
            "documents": int(current["documents"]),
            "chunks": int(current["chunks"]),
            "indexFile": "data/index/index.json",
        }
        self.statsChanged.emit()

    def _refresh_memories(self) -> None:
        items = self.agent.memory_store.list_memories(
            session_id=self.current_session_id, status="active", limit=80
        )
        self._memories = [
            {
                "id": item.memory_id,
                "title": f"{item.memory_type} / {item.scope}",
                "status": item.status,
                "confidence": f"{item.confidence:.2f}",
                "content": item.content,
            }
            for item in items
        ]
        self.memoriesChanged.emit()

    def _load_session_messages(self) -> None:
        messages = self.agent.session_store.read_messages(self.current_session_id)
        self._messages = [
            {
                "role": item.role, "content": item.content,
                "attachments": [
                    view for asset_id in (
                        item.attachment_ids
                        or (item.metadata.get("attachment_ids", []) if isinstance(item.metadata, dict) else [])
                    )
                    if (view := self.agent.multimodal.attachment_view(str(asset_id))) is not None
                ],
            }
            for item in messages
        ]
        self._message_model.reset_messages(self._messages)
        assistant = next((item for item in reversed(messages) if item.role == "assistant"), None)
        metadata = assistant.metadata if assistant and isinstance(assistant.metadata, dict) else {}
        raw_steps = metadata.get("steps", [])
        raw_evidences = metadata.get("evidences", [])
        self._steps = [dict(item) for item in raw_steps if isinstance(item, dict)]
        self._evidences = [
            {
                "source": str(item.get("source", "")),
                "score": f"{float(item.get('score', 0.0)):.4f}",
                "text": str(item.get("text", "")),
            }
            for item in raw_evidences if isinstance(item, dict)
        ]
        self.stepsChanged.emit()
        self.evidencesChanged.emit()
        self._pending_action = None
        pending = self.agent.pending_actions.list_pending(session_id=self.current_session_id)
        if pending:
            self._maybe_request_confirmation(type("Result", (), {"pending_action": self.agent.pending_actions.public(pending[-1])})())

    def _set_messages(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self._message_model.reset_messages(self._messages)

    def _update_last_message(self, content: str) -> None:
        if not self._messages:
            return
        self._messages[-1]["content"] = content
        self._message_model.update_content(len(self._messages) - 1, content)

    def _set_result_panels(self, result: Any) -> None:
        self.current_session_id = result.session_id or self.current_session_id
        self._steps = [_step_dict(step) for step in result.steps]
        self._evidences = [_evidence_dict(item) for item in result.evidences]
        self._context = result.memory_context or "本轮没有额外上下文。"
        self.stepsChanged.emit()
        self.evidencesChanged.emit()
        self.contextChanged.emit()
        self._refresh_sessions()
        self._refresh_memories()

    def _start_stream(self, result: Any) -> None:
        """收束真实流式结果；不再对完整回答做二次打字机动画。"""
        self._flush_stream()
        self._stream_closed = True
        self.current_session_id = result.session_id or self.current_session_id
        final_text = str(getattr(result, "answer", getattr(result, "report", "")))
        self._update_last_message(final_text)
        self._set_result_panels(result)
        self._set_status("回答完成")
        self._maybe_request_confirmation(result)
        self.refreshTasks()
        self._end_operation()

    def _handle_stream_event(self, event: Any) -> None:
        """所有事件经 Qt signal 回到主线程，只更新一个消息 delegate。"""
        if not isinstance(event, dict):
            return
        if event.get("session_id") and event["session_id"] != self.current_session_id:
            return
        event_type = str(event.get("type", ""))
        stage = str(event.get("stage", "Model"))
        if event_type == "token":
            if self._stream_closed:
                return
            delta = str(event.get("text", ""))
            if not delta:
                return
            if not self._live_stream_received:
                self._live_stream_received = True
                self._update_last_message("")
            self._stream_buffer += delta
            if not self._stream_timer.isActive():
                self._stream_timer.start()
            self._set_status(f"正在生成 / {stage}")
            return
        if event_type == "model_start":
            self._set_status(f"正在处理 / {stage}")
            self._upsert_runtime_step(stage, "running", "模型调用进行中。")
        elif event_type == "model_end":
            ok = event.get("ok") is True
            detail = "模型调用完成。" if ok else f"模型调用失败：{event.get('error', '')}"
            self._upsert_runtime_step(stage, "success" if ok else "failed", detail)

    def _flush_stream(self) -> None:
        if not self._stream_buffer:
            return
        current = self._messages[-1].get("content", "") if self._messages else ""
        self._update_last_message(current + self._stream_buffer)
        self._stream_buffer = ""

    def _upsert_runtime_step(self, stage: str, status: str, detail: str) -> None:
        name = f"model:{stage}"
        replacement = {"name": name, "status": status, "detail": detail}
        for index, item in enumerate(self._steps):
            if item.get("name") == name:
                self._steps[index] = replacement
                break
        else:
            self._steps.append(replacement)
        self.stepsChanged.emit()

    def _maybe_request_confirmation(self, result: Any) -> None:
        pending = getattr(result, "pending_action", None)
        if not pending:
            return
        self._pending_action = pending
        reasons = pending.get("reasons", []) if isinstance(pending, dict) else []
        description = str(pending.get("description", "需要人工确认的操作"))
        reason_text = "\n".join(str(reason) for reason in reasons) or "该操作可能修改本地状态。"
        # 将确认请求写入聊天流，由 QML 在同一条消息中渲染确认/取消按钮。
        confirmation = (
            f"需要确认执行：{description}\n"
            f"风险等级：{pending.get('risk_level', 'high')}\n"
            f"原因：{reason_text}"
        )
        preview = str(pending.get("preview", "") or "")
        if preview:
            confirmation += "\n\n" + preview
        self._set_messages(self._messages + [{"role": "confirmation", "content": confirmation}])

    @Slot(str)
    def ask(self, question: str) -> None:
        question = question.strip()
        if (not question and not self._pending_attachments) or not self._begin_operation("正在处理"):
            return
        session_id = self.current_session_id
        attachments = list(self._pending_attachments)
        asset_ids = [str(item["assetId"]) for item in attachments]
        self._pending_attachments = []
        self.pendingAttachmentsChanged.emit()
        self._steps = []
        self._evidences = []
        self._live_stream_received = False
        self._stream_closed = False
        self._stream_buffer = ""
        self._stream_timer.stop()
        self.stepsChanged.emit()
        self.evidencesChanged.emit()
        self._set_messages(self._messages + [
            {"role": "user", "content": question or "请描述这些图片。", "attachments": attachments},
            {"role": "assistant", "content": "", "attachments": []},
        ])
        threading.Thread(target=self._run_answer, args=(question, session_id, asset_ids), daemon=True).start()

    def _run_answer(self, question: str, session_id: str, asset_ids: list[str] | None = None) -> None:
        try:
            self._visible_task_id = self.agent.task_service.submit("core.answer", {
                "question": question, "session_id": session_id, "asset_ids": asset_ids or [],
            }, session_id=session_id)
            self.agent.task_service.start()
        except Exception as exc:  # pragma: no cover - 防止后台异常让 GUI 无响应
            self._errorReady.emit(str(exc))

    def _handle_task_result(self, task_id: str, result: Any) -> None:
        self._stream_timer.stop()
        self._stream_buffer = ""
        self._stream_closed = True
        self.refreshTasks()
        if isinstance(result, list):
            self._index_finished(f"多模态入库完成：{len(result)} 页/图片，复用 {sum(bool(item.get('checkpoint_reused')) for item in result)} 个检查点")
            return
        if getattr(result, "session_id", "") and result.session_id != self.current_session_id:
            self._refresh_all()
            self._end_operation()
            self._set_status("后台任务已更新")
            return
        # 恢复任务可能没有本轮占位消息。以已持久化的会话为准，避免覆盖旧回答。
        self._load_session_messages()
        self._set_result_panels(result)
        self._set_status("等待确认" if getattr(result, "pending_action", None) else "回答完成")
        self._end_operation()

    def _handle_task_error(self, task_id: str, error: str) -> None:
        self._stream_timer.stop()
        self._stream_buffer = ""
        self._stream_closed = True
        self.refreshTasks()
        task = self.agent.task_ledger.store.get_task(task_id)
        state = task["status"] if task else "failed"
        if task and task.get("session_id") == self.current_session_id and state not in {"paused", "cancelled"}:
            answer = f"任务失败：{error or '请在 Tasks 中查看失败原因。'}"
            messages = self.agent.session_store.read_messages(self.current_session_id)
            if messages and messages[-1].role == "user":
                self.agent.session_store.append_message(
                    self.current_session_id, "assistant", answer,
                    metadata={"steps": [{"name": "task_error", "status": "failed", "detail": error}],
                              "task_id": task_id},
                )
            self._load_session_messages()
        self._end_operation()
        self._set_status({"paused": "任务已暂停", "cancelled": "任务已取消"}.get(state, "任务失败，请查看 Tasks"))
        if state not in {"paused", "cancelled"}:
            self.errorRaised.emit(error)

    @Slot()
    def refreshTasks(self) -> None:
        from .tasks.effects import effects_for_task

        store = self.agent.task_ledger.store
        store.expire_leases()
        items = []
        labels = {"queued": "排队", "running": "执行中", "retry_wait": "等待重试", "waiting_human": "等待审批",
                  "paused": "已暂停", "interrupted": "已中断", "recovery_required": "需人工核对",
                  "success": "已完成", "succeeded": "已完成", "failed": "失败", "cancelled": "已取消", "partial": "部分完成"}
        for task in reversed(store.list_tasks()[-100:]):
            job = self.agent.task_service.descriptor(task["task_id"])
            nodes = task["nodes"]
            usage = job["usage"] if job else {}
            complete = sum(node["status"] in {"success", "succeeded"} for node in nodes.values())
            pending = self.agent.pending_actions.list_pending(task_id=task["task_id"])
            effects = effects_for_task(store, task["task_id"])
            unfinished_compensation = any(item["state"] in {"compensating", "compensation_unknown"} for item in effects)
            uncertain_file = any(item["state"] in {"dispatched", "unknown"}
                                 and (item["tool_name"].startswith("files.write") or item["tool_name"] == "files.apply_move")
                                 for item in effects)
            file_effect = next((item for item in effects if item["state"] == "committed"
                                and (item["tool_name"].startswith("files.write") or item["tool_name"] == "files.apply_move")
                                and json.loads(item["result_json"] or "{}").get("compensatable")), None)
            mail_effect = next((item for item in effects if item["tool_name"] == "email.send"
                                and (item["state"] in {"unknown", "dispatched"} or item["state"] == "committed"
                                     and json.loads(item["result_json"] or "{}").get("sent_copy_saved") is False)), None)
            title = str(job["payload"].get("question") or job["payload"].get("path") or job["handler_key"]) if job else task["goal"]
            detail = f"{complete}/{len(nodes)} 节点 · {round(job['elapsed'], 1) if job else 0}s\n"
            pdf = job.get("pdf_progress") if job else None
            if pdf:
                detail = f"PDF {pdf['completed']}/{pdf['total']} 页 · 复用 {pdf['reused']} 页 · {round(job['elapsed'], 1)}s\n"
            detail += f"Tool {usage.get('tool_calls', 0)} · API {usage.get('api_attempts', 0)} · Token {usage.get('total_tokens', 0)}"
            detail += f" · Retry {usage.get('retries', 0)}"
            if usage.get("unknown_usage_calls"):
                detail += f" · usage 未上报 {usage['unknown_usage_calls']} 次"
            if task["pause_requested"] and task["status"] == "running":
                detail += "\n正在请求暂停"
            if task["cancel_requested"] and task["status"] == "running":
                detail += "\n正在请求取消"
            if task["error"]:
                detail += "\n" + task["error"]
            if file_effect:
                detail += "\n文件写入可申请补偿（将重新校验文件版本）"
            if mail_effect and mail_effect["state"] in {"unknown", "dispatched"}:
                detail += "\n邮件提交结果未知：禁止自动重发，请人工核对邮箱。"
            elif mail_effect:
                detail += "\nSMTP 已接受，已发送箱副本待核对。"
            if unfinished_compensation:
                detail += "\n文件补偿中断：请核对副作用日志和文件版本，勿直接重试。"
            if uncertain_file:
                detail += "\n文件提交结果未确认：请先核对文件及副作用日志。"
            state = task["status"]
            items.append({"id": task["task_id"], "title": title[:100], "state": state,
                          "status": labels.get(state, state), "detail": detail,
                          "progress": (pdf["completed"] / max(1, pdf["total"])) if pdf else
                                      (1 if state in {"success", "succeeded"} else complete / max(1, len(nodes))),
                          "canPause": state in {"queued", "running", "retry_wait"},
                          "canCancel": state in {"queued", "running", "waiting_human", "paused", "retry_wait"},
                          "canResume": bool(job) and state in {"paused", "failed", "interrupted", "partial", "recovery_required"}
                                       and not any(node["status"] == "unknown" for node in nodes.values()),
                          "canApprove": bool(pending) and state == "waiting_human"})
            items[-1]["compensationOperationId"] = file_effect["operation_id"] if file_effect else ""
            items[-1]["mailOperationId"] = mail_effect["operation_id"] if mail_effect else ""
        if items != self._tasks:
            self._tasks = items
            self.tasksChanged.emit()

    @Slot(str, str)
    def controlTask(self, task_id: str, action: str) -> None:
        try:
            if action == "resume":
                if not self._begin_operation("正在恢复任务"):
                    return
                self.agent.task_service.resume(task_id)
                self._visible_task_id = task_id
                self.agent.task_service.start()
            else:
                self.agent.task_service.request(task_id, action)
                if action == "cancel":
                    for pending in self.agent.pending_actions.list_pending(task_id=task_id):
                        self.agent.pending_actions.cancel(pending.action_id, pending.session_id)
            self.refreshTasks()
        except Exception as exc:
            if action == "resume":
                self._end_operation()
            self.errorRaised.emit(str(exc))

    @Slot(str)
    def previewTask(self, task_id: str) -> None:
        if self._busy:
            return
        task = self.agent.task_ledger.store.get_task(task_id)
        if not task:
            return
        if task["session_id"]:
            self.selectSession(task["session_id"])
        pending = self.agent.pending_actions.list_pending(task_id=task_id)
        if pending:
            item = pending[-1]
            refreshed = self.agent.pending_actions.renew(item.action_id, item.session_id)
            self._maybe_request_confirmation(type("Result", (), {"pending_action": refreshed})())
        self.refreshTasks()

    @Slot(str, str)
    def requestCompensation(self, task_id: str, operation_id: str) -> None:
        try:
            pending = self.agent.request_file_compensation(task_id, operation_id, self.current_session_id)
            self._maybe_request_confirmation(type("Result", (), {"pending_action": pending})())
            self._set_status("等待文件补偿确认")
        except Exception as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str, str)
    def reconcileMail(self, task_id: str, operation_id: str) -> None:
        if not self._begin_operation("正在核对邮件"):
            return
        session_id = self.current_session_id
        def run() -> None:
            try:
                self._mailReconcileReady.emit(self.agent.reconcile_mail_effect(task_id, operation_id, session_id))
            except Exception as exc:
                self._errorReady.emit(str(exc))
        threading.Thread(target=run, daemon=True).start()

    def _handle_mail_reconcile(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        self._set_messages(self._messages + [{"role": "assistant", "content":
            "邮件核对：" + str(data.get("reason") or data.get("action") or data.get("status"))}])
        self._set_status("邮件核对完成")
        self.refreshTasks()
        self._end_operation()

    @Slot(str, int)
    def research(self, topic: str, limit: int = 5) -> None:
        topic = topic.strip()
        if not topic or not self._begin_operation("正在调研"):
            return
        session_id = self.current_session_id
        self._set_messages(self._messages + [{"role": "user", "content": f"网页调研：{topic}"}, {"role": "assistant", "content": "正在搜索网页..."}])
        threading.Thread(target=self._run_research, args=(topic, session_id, max(1, min(limit, 10))), daemon=True).start()

    def _run_research(self, topic: str, session_id: str, limit: int) -> None:
        try:
            result = self.agent.research(topic, session_id=session_id, max_results=limit)
            # 调研结果使用 report 字段，_start_stream 会与普通回答统一显示。
            self._resultReady.emit(result)
        except Exception as exc:  # pragma: no cover
            self._errorReady.emit(str(exc))

    @Slot()
    def newSession(self) -> None:
        session = self.agent.session_store.create_session()
        self.current_session_id = session.session_id
        self._set_messages([])
        self._steps = []
        self._evidences = []
        self._context = ""
        self._pending_attachments = []
        self.pendingAttachmentsChanged.emit()
        self.stepsChanged.emit()
        self.evidencesChanged.emit()
        self.contextChanged.emit()
        self._refresh_all()

    @Slot(str)
    def selectSession(self, session_id: str) -> None:
        if not self.agent.session_store.get_session(session_id):
            return
        self.current_session_id = session_id
        self._load_session_messages()
        self._refresh_all()
        self._set_status("已切换会话")

    @Slot(str, str)
    def renameSession(self, session_id: str, title: str) -> None:
        try:
            self.agent.session_store.rename_session(session_id, title)
            self._refresh_sessions()
        except (OSError, ValueError) as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str)
    def togglePinSession(self, session_id: str) -> None:
        try:
            self.agent.session_store.set_pinned(session_id)
            self._refresh_sessions()
        except OSError as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str)
    def deleteSession(self, session_id: str) -> None:
        if session_id == self.current_session_id:
            self.errorRaised.emit("不能删除当前会话，请先切换到其他会话。")
            return
        try:
            self.agent.session_store.delete_session(session_id)
            self._refresh_sessions()
        except OSError as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str)
    def exportSession(self, session_id: str) -> None:
        path, _ = QFileDialog.getSaveFileName(None, "导出会话", f"{session_id}.md", "Markdown (*.md)")
        if not path:
            return
        try:
            self.agent.session_store.export_session(session_id, Path(path))
            self._set_status(f"会话已导出：{path}")
        except OSError as exc:
            self.errorRaised.emit(str(exc))

    @Slot()
    def compactSession(self) -> None:
        compacted, summary = self.agent.memory_compactor.compact_if_needed(
            self.current_session_id,
            memories=self.agent.memory_store.list_memories(session_id=self.current_session_id, limit=50),
            force=True,
        )
        self._context = ("会话摘要已更新。\n\n" if compacted else "当前会话摘要：\n\n") + (summary or "暂无摘要。")
        self.contextChanged.emit()

    @Slot()
    def chooseIndexFile(self) -> None:
        if not self._begin_operation("正在建立索引"):
            return
        path, _ = QFileDialog.getOpenFileName(
            None, "选择文档或图片", str(WORKSPACE_DIR if release_mode() else ROOT_DIR),
            "Documents and images (*.pdf *.docx *.pptx *.xlsx *.csv *.md *.txt *.png *.jpg *.jpeg *.webp);;All files (*)"
        )
        if path:
            threading.Thread(target=self._index_file, args=(Path(path),), daemon=True).start()
        else:
            self._end_operation()

    @Slot()
    def chooseIndexFolder(self) -> None:
        if not self._begin_operation("正在建立索引"):
            return
        path = QFileDialog.getExistingDirectory(None, "选择文件夹", str(WORKSPACE_DIR if release_mode() else ROOT_DIR))
        if path:
            threading.Thread(target=self._index_folder, args=(Path(path),), daemon=True).start()
        else:
            self._end_operation()

    def _index_file(self, path: Path) -> None:
        try:
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
                self.agent.task_service.submit("multimodal.index", {"path": str(path)},
                                               session_id=self.current_session_id, timeout_seconds=3600)
                self.agent.task_service.start()
            else:
                document, count = self.index.add_file(path)
                # PDF 同时建立文本索引；页图索引由显式多模态工具处理，避免普通索引意外增加成本。
                self._indexReady.emit(f"{document.title}: {count} 个片段")
        except Exception as exc:
            self._errorReady.emit(str(exc))

    @Slot()
    def chooseImageAttachments(self) -> None:
        if self._busy:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            None, "选择图片", str(WORKSPACE_DIR if release_mode() else ROOT_DIR), "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not paths:
            return
        self._add_image_paths([Path(path) for path in paths])

    def _add_image_paths(self, paths: list[Path]) -> None:
        try:
            max_count = max(1, min(int(__import__("os").getenv("IMAGE_MAX_COUNT_PER_TURN", "4")), 12))
            for path in paths:
                if len(self._pending_attachments) >= max_count:
                    raise ValueError(f"单轮最多选择 {max_count} 张图片")
                asset = self.agent.multimodal.add_chat_attachment(path)
                if any(item["assetId"] == asset.asset_id for item in self._pending_attachments):
                    continue
                view = self.agent.multimodal.attachment_view(asset.asset_id)
                if view:
                    self._pending_attachments.append(view)
            self.pendingAttachmentsChanged.emit()
            self._set_status(f"已选择 {len(self._pending_attachments)} 张图片")
        except Exception as exc:
            self.errorRaised.emit(str(exc))

    @Slot(str)
    def addImageUrls(self, encoded_urls: str) -> None:
        """接收 QML DropArea 的 URL；仅处理本地图片文件。"""
        try:
            values = json.loads(encoded_urls)
            paths = [Path(QUrl(str(value)).toLocalFile()) for value in values]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.errorRaised.emit(f"无法解析拖入的图片：{exc}")
            return
        self._add_image_paths([path for path in paths if str(path)])

    @Slot()
    def pasteImageAttachment(self) -> None:
        image = QApplication.clipboard().image()
        if image.isNull():
            self.errorRaised.emit("剪贴板中没有图片。")
            return
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        try:
            max_count = max(1, min(int(__import__("os").getenv("IMAGE_MAX_COUNT_PER_TURN", "4")), 12))
            if len(self._pending_attachments) >= max_count:
                raise ValueError(f"单轮最多选择 {max_count} 张图片")
            if not image.save(buffer, "PNG"):
                raise ValueError("剪贴板图片编码失败")
            asset = self.agent.multimodal.add_chat_attachment_bytes(bytes(buffer.data()))
            if not any(item["assetId"] == asset.asset_id for item in self._pending_attachments):
                view = self.agent.multimodal.attachment_view(asset.asset_id)
                if view:
                    self._pending_attachments.append(view)
                    self.pendingAttachmentsChanged.emit()
            self._set_status(f"已选择 {len(self._pending_attachments)} 张图片")
        except Exception as exc:
            self.errorRaised.emit(str(exc))
        finally:
            buffer.close()

    @Slot(str)
    def removePendingAttachment(self, asset_id: str) -> None:
        self._pending_attachments = [
            item for item in self._pending_attachments if item.get("assetId") != asset_id
        ]
        self.pendingAttachmentsChanged.emit()

    def _index_folder(self, path: Path) -> None:
        try:
            results = self.index.add_folder(path, progress=lambda done, total, file, trace:
                self._indexProgress.emit(
                    f"索引 {done}/{total} · {file.name} · {trace.get('duration_ms', 0)} ms"
                ))
            detail = "\n".join(f"{document.title}: {count} 个片段" for document, count in results)
            self._indexReady.emit(detail or "没有找到支持的文档。")
        except Exception as exc:
            self._errorReady.emit(str(exc))

    def _index_progress(self, detail: str) -> None:
        self._set_status(detail)
        self._steps = [{"name": "index_documents", "status": "running", "detail": detail}]
        self.stepsChanged.emit()

    def _index_finished(self, detail: str) -> None:
        self._end_operation()
        self._set_status("索引完成")
        self._refresh_stats()
        self._steps = [{"name": "index_documents", "status": "success", "detail": detail}]
        self.stepsChanged.emit()

    @Slot()
    def clearIndex(self) -> None:
        if self._busy:
            return
        self.index.clear()
        self._refresh_stats()
        self._set_status("索引已清空")

    @Slot(str)
    def approveAction(self, note: str = "") -> None:
        if not self._pending_action or not self._begin_operation("正在执行已确认操作"):
            return
        pending = self._pending_action
        self._pending_action = None
        threading.Thread(
            target=self._run_approved_action,
            args=(pending, self.current_session_id),
            daemon=True,
        ).start()

    @Slot()
    def cancelAction(self) -> None:
        """取消聊天中的待审批动作，不执行任何外部副作用。"""
        pending = self._pending_action
        self._pending_action = None
        if pending:
            self.agent.cancel_pending_action(pending, session_id=self.current_session_id)
        if self._messages and self._messages[-1].get("role") == "confirmation":
            cancelled = dict(self._messages[-1])
            cancelled["role"] = "assistant"
            cancelled["content"] = "已取消本次操作。"
            self._set_messages(self._messages[:-1] + [cancelled])
        self._set_status("已取消操作")

    def _run_approved_action(self, pending: dict[str, Any], session_id: str) -> None:
        try:
            result = self.agent.approve_pending_action(pending, session_id=session_id)
            self._resultReady.emit(result)
        except Exception as exc:
            self._errorReady.emit(str(exc))

    @Slot()
    def refreshMemories(self) -> None:
        self._refresh_memories()


def main() -> int:
    # 原生 Windows Style 不支持自定义控件背景，统一使用可定制的 Fusion。
    QQuickStyle.setStyle("Fusion")
    app = QApplication.instance() or QApplication(sys.argv)
    # Windows 中文办公场景优先使用系统常见字体，避免默认字体缺字显示方框。
    app.setFont(QFont("Microsoft YaHei UI", 10))
    engine = QQmlApplicationEngine()
    bridge = DeskPilotBridge()
    engine.rootContext().setContextProperty("deskPilot", bridge)
    qml_path = Path(__file__).with_name("qml") / "Main.qml"
    app.aboutToQuit.connect(bridge.agent.task_service.close)
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        return 1
    if "--smoke-ui" in sys.argv:
        QTimer.singleShot(200, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
