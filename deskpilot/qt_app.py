from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .core.agent import DocumentQAAgent
from .core.config import ROOT_DIR, ensure_dirs
from .rag.vector_index import DocumentIndex

try:
    from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Property, Qt, QTimer, Signal, Slot, QUrl
    from PySide6.QtWidgets import QApplication, QFileDialog
    from PySide6.QtGui import QFont
    from PySide6.QtQml import QQmlApplicationEngine
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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[dict[str, str]] = []

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API 命名
        return {self.MessageRole: b"messageRole", self.ContentRole: b"messageContent"}

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
        return None

    def reset_messages(self, messages: list[dict[str, str]]) -> None:
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
    confirmRequested = Signal(str, str, str)
    errorRaised = Signal(str)
    _resultReady = Signal(object)
    _indexReady = Signal(str)
    _errorReady = Signal(str)

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
        self._stream_base: list[dict[str, str]] = []
        self._stream_text = ""
        self._stream_index = 0
        self._stream_result: Any = None
        self._stream_generation = 0
        self._pending_action: dict[str, Any] | None = None
        # 所有后台任务都通过 signal 回到 Qt 主线程，再更新 QML 属性。
        self._resultReady.connect(self._start_stream)
        self._indexReady.connect(self._index_finished)
        self._errorReady.connect(self._background_failed)
        self._refresh_all()
        self._load_session_messages()

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
        self._end_operation()
        self._set_status("操作失败")
        self.errorRaised.emit(message)

    def _refresh_all(self) -> None:
        self._refresh_sessions()
        self._refresh_stats()
        self._refresh_memories()

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
            {"role": item.role, "content": item.content}
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

    def _set_messages(self, messages: list[dict[str, str]]) -> None:
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
        # 结果已写入会话存储，这里去掉最后一条完整回答，再在主线程逐字显示。
        self.current_session_id = result.session_id or self.current_session_id
        saved = self.agent.session_store.read_messages(self.current_session_id)
        base = saved[:-1] if saved and saved[-1].role == "assistant" else saved
        self._stream_base = [{"role": item.role, "content": item.content} for item in base]
        self._stream_text = str(getattr(result, "answer", getattr(result, "report", "")))
        self._stream_index = 0
        self._stream_result = result
        self._stream_generation += 1
        generation = self._stream_generation
        self._set_status("正在生成")
        # Agent 已经完成执行，Steps/Evidence 应立即可见，无需等待最多约 6 秒的
        # 逐字动画结束。聊天文本仍保持增量显示。
        self._set_result_panels(result)
        # 流开始时只重置一次，后续每一帧只更新最后一个 delegate。
        self._set_messages(self._stream_base + [{"role": "assistant", "content": ""}])
        self._stream_tick(generation)

    def _stream_tick(self, generation: int) -> None:
        # 新回答开始后丢弃旧定时器回调，避免两个流同时刷新同一消息模型。
        if generation != self._stream_generation:
            return
        # 长回答限制在约 500 次布局刷新内，同时保留短回答的平滑逐字效果。
        chunk_size = max(3, (len(self._stream_text) + 499) // 500)
        self._stream_index = min(len(self._stream_text), self._stream_index + chunk_size)
        self._update_last_message(self._stream_text[: self._stream_index])
        if self._stream_index < len(self._stream_text):
            QTimer.singleShot(12, lambda: self._stream_tick(generation))
            return
        result = self._stream_result
        self._stream_result = None
        if result is not None:
            self._set_status("回答完成")
            self._maybe_request_confirmation(result)
        self._end_operation()

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
        self._set_messages(self._messages + [{"role": "confirmation", "content": confirmation}])

    @Slot(str)
    def ask(self, question: str) -> None:
        question = question.strip()
        if not question or not self._begin_operation("正在处理"):
            return
        session_id = self.current_session_id
        self._steps = []
        self._evidences = []
        self.stepsChanged.emit()
        self.evidencesChanged.emit()
        self._set_messages(self._messages + [{"role": "user", "content": question}, {"role": "assistant", "content": "正在处理..."}])
        threading.Thread(target=self._run_answer, args=(question, session_id), daemon=True).start()

    def _run_answer(self, question: str, session_id: str) -> None:
        try:
            result = self.agent.answer(question, session_id=session_id)
            self._resultReady.emit(result)
        except Exception as exc:  # pragma: no cover - 防止后台异常让 GUI 无响应
            self._errorReady.emit(str(exc))

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
            None, "选择文档", str(ROOT_DIR), "Documents (*.pdf *.docx *.pptx *.xlsx *.csv *.md *.txt);;All files (*)"
        )
        if path:
            threading.Thread(target=self._index_file, args=(Path(path),), daemon=True).start()
        else:
            self._end_operation()

    @Slot()
    def chooseIndexFolder(self) -> None:
        if not self._begin_operation("正在建立索引"):
            return
        path = QFileDialog.getExistingDirectory(None, "选择文件夹", str(ROOT_DIR))
        if path:
            threading.Thread(target=self._index_folder, args=(Path(path),), daemon=True).start()
        else:
            self._end_operation()

    def _index_file(self, path: Path) -> None:
        try:
            document, count = self.index.add_file(path)
            self._indexReady.emit(f"{document.title}: {count} 个片段")
        except Exception as exc:
            self._errorReady.emit(str(exc))

    def _index_folder(self, path: Path) -> None:
        try:
            results = self.index.add_folder(path)
            detail = "\n".join(f"{document.title}: {count} 个片段" for document, count in results)
            self._indexReady.emit(detail or "没有找到支持的文档。")
        except Exception as exc:
            self._errorReady.emit(str(exc))

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
    app = QApplication(sys.argv)
    # Windows 中文办公场景优先使用系统常见字体，避免默认字体缺字显示方框。
    app.setFont(QFont("Microsoft YaHei UI", 10))
    engine = QQmlApplicationEngine()
    bridge = DeskPilotBridge()
    engine.rootContext().setContextProperty("deskPilot", bridge)
    qml_path = Path(__file__).with_name("qml") / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
