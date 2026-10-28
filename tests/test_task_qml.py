from __future__ import annotations

from pathlib import Path
import os

import pytest


@pytest.mark.parametrize("width,height", [(1440, 900), (1120, 720)])
def test_task_panel_loads_and_renders_offscreen(tmp_path: Path, monkeypatch, width: int, height: int) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("QT_QUICK_BACKEND", "software")
    from PySide6.QtCore import QObject, QUrl
    from PySide6.QtGui import QFont
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickWindow
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtWidgets import QApplication
    from deskpilot.qt_app import ChatMessageModel, DeskPilotBridge

    QQuickStyle.setStyle("Fusion")
    application = QApplication.instance() or QApplication([])
    application.setFont(QFont("Microsoft YaHei UI", 10))
    # 不创建真实 Agent，不读取私人会话、邮箱、图片或环境凭证。
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge._message_model = ChatMessageModel(bridge)
    bridge._message_model.reset_messages([{"role": "assistant", "content": "Synthetic task preview"}])
    for field in ("sessions", "steps", "evidences", "memories", "pending_attachments"):
        setattr(bridge, "_" + field, [])
    bridge._stats = {"documents": 0, "chunks": 0}
    bridge._status = "就绪"
    bridge._context = "Synthetic context"
    bridge._busy = False
    bridge._tasks = [{
        "id": "synthetic", "title": "报告与草稿：重启后继续原任务",
        "state": "waiting_human", "status": "等待审批", "progress": 0.75,
        "detail": "3/4 节点 · 12.5s\nTool 3 · API 2 · Token 1234\n已保存证据检查点",
        "canPause": False, "canCancel": True, "canResume": False, "canApprove": True,
    }, {
        "id": "synthetic2", "title": "PDF 图片批次入库", "state": "paused", "status": "已暂停",
        "progress": 0.5, "detail": "2/4 节点 · 21.3s\nTool 1 · API 0 · Token 0",
        "canPause": False, "canCancel": True, "canResume": True, "canApprove": False,
    }]
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
    engine.rootContext().setContextProperty("deskPilot", bridge)
    engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "deskpilot/qml/Main.qml")))
    assert engine.rootObjects(), warnings
    window = engine.rootObjects()[0]
    assert isinstance(window, QQuickWindow)
    try:
        window.resize(width, height)
        tabs = window.findChild(QObject, "observerTabs")
        assert tabs is not None
        tabs.setProperty("currentIndex", 4)
        for _ in range(20):
            application.processEvents()
        tasks = window.findChild(QObject, "tasksList")
        assert tasks is not None and tasks.property("count") == 2
        capture = window.grabWindow()
        assert not capture.isNull()
        assert capture.save(str(tmp_path / f"tasks_{width}.png"))
        capture_dir = os.getenv("DESKPILOT_QML_CAPTURE_DIR")
        if capture_dir:
            destination = Path(capture_dir)
            destination.mkdir(parents=True, exist_ok=True)
            assert capture.save(str(destination / f"tasks_ui_{width}.png"))
        colors = {capture.pixelColor(x, y).name() for x in range(0, width, 20) for y in range(0, height, 20)}
        assert len(colors) > 8, "Task panel must not be a blank canvas"
        bridge.errorRaised.emit("无法解密当前用户的凭据文件")
        for _ in range(10):
            application.processEvents()
        error_dialog = window.findChild(QObject, "errorDialog")
        assert error_dialog is not None and error_dialog.property("visible") is True
        error_text = window.findChild(QObject, "errorText")
        assert error_text is not None and "无法解密" in error_text.property("text")
        assert not warnings, warnings
    finally:
        window.close()
        engine.deleteLater()
        application.processEvents()
