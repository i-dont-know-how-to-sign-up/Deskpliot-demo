from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

from .profile import profile_home, save_settings, set_profile_home
from .secrets import SecretStore


def _test_model(api_key: str, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 8,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return "连接成功" if response.status == 200 else f"服务返回 HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return f"连接失败：HTTP {exc.code}。检查 Key、模型权限与余额。"
    except (urllib.error.URLError, TimeoutError):
        return "连接失败：请检查网络或代理设置。"


def run_first_run() -> bool:
    """Collect settings before core.config is imported; return False if cancelled."""
    if (profile_home() / "settings.json").exists():
        return True
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
        QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = QDialog()
    dialog.setWindowTitle("DeskPilot · 首次设置")
    dialog.setMinimumWidth(530)
    dialog.setStyleSheet("QDialog { background: #151a23; color: #edf2f7; } "
                         "QLineEdit { background: #202937; color: #edf2f7; padding: 7px; border: 1px solid #405064; } "
                         "QPushButton { padding: 7px 12px; } QLabel { color: #d6e0eb; }")
    layout = QVBoxLayout(dialog)
    title = QLabel("开始使用 DeskPilot")
    title.setStyleSheet("font-size: 19px; font-weight: 600;")
    layout.addWidget(title)
    layout.addWidget(QLabel("模型凭据仅保存在当前 Windows 用户的 DPAPI 加密存储中。"))
    form = QFormLayout()
    home = QLineEdit(str(profile_home()))
    workspace = QLineEdit(str(profile_home() / "data" / "workspace"))
    key = QLineEdit()
    key.setEchoMode(QLineEdit.EchoMode.Password)
    model = QLineEdit("qwen3.7-plus")
    for label, field in (("数据目录", home), ("授权工作区", workspace), ("模型 API Key", key), ("模型名称", model)):
        field.setAccessibleName(label)
        form.addRow(label, field)
    home_browse = QPushButton("选择…")
    home_browse.setToolTip("选择存储索引、会话和任务的目录")

    def choose_home() -> None:
        selected = QFileDialog.getExistingDirectory(dialog, "选择 DeskPilot 数据目录", home.text())
        if selected:
            home.setText(selected)
            workspace.setText(str(Path(selected) / "data" / "workspace"))

    home_browse.clicked.connect(choose_home)
    row = QHBoxLayout()
    row.addLayout(form)
    row.addWidget(home_browse, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addLayout(row)
    cloud = QCheckBox("允许将选中的图片发送至云端视觉模型")
    cloud.setChecked(False)
    layout.addWidget(cloud)
    status = QLabel("可先测试连接；保存后可通过发布维护命令更换配置。")
    status.setWordWrap(True)
    layout.addWidget(status)
    check = QPushButton("测试模型连接")
    class CheckSignals(QObject):
        finished = Signal(str)

    signals = CheckSignals(dialog)

    def connection_result(result: str) -> None:
        status.setText(result)
        check.setEnabled(True)

    signals.finished.connect(connection_result)

    def test_connection() -> None:
        if not key.text().strip():
            status.setText("请先输入 API Key。")
            return
        check.setEnabled(False)
        status.setText("正在测试模型连接…")
        api_key, model_name = key.text().strip(), model.text().strip()
        threading.Thread(target=lambda: signals.finished.emit(_test_model(api_key, model_name)), daemon=True).start()

    check.clicked.connect(test_connection)
    layout.addWidget(check, alignment=Qt.AlignmentFlag.AlignLeft)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)

    def save() -> None:
        try:
            selected = Path(home.text().strip()).expanduser()
            root = Path(workspace.text().strip()).expanduser()
            if not selected.is_absolute() or not root.is_absolute() or not model.text().strip():
                raise ValueError("请填写绝对数据目录、绝对工作区和模型名称。")
            if not key.text().strip():
                raise ValueError("请填写模型 API Key。")
            selected = set_profile_home(selected)
            root.mkdir(parents=True, exist_ok=True)
            settings = {
                "QWEN_MODEL": model.text().strip(),
                "QWEN_ENABLE_THINKING": "false",
                "TOOL_SAFE_ROOTS": str(root.resolve()),
                "ALLOW_CLOUD_IMAGE_UPLOAD": "true" if cloud.isChecked() else "false",
            }
            SecretStore(selected / "secrets.dpapi").set("DASHSCOPE_API_KEY", key.text().strip())
            save_settings(settings, selected)
            dialog.accept()
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(dialog, "设置未保存", str(exc))

    buttons.accepted.connect(save)
    buttons.rejected.connect(dialog.reject)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    if not accepted and QApplication.instance() is app:
        # qt_app.main() reuses this QApplication after a successful setup.
        dialog.close()
    return accepted
