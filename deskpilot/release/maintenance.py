from __future__ import annotations

import sys
import subprocess
import threading
from pathlib import Path

from .diagnostics import export_diagnostics
from .profile import profile_home, set_profile_home
from .storage import create_backup, migrate_legacy, restore_backup


def run_maintenance() -> int:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import (
        QApplication, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = QDialog()
    dialog.setWindowTitle("DeskPilot · 备份与迁移")
    dialog.setMinimumWidth(570)
    dialog.setStyleSheet("QDialog { background: #151a23; color: #edf2f7; } "
                         "QLineEdit { background: #202937; color: #edf2f7; padding: 7px; border: 1px solid #405064; } "
                         "QPushButton { padding: 7px 12px; } QLabel { color: #d6e0eb; }")
    layout = QVBoxLayout(dialog)
    heading = QLabel("备份、恢复与旧版数据迁移")
    heading.setStyleSheet("font-size: 18px; font-weight: 600;")
    layout.addWidget(heading)
    note = QLabel("操作前请退出 DeskPilot 主程序和后台 worker。恢复/迁移只写入全新目录，不覆盖原数据。")
    note.setWordWrap(True)
    layout.addWidget(note)
    layout.addWidget(QLabel("当前数据目录：" + str(profile_home() / "data")))
    form = QFormLayout()
    archive = QLineEdit()
    archive.setPlaceholderText("备份文件路径（.dpb）")
    source = QLineEdit()
    source.setPlaceholderText("恢复时填 .dpb；迁移时填旧 data 目录")
    target = QLineEdit()
    target.setPlaceholderText("恢复/迁移目标：全新的 DeskPilot 主目录")
    password = QLineEdit()
    password.setEchoMode(QLineEdit.EchoMode.Password)
    password.setPlaceholderText("至少 12 个字符；丢失后无法恢复备份")
    repeat_password = QLineEdit()
    repeat_password.setEchoMode(QLineEdit.EchoMode.Password)
    for label, field in (("新备份文件", archive), ("恢复/迁移来源", source), ("新主目录", target), ("备份口令", password)):
        form.addRow(label, field)
    form.addRow("确认备份口令", repeat_password)
    layout.addLayout(form)
    buttons = QHBoxLayout()
    backup_button = QPushButton("创建加密备份")
    restore_button = QPushButton("恢复备份")
    migrate_button = QPushButton("迁移旧 data")
    diagnostics_button = QPushButton("导出诊断")
    for button in (backup_button, restore_button, migrate_button, diagnostics_button):
        buttons.addWidget(button)
    layout.addLayout(buttons)
    models_row = QHBoxLayout()
    siglip_button = QPushButton("下载 SigLIP2（Full 包）")
    ocr_button = QPushButton("下载 PaddleOCR（Full 包）")
    models_row.addWidget(siglip_button)
    models_row.addWidget(ocr_button)
    layout.addLayout(models_row)
    model_status = QLabel("基础包不含本地模型运行时；云端图文问答无需下载。")
    model_status.setWordWrap(True)
    layout.addWidget(model_status)

    class ModelSignals(QObject):
        finished = Signal(bool, str)

    signals = ModelSignals(dialog)

    def model_finished(ok: bool, detail: str) -> None:
        model_status.setText(("已安装：" if ok else "下载失败：") + detail)
        siglip_button.setEnabled(True)
        ocr_button.setEnabled(True)

    signals.finished.connect(model_finished)

    def download_model(kind: str) -> None:
        siglip_button.setEnabled(False)
        ocr_button.setEnabled(False)
        model_status.setText("正在下载模型…")

        def execute() -> None:
            command = ([sys.executable, "install-model", kind] if getattr(sys, "frozen", False)
                       else [sys.executable, "-m", "deskpilot.release", "install-model", kind])
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
                detail = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()[-500:]
                signals.finished.emit(result.returncode == 0, detail or f"exit={result.returncode}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                signals.finished.emit(False, str(exc))

        threading.Thread(target=execute, daemon=True).start()

    siglip_button.clicked.connect(lambda: download_model("siglip2"))
    ocr_button.clicked.connect(lambda: download_model("ocr"))

    def report(action, message: str) -> None:
        try:
            if action():
                QMessageBox.information(dialog, "操作完成", message)
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            QMessageBox.warning(dialog, "操作未完成", str(exc))

    def backup_action() -> bool:
        path = archive.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(dialog, "选择备份文件", str(profile_home().parent / "DeskPilot-backup.dpb"), "DeskPilot Backup (*.dpb)")
        if not path:
            return False
        if password.text() != repeat_password.text():
            raise ValueError("两次输入的备份口令不一致")
        create_backup(profile_home() / "data", Path(path), password.text())
        archive.setText(path)
        return True

    def restore_action() -> bool:
        if not source.text().strip() or not target.text().strip():
            raise ValueError("请填写备份文件和新的 DeskPilot 主目录")
        new_home = Path(target.text().strip()).expanduser().resolve()
        restore_backup(Path(source.text().strip()), new_home / "data", password.text())
        set_profile_home(new_home)
        return True

    def migrate_action() -> bool:
        if not source.text().strip() or not target.text().strip():
            raise ValueError("请填写旧 data 目录和新的 DeskPilot 主目录")
        new_home = Path(target.text().strip()).expanduser().resolve()
        migrate_legacy(Path(source.text().strip()), new_home / "data")
        set_profile_home(new_home)
        return True

    def diagnostics_action() -> bool:
        path, _ = QFileDialog.getSaveFileName(dialog, "导出诊断", str(profile_home().parent / "DeskPilot-diagnostics.json"), "JSON (*.json)")
        if path:
            export_diagnostics(Path(path))
            return True
        return False

    backup_button.clicked.connect(lambda: report(backup_action, "加密备份已创建。请保管口令；未备份模型凭据。"))
    restore_button.clicked.connect(lambda: report(restore_action, "备份已恢复到新目录。重启 DeskPilot 并重新配置模型凭据。"))
    migrate_button.clicked.connect(lambda: report(migrate_action, "旧数据已复制到新目录。原目录未删除，重启后完成首次设置。"))
    diagnostics_button.clicked.connect(lambda: report(diagnostics_action, "诊断信息已导出，不包含会话正文或凭据。"))
    close_button = QPushButton("关闭")
    close_button.clicked.connect(dialog.accept)
    layout.addWidget(close_button)
    dialog.exec()
    return 0
