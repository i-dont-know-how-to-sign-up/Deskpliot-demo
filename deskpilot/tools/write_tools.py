from __future__ import annotations

import html
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any

from .permissions import assess_path_write, is_foreign_windows_path, require_permission
from ..tasks.control import get_execution_control


TEXT_FORMATS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".log",
}


def write_file(
    path: str | Path,
    content: str,
    file_format: str | None = None,
    overwrite: bool = False,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    # 必须先用原始字符串鉴权。POSIX 的 Path.resolve() 会把 C:\... 错当成
    # 当前目录下的相对文件名，进而绕过工作区外写入的人工确认。
    raw_path = str(path).strip()
    foreign_windows_path = is_foreign_windows_path(raw_path)
    lexical_path = PureWindowsPath(raw_path) if foreign_windows_path else Path(raw_path).expanduser()
    suffix = _normalize_format(file_format or lexical_path.suffix)
    if not suffix:
        suffix = ".txt"
        lexical_path = lexical_path.with_suffix(suffix)
    if suffix not in TEXT_FORMATS and suffix not in {".docx", ".pdf"}:
        raise ValueError(f"Unsupported output format: {suffix}")

    display_path = str(lexical_path)
    decision = assess_path_write(display_path, operation="write_file", additional_safe_roots=safe_roots)
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "written": False,
            "path": display_path,
            "format": suffix,
            **permission,
        }

    # 确认仅代表风险授权，并不能让当前系统执行另一平台的路径语义。
    # 显式拒绝可避免在 POSIX 工作目录中创建名为 ``C:\...`` 的伪文件。
    if foreign_windows_path:
        return {
            "ok": True,
            "written": False,
            "path": display_path,
            "format": suffix,
            "permission": decision.to_dict(),
            "message": "The Windows path cannot be written from the current POSIX host.",
        }

    target = Path(lexical_path).resolve(strict=False)
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"Target must be a file: {target}")
    if target.exists() and not overwrite:
        return {
            "ok": True,
            "written": False,
            "path": str(target),
            "format": suffix,
            "permission": decision.to_dict(),
            "message": "Target already exists. Retry with overwrite=True after checking the file.",
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    control = get_execution_control()
    operation_id = str(getattr(control, "active_operation_id", "") or "")
    before_sha256 = _sha256(target) if target.is_file() else ""
    backup_path = ""
    stage = _stage_path(target)
    try:
        if suffix in TEXT_FORMATS:
            stage.write_text(str(content), encoding="utf-8", newline="\n")
        elif suffix == ".docx":
            _write_docx(stage, str(content))
        elif suffix == ".pdf":
            _write_pdf(stage, str(content))
        after_sha256 = _sha256(stage)
        if not after_sha256:
            raise RuntimeError("Staged file validation failed")
        if target.exists() and overwrite:
            if operation_id and control is not None and getattr(control, "task_store", None):
                backup = control.task_store.path.parent / "effect_backups" / operation_id
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                if _sha256(backup) != before_sha256:
                    raise RuntimeError("Source changed while preparing backup")
                backup_path = str(backup)
            if _sha256(target) != before_sha256:
                raise RuntimeError("Target changed before commit")
            os.replace(stage, target)
        else:
            # 同卷 hard link 的创建具有排他性；不能用 os.replace 覆盖刚出现的文件。
            os.link(stage, target)
            stage.unlink()
    finally:
        stage.unlink(missing_ok=True)

    return {
        "ok": True,
        "written": True,
        "path": str(target),
        "format": suffix,
        "size": target.stat().st_size,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "backup_path": backup_path,
        "compensatable": bool(operation_id),
        "permission": decision.to_dict(),
        "message": "File written successfully.",
    }


def _stage_path(target: Path) -> Path:
    handle, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent)
    os.close(handle)
    return Path(name)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_markdown(
    path: str | Path,
    content: str,
    overwrite: bool = False,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    return write_file(path, content, file_format=".md", overwrite=overwrite, confirm=confirm, safe_roots=safe_roots)


def write_text(
    path: str | Path,
    content: str,
    overwrite: bool = False,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    return write_file(path, content, file_format=".txt", overwrite=overwrite, confirm=confirm, safe_roots=safe_roots)


def write_docx(
    path: str | Path,
    content: str,
    overwrite: bool = False,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    return write_file(path, content, file_format=".docx", overwrite=overwrite, confirm=confirm, safe_roots=safe_roots)


def write_pdf(
    path: str | Path,
    content: str,
    overwrite: bool = False,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    return write_file(path, content, file_format=".pdf", overwrite=overwrite, confirm=confirm, safe_roots=safe_roots)


def _normalize_format(value: str | None) -> str:
    if not value:
        return ""
    normalized = str(value).strip().lower()
    if normalized and not normalized.startswith("."):
        normalized = "." + normalized
    return normalized


def _write_docx(path: Path, content: str) -> None:
    paragraphs = content.splitlines() or [""]
    body = "\n".join(
        f'<w:p><w:r><w:t xml:space="preserve">{html.escape(paragraph)}</w:t></w:r></w:p>'
        for paragraph in paragraphs
    )
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)


def _write_pdf(path: Path, content: str) -> None:
    try:
        import pymupdf as pdf_lib
    except ImportError as exc:
        try:
            import fitz as pdf_lib
        except ImportError:
            raise RuntimeError("PDF export requires PyMuPDF. Install it with: pip install PyMuPDF") from exc

    document = pdf_lib.open()
    try:
        page = document.new_page()
        rect = pdf_lib.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, content, fontsize=11, fontname="helv")
        document.save(str(path))
    finally:
        document.close()
