from __future__ import annotations

import html
import zipfile
from pathlib import Path
from typing import Any

from .permissions import assess_path_write, require_permission


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
    target = Path(path).expanduser().resolve(strict=False)
    suffix = _normalize_format(file_format or target.suffix)
    if not suffix:
        suffix = ".txt"
        target = target.with_suffix(suffix)
    if suffix not in TEXT_FORMATS and suffix not in {".docx", ".pdf"}:
        raise ValueError(f"Unsupported output format: {suffix}")

    decision = assess_path_write(target, operation="write_file", additional_safe_roots=safe_roots)
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "written": False,
            "path": str(target),
            "format": suffix,
            **permission,
        }
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
    if suffix in TEXT_FORMATS:
        target.write_text(str(content), encoding="utf-8", newline="\n")
    elif suffix == ".docx":
        _write_docx(target, str(content))
    elif suffix == ".pdf":
        _write_pdf(target, str(content))

    return {
        "ok": True,
        "written": True,
        "path": str(target),
        "format": suffix,
        "size": target.stat().st_size,
        "permission": decision.to_dict(),
        "message": "File written successfully.",
    }


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
