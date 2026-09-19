from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ..core.encoding_utils import fix_mojibake, text_quality_score
from ..core.models import Document, make_doc_id


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pptx", ".xlsx", ".csv", ".pdf"}
PARSER_VERSION = "structured-v1"


def parse_document(path: Path) -> Document:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        content = _read_text_file(path)
    elif suffix == ".docx":
        content = _parse_docx(path)
    elif suffix == ".pptx":
        content = _parse_pptx(path)
    elif suffix == ".xlsx":
        content = _parse_xlsx(path)
    elif suffix == ".pdf":
        content = _parse_pdf(path)
    else:
        content = ""

    content = fix_mojibake(_normalize_text(content))
    if not content.strip():
        content = f"[No extractable text found in {path.name}]"

    return Document(
        doc_id=make_doc_id(path, digest),
        path=str(path),
        title=path.name,
        source_type=suffix.replace(".", ""),
        content=content,
        metadata={"sha256": digest, "size": path.stat().st_size},
    )


def scan_supported_files(folder: Path) -> list[Path]:
    folder = folder.resolve()
    paths: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return fix_mojibake(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return fix_mojibake(path.read_bytes().decode("utf-8", errors="ignore"))


def _parse_docx(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" in archive.namelist():
            parts.extend(_extract_docx_blocks(archive.read("word/document.xml")))
        for name in archive.namelist():
            if name.startswith("word/header") or name.startswith("word/footer"):
                parts.extend(_extract_xml_text(archive.read(name)))
    return "\n".join(parts)


def _extract_docx_blocks(xml_bytes: bytes) -> list[str]:
    """保留 DOCX 标题和表格边界，供后续结构化分块使用。"""
    root = ElementTree.fromstring(xml_bytes)
    body = next((elem for elem in root.iter() if elem.tag.endswith("}body")), root)
    parts: list[str] = []
    for child in body:
        if child.tag.endswith("}p"):
            text = "".join(elem.text or "" for elem in child.iter() if elem.tag.endswith("}t")).strip()
            if not text:
                continue
            style = ""
            for elem in child.iter():
                if elem.tag.endswith("}pStyle"):
                    style = next((value for key, value in elem.attrib.items() if key.endswith("}val")), "")
                    break
            heading = re.search(r"(?:Heading|标题)\s*([1-6])", style, re.IGNORECASE)
            parts.append(f"{'#' * int(heading.group(1))} {text}" if heading else text)
        elif child.tag.endswith("}tbl"):
            rows: list[str] = []
            for row in child:
                if not row.tag.endswith("}tr"):
                    continue
                cells: list[str] = []
                for cell in row:
                    if cell.tag.endswith("}tc"):
                        value = "".join(
                            elem.text or "" for elem in cell.iter() if elem.tag.endswith("}t")
                        ).strip()
                        cells.append(value)
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                parts.append("\n".join(rows))
    return parts


def _parse_pptx(path: Path) -> str:
    slide_texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            (name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")),
            key=_natural_key,
        )
        for idx, name in enumerate(slide_names, start=1):
            texts = _extract_xml_text(archive.read(name))
            if texts:
                slide_texts.append(f"[Slide {idx}]\n" + "\n".join(texts))
    return "\n\n".join(slide_texts)


def _parse_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_names = sorted(
            (name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")),
            key=_natural_key,
        )
        sheets: list[str] = []
        for idx, name in enumerate(sheet_names, start=1):
            rows = _read_sheet_rows(archive.read(name), shared_strings)
            if rows:
                rendered = "\n".join(" | ".join(cell for cell in row if cell) for row in rows[:200])
                sheets.append(f"[Sheet {idx}]\n{rendered}")
        return "\n\n".join(sheets)


def _parse_pdf(path: Path) -> str:
    candidates = []
    pymupdf_text = _parse_pdf_pymupdf(path)
    if pymupdf_text:
        candidates.append(pymupdf_text)
    pypdf_text = _parse_pdf_pypdf(path)
    if pypdf_text:
        candidates.append(pypdf_text)
    if not candidates:
        return _parse_pdf_raw_fallback(path)
    return max(candidates, key=text_quality_score)


def _parse_pdf_pymupdf(path: Path) -> str:
    try:
        import fitz  # type: ignore

        pages: list[str] = []
        with fitz.open(str(path)) as document:
            for idx, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(f"[Page {idx}]\n{text}")
        return "\n\n".join(pages)
    except Exception:
        return ""


def _parse_pdf_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages: list[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {idx}]\n{text}")
        return "\n\n".join(pages)
    except Exception:
        return ""


def _parse_pdf_raw_fallback(path: Path) -> str:
    raw = path.read_bytes()
    decoded = raw.decode("latin-1", errors="ignore")
    candidates = re.findall(r"\(([^()]{3,})\)", decoded)
    return "\n".join(candidates[:200])


def _extract_xml_text(xml_bytes: bytes) -> list[str]:
    root = ElementTree.fromstring(xml_bytes)
    texts: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("}t") and elem.text:
            texts.append(elem.text)
    return texts


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    return _extract_xml_text(archive.read("xl/sharedStrings.xml"))


def _read_sheet_rows(xml_bytes: bytes, shared_strings: list[str]) -> list[list[str]]:
    root = ElementTree.fromstring(xml_bytes)
    rows: list[list[str]] = []
    for row in root.iter():
        if not row.tag.endswith("}row"):
            continue
        values: list[str] = []
        for cell in row:
            if not cell.tag.endswith("}c"):
                continue
            cell_type = cell.attrib.get("t")
            value = ""
            for child in cell:
                if child.tag.endswith("}v") and child.text is not None:
                    value = child.text
                    break
            if cell_type == "s" and value.isdigit():
                idx = int(value)
                value = shared_strings[idx] if idx < len(shared_strings) else value
            values.append(value)
        if values:
            rows.append(values)
    return rows


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]
