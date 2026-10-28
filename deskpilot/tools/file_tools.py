from __future__ import annotations

import shutil
import os
import hashlib
import sys
from pathlib import Path
from typing import Any

from .permissions import assess_file_move, assess_path_write, is_confirmed, require_permission
from .write_tools import _sha256, _stage_path
from ..tasks.control import get_execution_control
from ..tasks.retry import UnknownSideEffect

from ..rag.document_parsers import parse_document


FILE_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("resume", ("简历", "resume", "cv")),
    ("meeting", ("会议", "纪要", "meeting", "minutes")),
    ("research", ("研究", "调研", "research", "report", "analysis")),
    ("invoice", ("发票", "invoice", "收据", "receipt")),
    ("finance", ("财务", "预算", "budget", "expense", "费用")),
    ("project", ("方案", "计划", "project", "roadmap", "proposal")),
    ("presentation", ("汇报", "演示", "ppt", "slides", "presentation")),
    ("notes", ("笔记", "note", "notes", "记录")),
    ("paper", ("论文", "paper", "thesis", "survey")),
]


SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pptx", ".xlsx", ".csv", ".pdf"}


def delete_file(path: str | Path, *, expected_sha256: str = "", confirm: bool = False,
                safe_roots: list[Path] | None = None) -> dict[str, Any]:
    """Delete only one regular file after approval, bound to its previewed content hash."""
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("Only an existing regular file can be deleted")
    target = raw.resolve(strict=True)
    if os.getenv("DESKPILOT_RELEASE_MODE") == "1" and getattr(sys, "frozen", False):
        install_root = Path(sys.executable).resolve().parent
        if target == install_root or install_root in target.parents:
            raise ValueError("The application installation directory is protected; remove stray files manually")
    decision = assess_path_write(target, "delete_file", additional_safe_roots=safe_roots)
    if decision.blocked:
        raise PermissionError("; ".join(decision.reasons))
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError("File changed after approval preview; request a new approval")
    if not is_confirmed(confirm):
        return {
            "path": str(target), "sha256": actual_sha256, "deleted": False,
            "permission": {"requires_confirmation": True, "risk_level": "high",
                           "reasons": ["Deleting a file is irreversible in this tool.", *decision.reasons]},
        }
    if not expected_sha256:
        raise ValueError("Approved deletion requires the previewed file hash")
    target.unlink()
    return {"path": str(target), "sha256": actual_sha256, "deleted": True}

LEADING_ACTION_WORDS = (
    "请帮我把",
    "请帮我",
    "请你帮我",
    "请你",
    "麻烦帮我",
    "麻烦",
    "帮我把",
    "帮我",
    "请",
    "当前目录下的",
    "当目录下的",
    "本目录下的",
    "目录下的",
    "然后",
    "接着",
    "打开",
    "读取",
    "查看",
    "阅读",
    "总结",
    "分析",
    "概括",
    "说明",
    "提炼",
    "展示",
    "把",
    "将",
)


def resolve_document_path(target: str | Path, search_roots: list[Path] | None = None) -> Path:
    candidate = Path(str(target).strip().strip('"').strip("'"))
    # 调用方显式给出的目录代表用户限定的搜索范围，必须优先于进程 cwd；
    # 否则项目根目录的同名文件会污染隔离工作区或指定文件夹的解析结果。
    roots = list(search_roots or []) + [Path.cwd()]
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    if candidate.exists() and not search_roots:
        return candidate.resolve()
    normalized = _normalize_document_query(candidate.name)
    if normalized and normalized != candidate.name:
        normalized_candidate = candidate.with_name(normalized)
        if normalized_candidate.exists() and not search_roots:
            return normalized_candidate.resolve()
        for root in roots:
            direct = (root / normalized_candidate).resolve()
            if direct.exists():
                return direct
    for root in roots:
        direct = (root / candidate).resolve()
        if direct.exists():
            return direct
    filename = candidate.name.lower()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() == filename:
                return path.resolve()
    if normalized and normalized.lower() != filename:
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.name.lower() == normalized.lower():
                    return path.resolve()
    raise FileNotFoundError(f"Document not found: {target}")


def read_document(path: str | Path, search_roots: list[Path] | None = None) -> dict[str, Any]:
    document_path = resolve_document_path(path, search_roots=search_roots)
    document = parse_document(document_path)
    return {
        "path": str(document_path),
        "document": document.to_dict(),
        "content": document.content,
        "title": document.title,
        "source_type": document.source_type,
        "size": document.metadata.get("size", 0),
    }


def _normalize_document_query(value: str) -> str:
    text = str(value or "").strip().strip('"').strip("'").strip()
    text = text.replace(" ", "")
    text = text.strip("，。；;,:：")
    changed = True
    while changed and text:
        changed = False
        for prefix in LEADING_ACTION_WORDS:
            if text.startswith(prefix) and len(text) > len(prefix) + 3:
                candidate = text[len(prefix) :]
                if "." in candidate:
                    text = candidate
                    changed = True
                    break
    return text


def scan_folder(folder: Path) -> list[dict[str, Any]]:
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    items: list[dict[str, Any]] = []
    for path in sorted((p for p in folder.rglob("*") if p.is_file()), key=lambda item: str(item).lower()):
        category, reason = classify_path(path)
        items.append(
            {
                "path": str(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
                "supported": path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS,
                "category": category,
                "reason": reason,
                "suggested_folder": category,
            }
        )
    return items


def classify_folder(folder: Path) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    items = scan_folder(folder)
    categories: dict[str, int] = {}
    for item in items:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    return {
        "folder": str(folder),
        "total_files": len(items),
        "categories": categories,
        "items": items,
    }


def build_organization_plan(folder: Path, target_root: Path | None = None, max_files: int = 200) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    target_root = target_root.expanduser().resolve() if target_root else folder / "_organized"
    items = scan_folder(folder)
    moves: list[dict[str, Any]] = []
    for item in items[:max_files]:
        source = Path(item["path"])
        category = item["category"]
        destination = target_root / category / source.name
        if source.resolve() == destination.resolve():
            continue
        moves.append(
            {
                "source": str(source),
                "destination": str(destination),
                "category": category,
                "reason": item["reason"],
                "supported": item["supported"],
            }
        )
    return {
        "folder": str(folder),
        "target_root": str(target_root),
        "total_candidates": len(items),
        "move_count": len(moves),
        "moves": moves,
    }


def apply_file_move(
    source: Path,
    destination: Path,
    confirm: bool = False,
    safe_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.is_dir():
        raise IsADirectoryError(f"Source must be a file: {source}")
    decision = assess_file_move(source, destination, additional_safe_roots=safe_roots)
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "confirmed": False,
            "source": str(source),
            "destination": str(destination),
            **permission,
        }
    if destination.exists():
        raise FileExistsError(f"Move target already exists: {destination}")
    source_sha256 = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = _stage_path(destination)
    linked = False
    try:
        shutil.copy2(source, stage)
        if _sha256(stage) != source_sha256 or _sha256(source) != source_sha256:
            raise RuntimeError("Source changed while preparing move")
        os.link(stage, destination)
        linked = True
        if _sha256(source) != source_sha256:
            raise UnknownSideEffect("移动目标已建立，但源文件在提交期间改变；需人工核对")
        source.unlink()
    except Exception as exc:
        if linked and not isinstance(exc, UnknownSideEffect):
            raise UnknownSideEffect("文件移动已部分提交，需人工核对源和目标") from exc
        raise
    finally:
        stage.unlink(missing_ok=True)
    control = get_execution_control()
    return {
        "ok": True,
        "confirmed": True,
        "source": str(source),
        "destination": str(destination),
        "before_sha256": source_sha256,
        "after_sha256": _sha256(destination),
        "compensatable": bool(getattr(control, "active_operation_id", "")),
        "message": "File moved successfully.",
        "permission": decision.to_dict(),
    }


def build_supported_file_index(index, folder: Path) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    results = index.add_folder(folder)
    return {
        "folder": str(folder),
        "indexed_documents": len(results),
        "documents": [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "path": document.path,
                "chunks": chunk_count,
            }
            for document, chunk_count in results
        ],
    }


def classify_path(path: Path) -> tuple[str, str]:
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    joined = f"{path.name} {stem}"
    for category, keywords in FILE_CATEGORY_RULES:
        if any(keyword.lower() in joined for keyword in keywords):
            return category, f"Matched filename keyword for {category}."
    if suffix in {".docx", ".md", ".markdown", ".txt", ".pdf"}:
        return "documents", "Generic document file."
    if suffix in {".pptx"}:
        return "presentation", "Presentation file extension."
    if suffix in {".xlsx", ".csv"}:
        return "spreadsheet", "Spreadsheet file extension."
    if suffix in {".zip", ".7z", ".rar", ".tar", ".gz"}:
        return "archive", "Archive file extension."
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return "image", "Image file extension."
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".go", ".rs", ".ipynb"}:
        return "code", "Code or notebook file."
    if suffix:
        return "other", "Fallback extension bucket."
    return "other", "No extension."
