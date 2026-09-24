from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import RagCatalog
from .chunker import CHUNKER_VERSION
from .document_parsers import PARSER_VERSION
from ..core.models import Chunk, Document


def inspect_legacy_index(path: Path) -> tuple[dict[str, Document], dict[str, Chunk], dict[str, Any]]:
    """读取旧 JSON 快照并执行无副作用的引用完整性校验。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_documents = data.get("documents", {})
    raw_chunks = data.get("chunks", {})
    if not isinstance(raw_documents, dict) or not isinstance(raw_chunks, dict):
        raise ValueError("index.json 必须包含对象类型的 documents 和 chunks")
    documents = {str(key): Document.from_dict(value) for key, value in raw_documents.items()}
    chunks = {str(key): Chunk.from_dict(value) for key, value in raw_chunks.items()}
    if any(key != value.doc_id for key, value in documents.items()):
        raise ValueError("documents 的键与 doc_id 不一致")
    if any(key != value.chunk_id for key, value in chunks.items()):
        raise ValueError("chunks 的键与 chunk_id 不一致")
    orphans = sorted(value.chunk_id for value in chunks.values() if value.doc_id not in documents)
    if orphans:
        raise ValueError(f"发现 {len(orphans)} 个片段引用不存在的文档：{orphans[:5]}")
    structured = sum(bool(value.metadata.get("parent_chunk_id")) for value in chunks.values())
    report = {
        "documents": len(documents),
        "chunks": len(chunks),
        "structured_chunks": structured,
        "legacy_fallback_chunks": len(chunks) - structured,
        "source": str(path.resolve()),
    }
    return documents, chunks, report


def migrate_legacy_index(source: Path, catalog_path: Path, dry_run: bool = False) -> dict[str, Any]:
    documents, chunks, report = inspect_legacy_index(source)
    if dry_run:
        return {**report, "mode": "dry-run", "written": False}

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if catalog_path.exists():
        with closing(sqlite3.connect(catalog_path)) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = catalog_path.with_name(f"{catalog_path.name}.backup_{stamp}")
        shutil.copy2(catalog_path, backup)

    # 在同一目录生成完整临时库，校验后再替换，避免半迁移状态被桌面端读取。
    with tempfile.TemporaryDirectory(prefix="deskpilot_rag_migrate_", dir=str(catalog_path.parent)) as raw:
        temporary = Path(raw) / catalog_path.name
        catalog = RagCatalog(temporary)
        by_document: dict[str, list[Chunk]] = {}
        for chunk in chunks.values():
            by_document.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, document in documents.items():
            catalog.commit_document(
                document,
                sorted(by_document.get(doc_id, []), key=lambda value: value.position),
                sentence_nodes=[],
                parent_chunks=[],
                parser_version=str(document.metadata.get("parser_version", PARSER_VERSION)),
                chunker_version=str(document.metadata.get("chunker_version", CHUNKER_VERSION)),
                embedding_space=str(document.metadata.get("embedding_space", "legacy-unknown")),
            )
        stats = catalog.stats()
        if stats["documents"] != len(documents) or stats["chunks"] != len(chunks):
            raise RuntimeError(f"迁移后计数校验失败：{stats}")
        temporary.replace(catalog_path)
    # 目标库可能曾使用 WAL；原子替换后不能让旧 sidecar 被 SQLite 重放。
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(catalog_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    return {**report, "mode": "apply", "written": True,
            "catalog": str(catalog_path.resolve()), "backup": str(backup) if backup else ""}


def rollback_catalog(catalog_path: Path, backup: Path) -> dict[str, Any]:
    if not backup.is_file():
        raise FileNotFoundError(backup)
    if catalog_path.exists():
        with closing(sqlite3.connect(catalog_path)) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        safety = catalog_path.with_name(f"{catalog_path.name}.before_rollback")
        shutil.copy2(catalog_path, safety)
    shutil.copy2(backup, catalog_path)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(catalog_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    return {"mode": "rollback", "catalog": str(catalog_path.resolve()),
            "backup": str(backup.resolve()), "restored": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="DeskPilot RAG 旧 JSON 索引迁移与回滚工具")
    parser.add_argument("--from-json", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", type=Path)
    args = parser.parse_args()
    if args.rollback:
        result = rollback_catalog(args.catalog, args.rollback)
    elif args.from_json and (args.dry_run or args.apply):
        result = migrate_legacy_index(args.from_json, args.catalog, dry_run=args.dry_run)
    else:
        parser.error("请使用 --from-json 配合 --dry-run/--apply，或使用 --rollback")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
