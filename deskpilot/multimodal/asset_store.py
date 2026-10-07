from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..core.config import DATA_DIR
from .image_processor import ImageProcessor, ProcessedImage
from .models import MediaAsset


class AssetStore:
    """原图资产与元数据目录。数据库不保存图片二进制或 Base64。"""

    def __init__(self, root: Path | None = None, processor: ImageProcessor | None = None) -> None:
        self.root = (root or DATA_DIR / "workspace" / "assets").resolve()
        self.thumbnail_root = self.root / "thumbnails"
        self.database = self.root / "assets.sqlite3"
        self.processor = processor or ImageProcessor()
        self.root.mkdir(parents=True, exist_ok=True)
        self.thumbnail_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, media_type TEXT NOT NULL,
                    original_path TEXT NOT NULL, source_path TEXT, width INTEGER NOT NULL,
                    height INTEGER NOT NULL, size_bytes INTEGER NOT NULL, thumbnail_path TEXT NOT NULL,
                    source_kind TEXT NOT NULL, parent_doc_id TEXT, page_number INTEGER,
                    sensitivity TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
                )"""
            )

    def ingest_file(self, path: Path, **metadata: Any) -> MediaAsset:
        processed = self.processor.process_file(path)
        return self._store(processed, source_path=str(path.expanduser().resolve()), **metadata)

    def ingest_bytes(self, content: bytes, source_path: str | None = None, **metadata: Any) -> MediaAsset:
        return self._store(self.processor.process_bytes(content), source_path=source_path, **metadata)

    def _store(
        self, processed: ProcessedImage, *, source_path: str | None,
        source_kind: str = "chat", parent_doc_id: str | None = None,
        page_number: int | None = None, sensitivity: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> MediaAsset:
        digest = hashlib.sha256(processed.content).hexdigest()
        existing = self.get_by_sha256(digest)
        if existing:
            return existing
        asset_id = f"asset_{digest[:20]}"
        asset_dir = self.root / digest[:2]
        asset_dir.mkdir(parents=True, exist_ok=True)
        original_path = asset_dir / f"{digest}{processed.extension}"
        thumbnail_path = self.thumbnail_root / f"{digest}.jpg"
        original_path.write_bytes(processed.content)
        thumbnail_path.write_bytes(self.processor.thumbnail(processed.content))
        asset = MediaAsset(
            asset_id=asset_id, sha256=digest, media_type=processed.media_type,
            original_path=str(original_path), source_path=source_path,
            width=processed.width, height=processed.height, size_bytes=processed.source_size,
            thumbnail_path=str(thumbnail_path), source_kind=source_kind,
            parent_doc_id=parent_doc_id, page_number=page_number,
            sensitivity=sensitivity, metadata=metadata or {},
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset.asset_id, asset.sha256, asset.media_type, asset.original_path, asset.source_path,
                    asset.width, asset.height, asset.size_bytes, asset.thumbnail_path, asset.source_kind,
                    asset.parent_doc_id, asset.page_number, asset.sensitivity,
                    json.dumps(asset.metadata, ensure_ascii=False), asset.created_at,
                ),
            )
        return asset

    def get(self, asset_id: str) -> MediaAsset | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return self._from_row(row) if row else None

    def get_by_sha256(self, digest: str) -> MediaAsset | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assets WHERE sha256 = ?", (digest,)).fetchone()
        return self._from_row(row) if row else None

    def list_assets(self, limit: int = 500) -> list[MediaAsset]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assets ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 5000)),)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MediaAsset:
        return MediaAsset(
            asset_id=row["asset_id"], sha256=row["sha256"], media_type=row["media_type"],
            original_path=row["original_path"], source_path=row["source_path"], width=row["width"],
            height=row["height"], size_bytes=row["size_bytes"], thumbnail_path=row["thumbnail_path"],
            source_kind=row["source_kind"], parent_doc_id=row["parent_doc_id"],
            page_number=row["page_number"], sensitivity=row["sensitivity"],
            metadata=json.loads(row["metadata_json"] or "{}"), created_at=row["created_at"],
        )
