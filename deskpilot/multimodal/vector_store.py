from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..core.config import INDEX_DIR
from .models import MultimodalChunk, VectorRecord


class MultimodalVectorStore:
    """小型个人知识库使用 SQLite 精确扫描，并严格隔离向量空间和维度。"""

    def __init__(self, database: Path | None = None) -> None:
        self.database = (database or INDEX_DIR / "multimodal_catalog.sqlite3").resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, doc_id TEXT,
                    modality TEXT NOT NULL, text TEXT NOT NULL, source_label TEXT NOT NULL,
                    page_number INTEGER, bbox_json TEXT, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY, owner_type TEXT NOT NULL, owner_id TEXT NOT NULL,
                    modality TEXT NOT NULL, embedding_space TEXT NOT NULL, model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL, dimension INTEGER NOT NULL,
                    vector_json TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vectors_space ON vectors(embedding_space, dimension);
                CREATE INDEX IF NOT EXISTS idx_chunks_asset ON chunks(asset_id);
                """
            )

    def upsert_chunk(self, chunk: MultimodalChunk) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.chunk_id, chunk.asset_id, chunk.doc_id, chunk.modality, chunk.text,
                    chunk.source_label, chunk.page_number,
                    json.dumps(chunk.bbox) if chunk.bbox else None,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                ),
            )

    def upsert_vector(self, record: VectorRecord) -> None:
        if record.dimension != len(record.vector):
            raise ValueError("向量维度元数据与实际向量长度不一致")
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.vector_id, record.owner_type, record.owner_id, record.modality,
                    record.embedding_space, record.model_id, record.model_revision, record.dimension,
                    json.dumps(record.vector), json.dumps(record.metadata, ensure_ascii=False),
                ),
            )

    def search(self, vector: list[float], embedding_space: str, limit: int = 20) -> list[tuple[str, float]]:
        if not vector:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT owner_id, vector_json FROM vectors WHERE embedding_space = ? AND dimension = ?",
                (embedding_space, len(vector)),
            ).fetchall()
        scored = [(row["owner_id"], self._cosine(vector, json.loads(row["vector_json"]))) for row in rows]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:max(1, min(limit, 200))]

    def text_search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        terms = self._query_terms(query)
        if not terms:
            # 中文没有空格时使用双字片段，避免只做整句精确匹配。
            compact = "".join(query.split()).casefold()
            terms = {compact[index:index + 2] for index in range(max(1, len(compact) - 1))}
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id, text FROM chunks WHERE text != ''").fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            content = str(row["text"]).casefold()
            hits = sum(1 for term in terms if term and term in content)
            if hits:
                scored.append((row["chunk_id"], hits / max(1, len(terms))))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:max(1, min(limit, 200))]

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        """英文按词、连续中文按双字片段生成字面检索词项。"""
        terms: set[str] = set()
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.casefold()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                if len(token) == 1:
                    terms.add(token)
                else:
                    terms.update(token[index:index + 2] for index in range(len(token) - 1))
            else:
                terms.add(token)
        return terms

    def get_chunk(self, chunk_id: str) -> MultimodalChunk | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if not row:
            return None
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        return MultimodalChunk(
            chunk_id=row["chunk_id"], asset_id=row["asset_id"], doc_id=row["doc_id"],
            modality=row["modality"], text=row["text"], source_label=row["source_label"],
            page_number=row["page_number"], bbox=tuple(bbox) if bbox else None,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def has_vector(self, owner_id: str, embedding_space: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM vectors WHERE owner_id = ? AND embedding_space = ? LIMIT 1",
                (owner_id, embedding_space),
            ).fetchone()
        return row is not None

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vectors = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            assets = connection.execute("SELECT COUNT(DISTINCT asset_id) FROM chunks").fetchone()[0]
        return {"assets": assets, "chunks": chunks, "vectors": vectors}

    def delete_asset(self, asset_id: str) -> None:
        with self._connect() as connection:
            ids = [row[0] for row in connection.execute(
                "SELECT chunk_id FROM chunks WHERE asset_id = ?", (asset_id,)
            ).fetchall()]
            connection.executemany("DELETE FROM vectors WHERE owner_id = ?", ((value,) for value in ids))
            connection.execute("DELETE FROM vectors WHERE owner_id = ?", (asset_id,))
            connection.execute("DELETE FROM chunks WHERE asset_id = ?", (asset_id,))

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
        return dot / norm if norm else 0.0
