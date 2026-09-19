from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..core.api_clients import cosine_similarity
from ..core.config import MEMORY_VECTOR_DIR, ensure_dirs


@dataclass
class VectorSearchResult:
    memory_id: str
    score: float


class MemoryVectorStore:
    provider_name = "base"

    def upsert(self, memory_id: str, embedding: list[float], document: str, metadata: dict) -> None:
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[VectorSearchResult]:
        raise NotImplementedError

    def delete(self, memory_id: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class SQLiteMemoryVectorStore(MemoryVectorStore):
    provider_name = "sqlite"

    def __init__(self, db_file: Path | None = None):
        ensure_dirs()
        self.db_file = db_file or (MEMORY_VECTOR_DIR / "memory_vectors.sqlite")
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert(self, memory_id: str, embedding: list[float], document: str, metadata: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into memory_vectors (memory_id, embedding, document, metadata)
                values (?, ?, ?, ?)
                on conflict(memory_id) do update set
                  embedding = excluded.embedding,
                  document = excluded.document,
                  metadata = excluded.metadata
                """,
                (
                    memory_id,
                    json.dumps(embedding),
                    document,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[VectorSearchResult]:
        if not query_embedding:
            return []
        with self._connect() as conn:
            rows = conn.execute("select memory_id, embedding from memory_vectors").fetchall()
        scored: list[VectorSearchResult] = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding"] or "[]")
            except json.JSONDecodeError:
                continue
            score = cosine_similarity(query_embedding, embedding)
            scored.append(VectorSearchResult(memory_id=row["memory_id"], score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def delete(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("delete from memory_vectors where memory_id = ?", (memory_id,))

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) as n from memory_vectors").fetchone()
        return int(row["n"]) if row else 0

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memory_vectors (
                    memory_id text primary key,
                    embedding text not null,
                    document text not null,
                    metadata text not null
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


class ChromaMemoryVectorStore(MemoryVectorStore):
    provider_name = "chroma"

    def __init__(self, persist_dir: Path | None = None, collection_name: str = "deskpilot_memories"):
        ensure_dirs()
        self.persist_dir = persist_dir or (MEMORY_VECTOR_DIR / "chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("chromadb is not installed") from exc
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, memory_id: str, embedding: list[float], document: str, metadata: dict) -> None:
        if not embedding:
            return
        self.collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[self._clean_metadata(metadata)],
        )

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[VectorSearchResult]:
        if not query_embedding:
            return []
        results = self.collection.query(query_embeddings=[query_embedding], n_results=max(top_k, 1))
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        output: list[VectorSearchResult] = []
        for memory_id, distance in zip(ids, distances):
            score = 1.0 / (1.0 + float(distance))
            output.append(VectorSearchResult(memory_id=str(memory_id), score=score))
        return output

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def count(self) -> int:
        return int(self.collection.count())

    def _clean_metadata(self, metadata: dict) -> dict:
        cleaned: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = json.dumps(value, ensure_ascii=False)
        return cleaned


def create_memory_vector_store(
    provider: str | None = None,
    base_dir: Path | None = None,
) -> MemoryVectorStore:
    ensure_dirs()
    provider = (provider or os.getenv("MEMORY_VECTOR_PROVIDER") or "auto").strip().lower()
    base_dir = base_dir or MEMORY_VECTOR_DIR
    if provider in {"chroma", "auto"}:
        try:
            return ChromaMemoryVectorStore(base_dir / "chroma")
        except RuntimeError:
            if provider == "chroma":
                raise
    return SQLiteMemoryVectorStore(base_dir / "memory_vectors.sqlite")
