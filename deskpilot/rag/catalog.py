from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .blocks import ParentChunk, SentenceNode
from ..core.models import Chunk, Document


SCHEMA_VERSION = 2


def chinese_bigrams(text: str) -> str:
    """为 SQLite unicode61 tokenizer 补充连续中文二元词。"""
    groups = re.findall(r"[\u4e00-\u9fff]+", text)
    return " ".join(group[index:index + 2] for group in groups for index in range(len(group) - 1))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RagCatalog:
    """RAG 元数据与 embedding 缓存。

    P0 仍保留 index.json 作为兼容快照；SQLite 负责版本状态、结构化元数据、
    句子邻接、任务状态和可复用的向量缓存。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    embedding_space TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_path_active
                    ON documents(canonical_path, active);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_chunk_id TEXT,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_active ON chunks(doc_id, active);
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_order INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_parents_doc ON parent_chunks(doc_id, parent_order);
                CREATE TABLE IF NOT EXISTS sentence_nodes (
                    sentence_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_chunk_id TEXT NOT NULL,
                    sentence_order INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    previous_ids_json TEXT NOT NULL,
                    next_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sentence_parent
                    ON sentence_nodes(parent_chunk_id, sentence_order);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT NOT NULL,
                    embedding_space TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(content_hash, embedding_space)
                );
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    title,
                    heading_path,
                    body,
                    tags,
                    zh_bigrams,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
            row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            else:
                connection.execute("UPDATE schema_info SET version=?", (SCHEMA_VERSION,))

    def unchanged_document(self, canonical_path: str, source_hash: str, parser_version: str,
                           chunker_version: str, embedding_space: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT doc_id FROM documents
                   WHERE canonical_path=? AND source_hash=? AND parser_version=? AND chunker_version=?
                     AND embedding_space=? AND active=1 LIMIT 1""",
                (canonical_path, source_hash, parser_version, chunker_version, embedding_space),
            ).fetchone()
        return str(row["doc_id"]) if row else None

    def get_cached_embeddings(self, hashes: list[str], embedding_space: str) -> dict[str, list[float]]:
        if not hashes:
            return {}
        result: dict[str, list[float]] = {}
        with self.connect() as connection:
            for start in range(0, len(hashes), 500):
                batch = hashes[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT content_hash, vector_json FROM embedding_cache "
                    f"WHERE embedding_space=? AND content_hash IN ({placeholders})",
                    (embedding_space, *batch),
                ).fetchall()
                for row in rows:
                    result[str(row["content_hash"])] = [float(value) for value in json.loads(row["vector_json"])]
        return result

    def put_cached_embeddings(self, values: dict[str, list[float]], embedding_space: str) -> None:
        if not values:
            return
        now = _utc_now()
        with self._lock, self.connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO embedding_cache
                   (content_hash, embedding_space, dimensions, vector_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (digest, embedding_space, len(vector), json.dumps(vector), now)
                    for digest, vector in values.items()
                ],
            )

    def commit_document(self, document: Document, chunks: list[Chunk], sentence_nodes: list[SentenceNode],
                        parent_chunks: list[ParentChunk], parser_version: str, chunker_version: str,
                        embedding_space: str) -> None:
        """在单个事务中写入新版本，成功后才停用同路径旧版本。"""
        canonical_path = str(Path(document.path).resolve()) if document.path else document.path
        source_hash = str(document.metadata.get("sha256", ""))
        now = _utc_now()
        with self._lock, self.connect() as connection:
            old_rows = connection.execute(
                "SELECT doc_id FROM documents WHERE canonical_path=? AND active=1 AND doc_id<>?",
                (canonical_path, document.doc_id),
            ).fetchall()
            old_ids = [str(row["doc_id"]) for row in old_rows]
            connection.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id, canonical_path, source_hash, source_type, title, parser_version,
                    chunker_version, embedding_space, active, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (document.doc_id, canonical_path, source_hash, document.source_type, document.title,
                 parser_version, chunker_version, embedding_space,
                 json.dumps(document.metadata, ensure_ascii=False), document.created_at, now),
            )
            connection.execute("DELETE FROM chunks WHERE doc_id=?", (document.doc_id,))
            connection.execute("DELETE FROM chunks_fts WHERE doc_id=?", (document.doc_id,))
            connection.execute("DELETE FROM parent_chunks WHERE doc_id=?", (document.doc_id,))
            connection.execute("DELETE FROM sentence_nodes WHERE doc_id=?", (document.doc_id,))
            connection.executemany(
                """INSERT INTO chunks
                   (chunk_id, doc_id, parent_chunk_id, position, text, source_label, content_hash,
                    token_count, metadata_json, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                [
                    (chunk.chunk_id, chunk.doc_id, chunk.metadata.get("parent_chunk_id"), chunk.position,
                     chunk.text, chunk.source_label, str(chunk.metadata.get("content_hash", "")),
                     int(chunk.metadata.get("token_count", 0)),
                     json.dumps(chunk.metadata, ensure_ascii=False))
                    for chunk in chunks
                ],
            )
            connection.executemany(
                """INSERT INTO chunks_fts
                   (chunk_id, doc_id, title, heading_path, body, tags, zh_bigrams)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        document.title,
                        "/".join(str(value) for value in chunk.metadata.get("heading_path", [])),
                        chunk.text,
                        " ".join(str(value) for value in chunk.metadata.get("tags", [])),
                        chinese_bigrams(" ".join((document.title, chunk.text))),
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                """INSERT INTO parent_chunks
                   (parent_chunk_id, doc_id, parent_order, text, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (parent.parent_chunk_id, parent.doc_id, parent.order, parent.text,
                     json.dumps(parent.metadata, ensure_ascii=False))
                    for parent in parent_chunks
                ],
            )
            connection.executemany(
                """INSERT INTO sentence_nodes
                   (sentence_id, doc_id, parent_chunk_id, sentence_order, text, previous_ids_json,
                    next_ids_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (node.sentence_id, node.doc_id, node.parent_chunk_id, node.order, node.text,
                     json.dumps(node.previous_sentence_ids), json.dumps(node.next_sentence_ids),
                     json.dumps(node.metadata, ensure_ascii=False))
                    for node in sentence_nodes
                ],
            )
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                connection.execute(f"UPDATE documents SET active=0, updated_at=? WHERE doc_id IN ({placeholders})",
                                   (now, *old_ids))
                connection.execute(f"UPDATE chunks SET active=0 WHERE doc_id IN ({placeholders})", old_ids)
                connection.execute(f"DELETE FROM chunks_fts WHERE doc_id IN ({placeholders})", old_ids)

    def sync_search_index(self, documents: dict[str, Document], chunks: dict[str, Chunk]) -> None:
        """兼容 P0 旧 catalog：FTS 缺失或数量不一致时从活动快照重建。"""
        with self._lock, self.connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])
            if count == len(chunks):
                return
            connection.execute("DELETE FROM chunks_fts")
            rows = []
            for chunk in chunks.values():
                document = documents.get(chunk.doc_id)
                if document is None:
                    continue
                rows.append((
                    chunk.chunk_id,
                    chunk.doc_id,
                    document.title,
                    "/".join(str(value) for value in chunk.metadata.get("heading_path", [])),
                    chunk.text,
                    " ".join(str(value) for value in chunk.metadata.get("tags", [])),
                    chinese_bigrams(" ".join((document.title, chunk.text))),
                ))
            connection.executemany(
                """INSERT INTO chunks_fts
                   (chunk_id, doc_id, title, heading_path, body, tags, zh_bigrams)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def sparse_search(self, query: str, allowed_chunk_ids: set[str], limit: int = 30) -> list[tuple[str, float]]:
        """执行字段加权 BM25；返回值已转换为越大越相关。"""
        terms = re.findall(r"[A-Za-z0-9_.@+-]{2,}|[\u4e00-\u9fff]{2,}", query.casefold())
        expanded = list(terms)
        expanded.extend(chinese_bigrams(query).split())
        tokens = list(dict.fromkeys(token.replace('"', '""') for token in expanded if token.strip()))[:40]
        if not tokens or not allowed_chunk_ids:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        rows: list[tuple[str, float]] = []
        allowed = list(allowed_chunk_ids)
        with self.connect() as connection:
            for start in range(0, len(allowed), 500):
                batch = allowed[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                found = connection.execute(
                    f"""SELECT chunk_id,
                               bm25(chunks_fts, 0.0, 0.0, 2.5, 2.0, 1.0, 1.5, 1.2) AS rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND chunk_id IN ({placeholders})
                        ORDER BY rank LIMIT ?""",
                    (expression, *batch, max(1, limit)),
                ).fetchall()
                rows.extend((str(row["chunk_id"]), 1.0 / (1.0 + abs(float(row["rank"])))) for row in found)
        return sorted(rows, key=lambda item: item[1], reverse=True)[:limit]

    def clear_content(self) -> None:
        """保留 schema，清空索引内容；embedding cache 可复用。"""
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM sentence_nodes")
            connection.execute("DELETE FROM parent_chunks")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM index_jobs")

    def stats(self) -> dict[str, int]:
        with self.connect() as connection:
            documents = int(connection.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM chunks WHERE active=1").fetchone()[0])
            sentences = int(connection.execute("SELECT COUNT(*) FROM sentence_nodes").fetchone()[0])
            cached = int(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
        return {"documents": documents, "chunks": chunks, "sentence_nodes": sentences,
                "cached_embeddings": cached}
