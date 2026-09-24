from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Iterator

from .memory_models import MemoryItem
from .memory_vector_store import MemoryVectorStore, create_memory_vector_store
from .policy import MemoryPolicy
from ..core.api_clients import OpenAICompatibleClient, cosine_similarity
from ..core.config import MEMORY_INDEX_DIR, MEMORY_VECTOR_DIR, MEMORY_WORKSPACE_DIR, ensure_dirs, load_config
from ..core.encoding_utils import fix_mojibake
from ..core.models import utc_now


CATALOG_FILE = MEMORY_INDEX_DIR / "memory_catalog.sqlite"


class MemoryStore:
    def __init__(
        self,
        db_file: Path = CATALOG_FILE,
        workspace_dir: Path = MEMORY_WORKSPACE_DIR,
        vector_store: MemoryVectorStore | None = None,
        vector_provider: str | None = None,
    ):
        ensure_dirs()
        self.db_file = db_file
        self.workspace_dir = workspace_dir
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.client = OpenAICompatibleClient(load_config())
        vector_base_dir = MEMORY_VECTOR_DIR if db_file == CATALOG_FILE else db_file.parent.parent / "vectors"
        self.vector_store = vector_store or create_memory_vector_store(vector_provider, vector_base_dir)
        self.policy = MemoryPolicy()
        self._init_db()
        self.ensure_workspace_files()

    def ensure_workspace_files(self) -> None:
        templates = {
            "MEMORY.md": "# DeskPilot Workspace Memory\n\n- 当前项目：DeskPilot 个人办公助手 Agent。\n",
            "USER.md": "# User Preferences\n\n",
            "AGENTS.md": "# Agent Rules\n\n- 优先使用本地、可审计、可回放的实现。\n",
            "PERMISSIONS.md": "# Permissions\n\n- API Key、密码、token 禁止写入长期记忆。\n",
            "SKILLS.md": "# Skills\n\n- document_qa：文档解析、索引和带证据问答。\n",
        }
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in templates.items():
            path = self.workspace_dir / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")

    def workspace_text(self) -> str:
        sections: list[str] = []
        for filename in ("MEMORY.md", "USER.md", "AGENTS.md", "PERMISSIONS.md", "SKILLS.md"):
            path = self.workspace_dir / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(f"### {filename}\n{content}")
        return "\n\n".join(sections)

    def add_memory(
        self,
        scope: str,
        memory_type: str,
        content: str,
        source_session_id: str | None = None,
        source_message_ids: list[str] | None = None,
        confidence: float = 0.75,
        status: str = "active",
        tags: list[str] | None = None,
        expires_at: str | None = None,
    ) -> MemoryItem | None:
        content = fix_mojibake(content).strip()
        if not content or self._is_sensitive(content):
            return None
        duplicate = self._find_duplicate(content, memory_type, source_session_id)
        if duplicate:
            return duplicate
        status = self._normalize_status(status, memory_type, confidence)
        now = utc_now()
        item = MemoryItem(
            memory_id=f"mem_{uuid.uuid4().hex}",
            scope=scope,
            memory_type=memory_type,
            content=content,
            source_session_id=source_session_id,
            source_message_ids=source_message_ids or [],
            confidence=confidence,
            status=status,
            tags=tags or [],
            created_at=now,
            updated_at=now,
            expires_at=expires_at or self.policy.default_expires_at(scope, memory_type, tags),
        )
        item.embedding = self._embed_text(content)
        self._supersede_conflicts(item)
        with self._connect() as conn:
            conn.execute(
                """
                insert into memories (
                    memory_id, scope, memory_type, content, source_session_id,
                    source_message_ids, confidence, status, tags, created_at,
                    updated_at, expires_at, embedding
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._to_row(item),
            )
        self.vector_store.upsert(
            memory_id=item.memory_id,
            embedding=item.embedding,
            document=item.content,
            metadata={
                "scope": item.scope,
                "memory_type": item.memory_type,
                "source_session_id": item.source_session_id or "",
                "status": item.status,
                "confidence": item.confidence,
                "tags": item.tags,
            },
        )
        return item

    def list_memories(
        self,
        session_id: str | None = None,
        status: str | None = "active",
        limit: int = 100,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        args: list[object] = []
        if session_id:
            clauses.append("(source_session_id = ? or scope in ('workspace', 'user'))")
            args.append(session_id)
        if status:
            clauses.append("status = ?")
            args.append(status)
        where = " where " + " and ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"select * from memories{where} order by updated_at desc limit ?",
                (*args, limit),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def search(self, query: str, session_id: str | None = None, top_k: int = 5) -> list[MemoryItem]:
        query = query.strip()
        if not query:
            return []
        query_embedding = self._embed_text(query)
        vector_results = self.vector_store.search(query_embedding, top_k=max(top_k * 12, 30))
        vector_scores = {item.memory_id: item.score for item in vector_results}
        candidates = self.list_memories(session_id=session_id, status="active", limit=500)
        if not candidates:
            return []
        scored: list[tuple[float, MemoryItem]] = []
        query_lower = query.lower()
        for item in candidates:
            if self._is_expired_or_inactive(item):
                continue
            vector_score = vector_scores.get(item.memory_id, 0.0)
            keyword_score = self._keyword_score(query_lower, item.content.lower())
            semantic_lexical = max(0.0, min(1.0, vector_score)) * 0.65 + min(1.0, keyword_score / 0.45) * 0.35
            recency = self._recency_score(item.updated_at, item.memory_type)
            type_bonus = 0.03 if item.memory_type in {"decision", "preference"} else 0.0
            context_signal = recency * 0.15 + item.confidence * 0.10 + type_bonus
            # 新近性只能重排相关记忆，不能让完全无关的新 task 凭时间奖励进入结果。
            score = semantic_lexical * 0.75 + context_signal * min(1.0, semantic_lexical * 4.0)
            item.retrieval_score = score
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return self._mmr_memories([(score, item) for score, item in scored if score > 0.08], top_k)

    def _mmr_memories(self, candidates: list[tuple[float, MemoryItem]], top_k: int) -> list[MemoryItem]:
        remaining = list(candidates[: max(top_k * 4, top_k)])
        selected: list[tuple[float, MemoryItem]] = []
        while remaining and len(selected) < top_k:
            best_index = 0
            best_value = float("-inf")
            for index, (relevance, item) in enumerate(remaining):
                redundancy = max((self._memory_similarity(item, chosen) for _, chosen in selected), default=0.0)
                value = 0.70 * relevance - 0.30 * redundancy
                if value > best_value:
                    best_index, best_value = index, value
            selected.append(remaining.pop(best_index))
        return [item for _, item in selected]

    def _memory_similarity(self, left: MemoryItem, right: MemoryItem) -> float:
        if left.embedding and right.embedding:
            return max(0.0, cosine_similarity(left.embedding, right.embedding))
        left_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2}", left.content.casefold()))
        right_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2}", right.content.casefold()))
        union = left_terms | right_terms
        return len(left_terms & right_terms) / len(union) if union else 0.0

    def list_pending_memories(self, session_id: str | None = None, limit: int = 100) -> list[MemoryItem]:
        return self.list_memories(session_id=session_id, status="pending", limit=limit)

    def approve_memory(self, memory_id: str) -> bool:
        return self.update_status(memory_id, "active")

    def update_status(self, memory_id: str, status: str) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                "update memories set status = ?, updated_at = ? where memory_id = ?",
                (status, utc_now(), memory_id),
            )
            changed = result.rowcount > 0
        if changed:
            item = self.get_memory(memory_id)
            if item and status == "active":
                self.vector_store.upsert(
                    memory_id=item.memory_id,
                    embedding=item.embedding,
                    document=item.content,
                    metadata={
                        "scope": item.scope,
                        "memory_type": item.memory_type,
                        "source_session_id": item.source_session_id or "",
                        "status": item.status,
                        "confidence": item.confidence,
                        "tags": item.tags,
                    },
                )
            elif status == "deleted":
                self.vector_store.delete(memory_id)
        return changed

    def delete_memory(self, memory_id: str) -> bool:
        return self.update_status(memory_id, "deleted")

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute("select * from memories where memory_id = ?", (memory_id,)).fetchone()
        return self._from_row(row) if row else None

    def stats(self) -> dict[str, int | str]:
        with self._connect() as conn:
            rows = conn.execute("select status, count(*) as n from memories group by status").fetchall()
        stats: dict[str, int | str] = {
            "provider": self.vector_store.provider_name,
            "vectors": self.vector_store.count(),
            "total": 0,
            "active": 0,
            "pending": 0,
            "deleted": 0,
            "superseded": 0,
        }
        for row in rows:
            status = row["status"]
            count = int(row["n"])
            stats[str(status)] = count
            stats["total"] = int(stats["total"]) + count
        return stats

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memories (
                    memory_id text primary key,
                    scope text not null,
                    memory_type text not null,
                    content text not null,
                    source_session_id text,
                    source_message_ids text not null,
                    confidence real not null,
                    status text not null,
                    tags text not null,
                    created_at text not null,
                    updated_at text not null,
                    expires_at text,
                    embedding text not null
                )
                """
            )
            conn.execute("create index if not exists idx_memories_status on memories(status)")
            conn.execute("create index if not exists idx_memories_session on memories(source_session_id)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _to_row(self, item: MemoryItem) -> tuple:
        return (
            item.memory_id,
            item.scope,
            item.memory_type,
            item.content,
            item.source_session_id,
            json.dumps(item.source_message_ids, ensure_ascii=False),
            item.confidence,
            item.status,
            json.dumps(item.tags, ensure_ascii=False),
            item.created_at,
            item.updated_at,
            item.expires_at,
            json.dumps(item.embedding),
        )

    def _from_row(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            scope=row["scope"],
            memory_type=row["memory_type"],
            content=row["content"],
            source_session_id=row["source_session_id"],
            source_message_ids=json.loads(row["source_message_ids"] or "[]"),
            confidence=float(row["confidence"]),
            status=row["status"],
            tags=json.loads(row["tags"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            embedding=json.loads(row["embedding"] or "[]"),
        )

    def _is_expired_or_inactive(self, item: MemoryItem) -> bool:
        normalized_tags = {str(tag).strip().lower() for tag in item.tags}
        if normalized_tags.intersection({"completed", "cancelled", "obsolete", "expired"}):
            return True
        if not item.expires_at:
            return False
        try:
            expires_at = datetime.fromisoformat(item.expires_at.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at <= datetime.now(timezone.utc)
        except (TypeError, ValueError):
            return False

    def _recency_score(self, updated_at: str, memory_type: str) -> float:
        half_life_days = {
            "task": 14.0,
            "artifact": 30.0,
            "fact": 120.0,
            "decision": 180.0,
            "preference": 365.0,
        }.get(memory_type, 90.0)
        try:
            timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400)
            return math.exp(-math.log(2) * age_days / half_life_days)
        except (TypeError, ValueError):
            return 0.0

    def _embed_text(self, text: str) -> list[float]:
        try:
            return self.client.embed([text])[0]
        except Exception:
            return []

    def _keyword_score(self, query: str, content: str) -> float:
        terms = [term for term in query.replace("？", " ").replace("，", " ").split() if term]
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        for term in chinese_terms:
            terms.extend(term[i : i + 2] for i in range(max(len(term) - 1, 0)))
            terms.extend(term[i : i + 3] for i in range(max(len(term) - 2, 0)))
        terms = list(dict.fromkeys(term for term in terms if len(term) >= 2))
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in content)
        return min(0.45, hits / max(len(terms), 1) * 0.45)

    def _find_duplicate(self, content: str, memory_type: str, source_session_id: str | None) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from memories
                where content = ? and memory_type = ? and coalesce(source_session_id, '') = coalesce(?, '')
                  and status != 'deleted'
                limit 1
                """,
                (content, memory_type, source_session_id),
            ).fetchone()
        return self._from_row(row) if row else None

    def _is_sensitive(self, content: str) -> bool:
        lowered = content.lower()
        sensitive_markers = ("api_key", "apikey", "secret", "password", "token=", "bearer ")
        return any(marker in lowered for marker in sensitive_markers) or bool(re.search(r"sk-[a-zA-Z0-9]{12,}", content))

    def _normalize_status(self, status: str, memory_type: str, confidence: float) -> str:
        if status in {"deleted", "superseded", "archived"}:
            return status
        if status == "pending":
            return status
        if memory_type == "preference" and confidence < 0.82:
            return "pending"
        if confidence < 0.6:
            return "pending"
        return "active"

    def _supersede_conflicts(self, incoming: MemoryItem) -> None:
        if incoming.memory_type not in {"decision", "preference"}:
            return
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from memories
                where memory_type = ? and status = 'active'
                """,
                (incoming.memory_type,),
            ).fetchall()
            for row in rows:
                item = self._from_row(row)
                if self._same_memory_topic(incoming, item):
                    conn.execute(
                        "update memories set status = 'superseded', updated_at = ? where memory_id = ?",
                        (utc_now(), item.memory_id),
                    )
                    self.vector_store.delete(item.memory_id)

    def _same_memory_topic(self, incoming: MemoryItem, existing: MemoryItem) -> bool:
        incoming_topics = {str(tag) for tag in incoming.tags if str(tag).startswith("topic:")}
        existing_topics = {str(tag) for tag in existing.tags if str(tag).startswith("topic:")}
        if incoming_topics and existing_topics:
            return bool(incoming_topics & existing_topics)
        left = self._topic_terms(incoming.content)
        right = self._topic_terms(existing.content)
        overlap = len(left & right) / max(1, min(len(left), len(right)))
        semantic = cosine_similarity(incoming.embedding, existing.embedding) if incoming.embedding and existing.embedding else 0.0
        return overlap >= 0.65 or (overlap >= 0.35 and semantic >= 0.35)

    @staticmethod
    def _topic_terms(content: str) -> set[str]:
        stop = {"用户", "偏好", "项目", "决策", "使用", "采用", "改为", "调整", "希望", "以后"}
        lowered = content.casefold()
        terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_.+-]{2,}", lowered))
        for span in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            terms.update(span[index:index + 2] for index in range(len(span) - 1))
            terms.update(span[index:index + 3] for index in range(max(0, len(span) - 2)))
        return {term for term in terms if term not in stop}
