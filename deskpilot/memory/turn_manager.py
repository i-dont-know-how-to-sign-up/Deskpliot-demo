from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_compactor import MemoryCompactor
from .memory_extractor import MemoryExtractor
from .memory_store import MemoryStore
from .policy import MemoryPolicy
from .session_store import SessionStore


@dataclass(frozen=True)
class MemoryTurnResult:
    extracted: int
    compacted: bool
    skipped: bool
    reason: str


class MemoryTurnManager:
    """封装回答结束后的 Gate、抽取、持久化和滚动压缩。"""

    def __init__(self, extractor: MemoryExtractor, store: MemoryStore, session_store: SessionStore,
                 compactor: MemoryCompactor, policy: MemoryPolicy | None = None) -> None:
        self.extractor = extractor
        self.store = store
        self.session_store = session_store
        self.compactor = compactor
        self.policy = policy or MemoryPolicy()

    def process(
        self, *, user_message: str, assistant_message: str, session_id: str,
        source_message_ids: list[str], grounded: bool, steps: list[Any],
        pending_action: dict[str, Any] | None = None,
    ) -> MemoryTurnResult:
        should_extract, reason = self.policy.should_extract(
            user_message, grounded=grounded, steps=steps, pending_action=pending_action,
        )
        new_memories = []
        if should_extract:
            candidates = self.extractor.extract(
                user_message=user_message, assistant_message=assistant_message,
                session_id=session_id, source_message_ids=source_message_ids, grounded=grounded,
            )
            for item in candidates:
                memory = self.store.add_memory(**item)
                if memory:
                    new_memories.append(memory)
                    self.session_store.append_session_item(session_id, memory.memory_type, memory.to_dict())
        compacted, _summary = self.compactor.compact_if_needed(
            session_id, memories=self.store.list_memories(session_id=session_id, limit=50),
        )
        return MemoryTurnResult(len(new_memories), compacted, not should_extract, reason)
