from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.memory_vector_store import SQLiteMemoryVectorStore, create_memory_vector_store
from deskpilot.memory.session_store import SessionStore


def force_local_sqlite_vector_store() -> None:
    for name in ("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"):
        os.environ[name] = ""
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"
    os.environ["MEMORY_VECTOR_PROVIDER"] = "sqlite"


def test_sqlite_vector_store_roundtrip(base: Path) -> None:
    store = create_memory_vector_store("sqlite", base / "vectors")
    assert isinstance(store, SQLiteMemoryVectorStore)
    store.upsert(
        memory_id="mem_1",
        embedding=[1.0, 0.0, 0.0],
        document="第二阶段优先实现会话记忆系统",
        metadata={"memory_type": "decision"},
    )
    store.upsert(
        memory_id="mem_2",
        embedding=[0.0, 1.0, 0.0],
        document="邮件助手顺延",
        metadata={"memory_type": "decision"},
    )

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert store.count() == 2
    assert results[0].memory_id == "mem_1"


def test_memory_store_uses_vector_store_and_pending_approval(base: Path) -> None:
    vector_store = create_memory_vector_store("sqlite", base / "vectors_pending")
    store = MemoryStore(base / "indexes_pending" / "memory.sqlite", base / "workspace_pending", vector_store=vector_store)
    pending = store.add_memory(
        scope="user",
        memory_type="preference",
        content="用户偏好：我希望回答时先给结论。",
        source_session_id="sess_test",
        confidence=0.72,
    )

    assert pending is not None
    assert pending.status == "pending"
    assert store.stats()["vectors"] == 1
    assert not store.search("回答时先给什么", session_id="sess_test")

    assert store.approve_memory(pending.memory_id)
    results = store.search("回答时先给什么", session_id="sess_test")
    assert any(item.memory_id == pending.memory_id for item in results)


def test_negative_memory_filter() -> None:
    extractor = MemoryExtractor()
    items = extractor.extract(
        user_message="只是举例，不要记住：第二阶段改成做小游戏。",
        assistant_message="知道了。",
        session_id="sess_test",
        source_message_ids=["msg_1", "msg_2"],
    )

    assert items == []


def test_decision_conflict_supersedes_old_memory(base: Path) -> None:
    store = MemoryStore(
        base / "indexes" / "memory.sqlite",
        base / "workspace",
        vector_provider="sqlite",
    )
    old = store.add_memory(
        scope="workspace",
        memory_type="decision",
        content="项目决策：DeskPilot 第二阶段优先实现网页搜索。",
        source_session_id="sess_a",
        confidence=0.9,
    )
    new = store.add_memory(
        scope="workspace",
        memory_type="decision",
        content="项目决策：DeskPilot 第二阶段改成优先实现会话记忆系统。",
        source_session_id="sess_b",
        confidence=0.9,
    )

    assert old is not None
    assert new is not None
    assert store.get_memory(old.memory_id).status == "superseded"  # type: ignore[union-attr]
    results = store.search("第二阶段优先实现什么", session_id="sess_b", top_k=5)
    assert any(item.memory_id == new.memory_id for item in results)
    assert all(item.memory_id != old.memory_id for item in results)


def test_preferences_jsonl_written(base: Path) -> None:
    session_store = SessionStore(base / "sessions")
    session = session_store.create_session("偏好文件测试")
    session_store.append_session_item(
        session.session_id,
        "preference",
        {"content": "用户偏好：回答先给结论。"},
    )

    preference_file = base / "sessions" / session.session_id / "preferences.jsonl"
    assert preference_file.exists()
    assert "回答先给结论" in preference_file.read_text(encoding="utf-8")


def main() -> None:
    force_local_sqlite_vector_store()
    with tempfile.TemporaryDirectory(prefix="deskpilot_memory_p1_test_") as tmp:
        base = Path(tmp)
        test_sqlite_vector_store_roundtrip(base)
        test_memory_store_uses_vector_store_and_pending_approval(base)
        test_negative_memory_filter()
        test_decision_conflict_supersedes_old_memory(base)
        test_preferences_jsonl_written(base)
    print("Memory P1 tests passed.")


if __name__ == "__main__":
    main()
