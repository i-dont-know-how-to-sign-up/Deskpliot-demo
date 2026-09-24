from __future__ import annotations

import tempfile
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.context import ContextBuilder
from deskpilot.memory.memory_compactor import MemoryCompactor
from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.rag.vector_index import DocumentIndex


def force_local_fallback() -> None:
    for name in (
        "DASHSCOPE_API_KEY",
        "LLM_API_KEY",
        "EMBEDDING_API_KEY",
    ):
        os.environ[name] = ""
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"


def test_session_store_roundtrip(base: Path) -> None:
    store = SessionStore(base / "sessions")
    session = store.create_session("记忆系统测试")
    first = store.append_message(session.session_id, "user", "我决定第二阶段先做会话记忆系统。")
    second = store.append_message(session.session_id, "assistant", "已记录这个项目决策。")
    messages = store.read_messages(session.session_id)

    assert len(messages) == 2
    assert messages[0].message_id == first.message_id
    assert messages[1].message_id == second.message_id

    store.write_summary(session.session_id, "# 摘要\n\n- 第二阶段先做会话记忆系统。")
    assert "会话记忆系统" in store.read_summary(session.session_id)

    store.update_message_metadata(
        session.session_id,
        second.message_id,
        {"steps": [{"name": "route_intent", "status": "success", "detail": "direct_answer"}]},
    )
    restored = store.read_messages(session.session_id)[1]
    assert restored.metadata["steps"][0]["detail"] == "direct_answer"
    exported = store.export_session(session.session_id, base / "session.md")
    assert "### Steps" in exported.read_text(encoding="utf-8")
    assert "route_intent" in exported.read_text(encoding="utf-8")


def test_memory_store_search_and_delete(base: Path) -> None:
    store = MemoryStore(base / "indexes" / "memory.sqlite", base / "workspace")
    memory = store.add_memory(
        scope="workspace",
        memory_type="decision",
        content="DeskPilot 第二阶段优先实现会话记忆系统，网页搜索顺延。",
        source_session_id="sess_test",
        confidence=0.9,
        tags=["stage_2"],
    )

    assert memory is not None
    assert store.add_memory("user", "preference", "DASHSCOPE_API_KEY=should_not_store") is None

    results = store.search("第二阶段优先做什么", session_id="sess_test", top_k=3)
    assert any("会话记忆系统" in item.content for item in results)

    assert store.delete_memory(memory.memory_id)
    assert not any(item.memory_id == memory.memory_id for item in store.list_memories(status="active"))


def test_compactor_force(base: Path) -> None:
    session_store = SessionStore(base / "sessions")
    memory_store = MemoryStore(base / "indexes" / "memory.sqlite", base / "workspace")
    session = session_store.create_session("压缩测试")
    session_store.append_message(session.session_id, "user", "我决定第二阶段先做会话记忆系统。")
    memory = memory_store.add_memory(
        scope="workspace",
        memory_type="decision",
        content="第二阶段先做会话记忆系统。",
        source_session_id=session.session_id,
        confidence=0.9,
    )

    compactor = MemoryCompactor(session_store)
    compacted, summary = compactor.compact_if_needed(session.session_id, memories=[memory] if memory else [], force=True)

    assert compacted
    assert "会话记忆系统" in summary


def test_agent_memory_integration_without_api_key(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    agent = DocumentQAAgent(index)
    agent.session_store = SessionStore(base / "sessions")
    agent.memory_store = MemoryStore(base / "indexes" / "memory.sqlite", base / "workspace")
    agent.memory_extractor = MemoryExtractor()
    agent.memory_compactor = MemoryCompactor(agent.session_store, max_messages=100, max_chars=100000)
    agent.context_assembler = ContextBuilder()

    session = agent.session_store.create_session("集成测试")
    first = agent.answer(
        "我决定 DeskPilot 第二阶段优先实现会话记忆系统，网页搜索顺延。",
        session_id=session.session_id,
    )
    second = agent.answer("我们第二阶段优先做什么？", session_id=session.session_id)

    assert first.session_id == session.session_id
    assert second.session_id == session.session_id
    assert "会话记忆系统" in second.answer
    assert agent.session_store.message_count(session.session_id) == 4


def main() -> None:
    force_local_fallback()
    with tempfile.TemporaryDirectory(prefix="deskpilot_memory_test_") as tmp:
        base = Path(tmp)
        test_session_store_roundtrip(base)
        test_memory_store_search_and_delete(base)
        test_compactor_force(base)
        test_agent_memory_integration_without_api_key(base)
    print("Memory P0 smoke tests passed.")


if __name__ == "__main__":
    main()
