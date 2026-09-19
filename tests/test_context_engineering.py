from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.context import ContextBuilder, ContextConfig, ContextPacket
from deskpilot.context.budget import estimate_tokens
from deskpilot.context.compressor import ContextCompressor
from deskpilot.memory.memory_compactor import MemoryCompactor
from deskpilot.memory.memory_models import MemoryItem, SessionMessage
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.core.agent import DocumentQAAgent
from deskpilot.rag.vector_index import DocumentIndex


def configure_offline() -> None:
    for name in ("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"):
        os.environ[name] = ""
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"
    os.environ["MEMORY_VECTOR_PROVIDER"] = "sqlite"


def message(index: int, content: str) -> SessionMessage:
    return SessionMessage(
        message_id=f"msg_{index}",
        session_id="sess_test",
        role="user" if index % 2 == 0 else "assistant",
        content=content,
        created_at=(datetime.now(timezone.utc) + timedelta(seconds=index)).isoformat(),
    )


def test_token_estimator() -> None:
    assert estimate_tokens("上下文工程") >= 5
    assert estimate_tokens("context engineering 2026") >= 3
    assert estimate_tokens('{"ok": true}') > 0


def test_latest_messages_survive_small_budget() -> None:
    builder = ContextBuilder(ContextConfig(max_tokens=160, reserve_ratio=0.0, per_message_tokens=80))
    messages = [message(index, f"第{index}轮 " + "较长内容" * 20) for index in range(8)]
    result = builder.assemble("最后决定是什么", "静态规则" * 100, "旧摘要" * 100, messages, [])

    assert "第7轮" in result.text
    assert result.estimated_tokens <= 160
    assert result.quality["context_utilization"] <= 1.0


def test_structure_and_quality_are_observable() -> None:
    memory = MemoryItem(
        memory_id="mem_1",
        scope="workspace",
        memory_type="decision",
        content="P1 优先降低上下文成本",
        retrieval_score=0.9,
    )
    result = ContextBuilder(ContextConfig(max_tokens=500, reserve_ratio=0.1)).assemble(
        "下一步是什么", "规则", "摘要", [message(1, "先完成 P0")], [memory]
    )

    assert "## State" in result.text
    assert "### Recent Conversation" in result.text
    assert "### Retrieved Memories" in result.text
    assert result.quality["required_retention"] == 1.0
    assert result.quality["kind_distribution"]["memory"] == 1


def test_tool_output_compression_keeps_error_tail() -> None:
    packet = ContextPacket(
        packet_id="tool_1",
        kind="tool_output",
        content="status=failed\n" + "\n".join(f"normal output {index}" for index in range(200)) + "\nERROR: timeout at final step",
        source="tool:test",
        estimated_tokens=2000,
    )
    compressed = ContextCompressor(tool_output_tokens=100).compress(packet, 100)

    assert "中间输出已压缩" in compressed.content
    assert "ERROR: timeout" in compressed.content
    assert compressed.estimated_tokens <= 115


def test_compaction_checkpoint_prevents_repeat(base: Path) -> None:
    store = SessionStore(base / "sessions")
    session = store.create_session("checkpoint")
    for index in range(3):
        store.append_message(session.session_id, "user", f"消息 {index}")
    compactor = MemoryCompactor(store, max_messages=3, max_chars=10000)

    compacted, _ = compactor.compact_if_needed(session.session_id)
    repeated, _ = compactor.compact_if_needed(session.session_id)
    assert compacted
    assert not repeated
    assert store.read_compaction_state(session.session_id)["message_count"] == 3


def test_expired_and_completed_memories_are_filtered(base: Path) -> None:
    store = MemoryStore(base / "memory.sqlite", base / "workspace", vector_provider="sqlite")
    expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    store.add_memory("workspace", "task", "上下文工程编写旧报告", confidence=0.9, expires_at=expired)
    store.add_memory("workspace", "task", "上下文工程编写已完成报告", confidence=0.9, tags=["completed"])
    active = store.add_memory("workspace", "task", "上下文工程编写新测试", confidence=0.9)

    results = store.search("上下文工程编写", top_k=10)
    assert active is not None
    assert [item.memory_id for item in results] == [active.memory_id]


def test_recency_breaks_relevance_tie(base: Path) -> None:
    store = MemoryStore(base / "memory2.sqlite", base / "workspace2", vector_provider="sqlite")
    old = store.add_memory(
        "workspace", "fact", "上下文预算采用 token 估算", source_session_id="sess_old", confidence=0.9
    )
    new = store.add_memory(
        "workspace", "fact", "上下文预算采用 token 估算", source_session_id="sess_new", confidence=0.9
    )
    assert old and new
    old_time = (datetime.now(timezone.utc) - timedelta(days=300)).isoformat()
    with store._connect() as conn:  # 测试夹具直接调整时间，验证排序而不等待真实时间流逝。
        conn.execute("update memories set updated_at = ? where memory_id = ?", (old_time, old.memory_id))

    results = store.search("上下文预算 token 估算", top_k=2)
    assert results
    assert results[0].memory_id == new.memory_id


def test_current_question_is_a_single_task_packet(base: Path) -> None:
    agent = DocumentQAAgent(DocumentIndex(base / "agent_index.json"))
    agent.session_store = SessionStore(base / "agent_sessions")
    agent.memory_store = MemoryStore(base / "agent_memory.sqlite", base / "agent_workspace", vector_provider="sqlite")
    session = agent.session_store.create_session("deduplicate current question")
    agent.session_store.append_message(session.session_id, "user", "上一轮背景")
    agent.session_store.append_message(session.session_id, "assistant", "上一轮回答")
    current = "当前问题只应单独传入一次"
    agent.session_store.append_message(session.session_id, "user", current)

    context = agent._load_memory_context(current, session.session_id)
    assert "上一轮背景" in context.text
    assert context.text.count(current) == 1
    assert any(packet.kind == "task" and packet.content == current for packet in context.packets)


def main() -> None:
    configure_offline()
    test_token_estimator()
    test_latest_messages_survive_small_budget()
    test_structure_and_quality_are_observable()
    test_tool_output_compression_keeps_error_tail()
    with tempfile.TemporaryDirectory(prefix="deskpilot_context_test_") as raw:
        base = Path(raw)
        test_compaction_checkpoint_prevents_repeat(base)
        test_expired_and_completed_memories_are_filtered(base)
        test_recency_breaks_relevance_tie(base)
        test_current_question_is_a_single_task_packet(base)
    print("Context engineering tests passed: 8")


if __name__ == "__main__":
    main()
