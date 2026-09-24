from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.rag.migrations import migrate_legacy_index, rollback_catalog
from deskpilot.rag.retrieval import RetrievalFilters, RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


FIXTURES = ROOT / "eval" / "dataset" / "fixtures" / "docs"


def _index(paths: list[str], root: Path) -> DocumentIndex:
    index = DocumentIndex(root / "index.json")
    index.client.embed = lambda texts: [local_hash_embedding(value) for value in texts]  # type: ignore[method-assign]
    index._configured_embedding_space = lambda: "local-hash-v1:384"  # type: ignore[method-assign]
    index._embedding_space = "local-hash-v1:384"
    for name in paths:
        index.add_file(FIXTURES / name)
    return index


def _event(index: DocumentIndex, name: str) -> dict[str, object]:
    trace = index.last_trace.to_dict()
    return next(value for value in trace["events"] if value["stage"] == name)


def test_sentence_window_expands_hit_without_crossing_heading() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_window_") as raw, patch.dict(os.environ, {
        "RAG_CHILD_TOKENS": "40", "RAG_CHILD_MAX_TOKENS": "70", "RAG_MIN_CHUNK_TOKENS": "10",
    }):
        index = _index(["rag_p2_nebula.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "NEBULA-712 为什么不能直接重试？", ("NEBULA-712 过期令牌 直接重试",),
            top_k=3, context_expansion="sentence_window", sentence_window_size=2,
        ))
        text = "\n".join(item.text for item in result)
        assert "过期令牌" in text and "旧主节点" in text
        # 排名靠后的另一条证据可以来自其他标题，但命中窗口本身不能跨标题污染。
        assert "NEBULA-408" not in result[0].text
        assert _event(index, "context_expansion")["detail"]["strategy"] == "sentence_window"


def test_parent_expansion_keeps_child_and_obeys_budget() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_parent_") as raw:
        index = _index(["rag_p2_aurora.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "Aurora Cache 节点重启故障恢复", ("快照 回放 失效日志",), top_k=2,
            context_expansion="parent", max_context_tokens=120,
        ))
        assert result and "快照" in result[0].text and "回放" in result[0].text
        assert result[0].metadata["expansion"] == "parent"
        assert int(result[0].metadata["context_tokens"]) <= 130


def test_reranker_promotes_body_match_and_records_provider() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_rerank_") as raw:
        index = _index(["rag_p2_aurora.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "传播延迟 回源峰值 已知限制", ("传播延迟 回源峰值",), top_k=2,
            context_expansion="parent", rerank_provider="lexical",
        ))
        assert result and "传播延迟" in result[0].text
        event = _event(index, "rerank")
        assert event["detail"]["effective_provider"] == "lexical"
        assert result[0].metadata["reranker"] == "lexical"


def test_provider_failure_falls_back_to_rrf() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_fallback_") as raw:
        index = _index(["rag_p2_nebula.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "NEBULA-712", ("NEBULA-712",), top_k=2, rerank_provider="not-a-provider",
        ))
        event = _event(index, "rerank")
        assert result
        assert event["detail"]["fallback"] is True
        assert event["detail"]["effective_provider"] == "disabled_fallback"


def test_cross_encoder_never_downloads_without_local_model() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_cross_encoder_") as raw, patch.dict(
        os.environ, {"RAG_RERANK_MODEL": ""}
    ):
        index = _index(["rag_p2_nebula.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "NEBULA-712", ("NEBULA-712",), top_k=1, rerank_provider="cross_encoder",
        ))
        event = _event(index, "rerank")
        assert result and event["detail"]["fallback"] is True
        assert "拒绝隐式下载" in str(event["detail"]["error"])


def test_relevance_threshold_can_return_true_empty_result() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_threshold_") as raw:
        index = _index(["rag_p2_aurora.md"], Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "Aurora ZXQ-999 完全不存在的协议", ("Aurora ZXQ-999",), top_k=2,
            use_dense=False, rerank_provider="lexical", relevance_threshold=0.9,
        ))
        assert result == []
        threshold = _event(index, "relevance_threshold")
        assert threshold["input_count"] > 0 and threshold["output_count"] == 0


def test_mmr_preserves_explicit_multi_document_coverage() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_mmr_") as raw:
        index = _index(["rag_p2_falcon_fairness.md", "rag_p2_falcon_recovery.md"], Path(raw))
        doc_ids = tuple(index.documents)
        result = index.search_hybrid(RetrievalRequest(
            "比较 Falcon Scheduler 公平性和过载恢复",
            ("deficit counter aging", "接纳率 延迟队列"),
            filters=RetrievalFilters(doc_ids=doc_ids), top_k=4, context_expansion="parent",
        ))
        assert {item.doc_id for item in result} == set(doc_ids)
        assert _event(index, "final_mmr")["detail"]["required_documents"] == 2


def test_legacy_migration_dry_run_apply_and_rollback() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p2_migrate_") as raw:
        root = Path(raw)
        index = _index(["rag_p2_nebula.md"], root)
        target = root / "migrated.sqlite3"
        dry = migrate_legacy_index(index.index_file, target, dry_run=True)
        assert dry["written"] is False and not target.exists()
        applied = migrate_legacy_index(index.index_file, target)
        assert applied["written"] is True and target.exists()
        backup = root / "manual.backup"
        backup.write_bytes(target.read_bytes())
        restored = rollback_catalog(target, backup)
        assert restored["restored"] is True


def test_rag_p2_dataset_schema_and_files() -> None:
    cases = [json.loads(line) for line in (ROOT / "eval" / "dataset" / "rag_p2_cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    assert len(cases) == 12 and len({case["id"] for case in cases}) == 12
    for case in cases:
        assert case["query"] and case["relevant_sources"] and case["required_terms"]
        assert case["expansion"] in {"none", "sentence_window", "parent"}
        for relative in case["corpus"]:
            assert (ROOT / relative).is_file()


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"RAG P2 tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
