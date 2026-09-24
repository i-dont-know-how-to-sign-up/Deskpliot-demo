from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.core.models import Document
from deskpilot.rag.chunker import ChunkingConfig, chunk_document_with_metadata
from deskpilot.rag.vector_index import DocumentIndex


def _config(**changes: object) -> ChunkingConfig:
    values = {
        "policy": "adaptive", "min_chunk_tokens": 1, "target_chunk_tokens": 12,
        "max_chunk_tokens": 24, "parent_target_tokens": 80, "overlap_sentences": 1,
        "semantic_breakpoint_percentile": 70.0, "semantic_buffer_sentences": 1,
        "sentence_window_size": 1,
    }
    values.update(changes)
    return ChunkingConfig(**values)


def test_markdown_uses_heading_and_table_boundaries() -> None:
    content = "# A\n\n第一句。第二句。\n\n# B\n\n| key | value |\n| --- | --- |\n| x | 1 |"
    result = chunk_document_with_metadata(Document("d", "x.md", "x.md", "md", content), _config())
    paths = [chunk.metadata["heading_path"] for chunk in result.chunks]
    assert result.strategy == "structural"
    assert ["A"] in paths and ["B"] in paths
    assert any(chunk.metadata["content_type"] == "table" for chunk in result.chunks)
    assert all(not ({"A", "B"} <= set(chunk.metadata["heading_path"])) for chunk in result.chunks)


def test_semantic_chunker_breaks_topics_and_overlaps_whole_sentence() -> None:
    document = Document("d", "x.txt", "x.txt", "txt", "甲主题第一句。甲主题第二句。乙主题第一句。乙主题第二句。")
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    result = chunk_document_with_metadata(document, _config(target_chunk_tokens=100), lambda _: vectors)
    assert result.strategy == "semantic"
    assert len(result.chunks) == 2
    assert "甲主题第二句。" in result.chunks[0].text
    assert result.chunks[1].text.startswith("甲主题第二句。")
    assert "乙主题第一句。" in result.chunks[1].text


def test_page_and_sentence_window_never_cross_page() -> None:
    content = "[Page 1]\n第一页第一句。第一页第二句。\n\n[Page 2]\n第二页第一句。第二页第二句。"
    result = chunk_document_with_metadata(Document("d", "x.pdf", "x.pdf", "pdf", content), _config(),
                                          lambda texts: [local_hash_embedding(text) for text in texts])
    assert {chunk.metadata["page_number"] for chunk in result.chunks} == {1, 2}
    by_id = {node.sentence_id: node for node in result.sentence_nodes}
    for node in result.sentence_nodes:
        adjacent = node.previous_sentence_ids + node.next_sentence_ids
        assert all(by_id[item].metadata["page_number"] == node.metadata["page_number"] for item in adjacent)


def test_incremental_index_skips_unchanged_and_replaces_old_path_version() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "note.md"
        source.write_text("# 第一版\n\n稳定内容。", encoding="utf-8")
        index = DocumentIndex(root / "index.json")
        calls: list[list[str]] = []

        def embed(texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return [local_hash_embedding(text) for text in texts]

        with patch.object(index.client, "embed", side_effect=embed):
            first, count = index.add_file(source)
            initial_calls = len(calls)
            second, second_count = index.add_file(source)
            assert first.doc_id == second.doc_id and count == second_count
            assert len(calls) == initial_calls
            assert index.last_index_trace["status"] == "skipped_unchanged"
            source.write_text("# 第二版\n\n内容已经更新。", encoding="utf-8")
            third, _ = index.add_file(source)
        assert third.doc_id != first.doc_id
        assert set(index.documents) == {third.doc_id}
        assert {chunk.doc_id for chunk in index.chunks.values()} == {third.doc_id}


def test_catalog_persists_parent_sentence_metadata_and_embedding_cache() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "doc.md"
        source.write_text("# 标题\n\n第一句。第二句。第三句。", encoding="utf-8")
        index = DocumentIndex(root / "index.json")
        with patch.object(index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]):
            index.add_file(source)
        stats = index.stats()
        assert stats["sentence_nodes"] == 3
        assert stats["cached_embeddings"] >= 1
        connection = sqlite3.connect(index.catalog.path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0] == 1
            metadata = json.loads(connection.execute("SELECT metadata_json FROM chunks LIMIT 1").fetchone()[0])
        finally:
            connection.close()
        assert metadata["chunker_version"] == "adaptive-v1"
        assert metadata["heading_path"] == ["标题"]


def test_retrieval_trace_records_candidates_and_selection() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "trace.txt"
        source.write_text("RAG 使用检索证据回答问题。", encoding="utf-8")
        index = DocumentIndex(root / "index.json")
        with patch.object(index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]):
            index.add_file(source)
            found = index.search("什么是 RAG", top_k=2)
        trace = index.last_trace.to_dict()
        assert found and trace["query"] == "什么是 RAG"
        stages = [event["stage"] for event in trace["events"]]
        assert stages[:2] == ["metadata_filter", "rrf_fusion"]
        assert {"rerank", "relevance_threshold", "context_expansion", "final_mmr"}.issubset(stages)
        assert trace["selected_chunk_ids"] == [item.chunk_id for item in found]


def test_retrieval_hint_finds_q_former_without_embedding_call() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "blip-2-note.md"
        source.write_text(
            "# Architecture\n\nQ-Former connects a frozen visual encoder with a frozen language model.",
            encoding="utf-8",
        )
        index = DocumentIndex(root / "index.json")
        with patch.object(index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]):
            index.add_file(source)
        with patch.object(index.client, "embed", side_effect=AssertionError("route hint must not embed")):
            hints = index.retrieval_hint("Q-Former 如何连接冻结的视觉编码器和语言模型？")
        assert hints
        assert hints[0]["strong_match"] is True
        assert "q-former" in hints[0]["matched_entities"]
        assert len(hints) <= 3


def test_embedding_cache_batches_requests_and_reuses_results() -> None:
    with tempfile.TemporaryDirectory() as folder:
        index = DocumentIndex(Path(folder) / "index.json")
        index._embedding_space = "local-hash-v1:384"
        calls: list[int] = []

        def embed(texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            return [local_hash_embedding(text) for text in texts]

        with patch.dict(os.environ, {"RAG_EMBEDDING_BATCH_SIZE": "2"}), patch.object(
            index.client, "embed", side_effect=embed
        ):
            first = index._embed_cached([f"text-{number}" for number in range(5)])
            second = index._embed_cached([f"text-{number}" for number in range(5)])
        assert calls == [2, 2, 1]
        assert first == second


def test_rag_p0_dataset_is_extensible_and_local_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "eval" / "dataset" / "rag_p0_cases.jsonl"
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) == 10
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["query"] and case["relevant_sources"]
        for relative in case["corpus"]:
            assert (root / relative).is_file(), relative


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"RAG P0 tests passed: {len(tests)}")
