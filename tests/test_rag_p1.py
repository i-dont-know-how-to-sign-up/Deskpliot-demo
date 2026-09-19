from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.core.models import Chunk, Document
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.retrieval import RetrievalFilters, RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


def build_index(root: Path) -> DocumentIndex:
    index = DocumentIndex(root / "index.json")
    index.client.embed = lambda texts: [local_hash_embedding(text) for text in texts]  # type: ignore[method-assign]
    index.documents = {
        "atlas": Document("atlas", str(root / "atlas.md"), "atlas.md", "md", "", {}),
        "guide": Document("guide", str(root / "guide.txt"), "guide.txt", "txt", "", {}),
    }
    texts = {
        "a1": ("atlas", "ATLAS-503 表示分片服务过载，客户端应指数退避并最多重试三次。", 3),
        "a2": ("atlas", "Atlas 上传成功后使用 SHA-256 校验数据完整性。", 4),
        "g1": ("guide", "语义检索适合同义表达，BM25 适合错误码、版本号和精确产品型号。", 1),
        "g2": ("guide", "RRF 会融合稠密检索和稀疏检索的独立排名。", 2),
    }
    index.chunks = {
        chunk_id: Chunk(
            chunk_id, doc_id, text, f"{index.documents[doc_id].title}#page-{page}", page,
            metadata={"page_number": page}, embedding=local_hash_embedding(text),
        )
        for chunk_id, (doc_id, text, page) in texts.items()
    }
    index.catalog.sync_search_index(index.documents, index.chunks)
    return index


def test_bm25_recalls_exact_error_code() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_bm25_") as raw:
        index = build_index(Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "ATLAS-503", ("ATLAS-503",), top_k=2, use_dense=False, use_sparse=True,
        ))
        assert result and result[0].chunk_id == "a1"
        candidate = index.last_trace.to_dict()["candidates"][0]
        assert "bm25" in candidate["retriever_hits"]


def test_metadata_filter_is_applied_before_retrieval() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_filter_") as raw:
        index = build_index(Path(raw))
        result = index.search_hybrid(RetrievalRequest(
            "错误码和精确型号", ("错误码和精确型号",),
            filters=RetrievalFilters(doc_ids=("guide",), page_min=1, page_max=1), top_k=5,
        ))
        assert result and {item.doc_id for item in result} == {"guide"}
        assert {item.chunk_id for item in result} == {"g1"}
        first_event = index.last_trace.to_dict()["events"][0]
        assert first_event["stage"] == "metadata_filter"
        assert first_event["output_count"] == 1


def test_rrf_keeps_all_query_and_retriever_hits() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_rrf_") as raw:
        index = build_index(Path(raw))
        index.search_hybrid(RetrievalRequest(
            "ATLAS-503", ("ATLAS-503", "分片服务过载恢复"), top_k=3,
        ))
        candidate = next(item for item in index.last_trace.to_dict()["candidates"] if item["chunk_id"] == "a1")
        assert len(candidate["query_hits"]) >= 2
        assert set(candidate["retriever_hits"]) == {"dense", "bm25"}
        assert len(candidate["ranks"]) >= 3


def test_simple_query_does_not_call_analysis_llm() -> None:
    calls: list[str] = []
    analyzer = QueryAnalyzer(lambda prompt: calls.append(prompt) or "{}")
    result = analyzer.analyze("ATLAS-503 是什么？", complex_task=False)
    assert result.method == "deterministic"
    assert not calls


def test_complex_query_uses_bounded_multi_query_and_cache() -> None:
    calls: list[str] = []
    response = (
        '{"standalone_question":"比较 BLIP 与 BLIP-2",'
        '"entities":["BLIP","BLIP-2"],"must_terms":["architecture"],'
        '"task_type":"comparison","needs_multi_query":true,"needs_hyde":false,'
        '"sub_questions":["BLIP architecture","BLIP-2 architecture","training differences","ignored"],'
        '"context_expansion":"parent"}'
    )
    analyzer = QueryAnalyzer(lambda prompt: calls.append(prompt) or response)
    first = analyzer.analyze("比较两篇论文", index_version="2:4", complex_task=True)
    second = analyzer.analyze("比较两篇论文", index_version="2:4", complex_task=True)
    assert first.method == "llm" and len(first.sub_questions) == 3
    assert second.cache_hit and len(calls) == 1


def test_hyde_is_dense_only_and_never_becomes_evidence() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_hyde_") as raw:
        index = build_index(Path(raw))
        pseudo = "这是一个只用于检索的假设段落 UNIQUE_HYDE_TEXT"
        result = index.search_hybrid(RetrievalRequest(
            "如何恢复分片服务", ("如何恢复分片服务",), hyde_text=pseudo, top_k=3,
        ))
        trace = index.last_trace.to_dict()
        assert any("hyde:dense" in item["scores"] for item in trace["candidates"])
        assert all(pseudo not in item.text for item in result)
        assert all("hyde:bm25" not in item["scores"] for item in trace["candidates"])


def test_feature_flag_can_restore_p0_retrieval() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_flag_") as raw:
        index = build_index(Path(raw))
        old = os.environ.get("RAG_HYBRID_ENABLED")
        os.environ["RAG_HYBRID_ENABLED"] = "false"
        try:
            index.search("ATLAS-503", top_k=2)
            assert index.last_trace and index.last_trace.mode == "search"
        finally:
            if old is None:
                os.environ.pop("RAG_HYBRID_ENABLED", None)
            else:
                os.environ["RAG_HYBRID_ENABLED"] = old


def test_body_coverage_reranks_architecture_over_tables_and_conclusion() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_p1_body_rerank_") as raw:
        index = DocumentIndex(Path(raw) / "index.json")
        index.client.embed = lambda texts: [local_hash_embedding(text) for text in texts]  # type: ignore[method-assign]
        document = Document(
            "ram", str(Path(raw) / "recognize_anything.pdf"), "recognize_anything.pdf", "pdf", "", {},
        )
        index.documents = {"ram": document}
        values = {
            "table": "Recognition benchmark table COCO OpenImages common rare categories and scores.",
            "conclusion": "Recognize Anything is a strong foundation model for image tagging.",
            "architecture": (
                "Recognize Anything Model architecture consists of an image encoder, "
                "an image-tag recognition decoder and a text generation encoder-decoder."
            ),
        }
        index.chunks = {
            chunk_id: Chunk(
                chunk_id, "ram", text, f"recognize_anything.pdf#page-{position}", position,
                metadata={"page_number": position}, embedding=local_hash_embedding(text),
            )
            for position, (chunk_id, text) in enumerate(values.items(), start=1)
        }
        index.catalog.sync_search_index(index.documents, index.chunks)
        query = "recognize anything 的工作流程 workflow architecture process pipeline"
        results = index.search_hybrid(RetrievalRequest(
            query,
            (query,),
            filters=RetrievalFilters(doc_ids=("ram",)),
            top_k=3,
            use_dense=False,
        ))

        assert results and results[0].chunk_id == "architecture"
        candidate = next(
            item for item in index.last_trace.to_dict()["candidates"]
            if item["chunk_id"] == "architecture"
        )
        assert candidate["rerank_score"] > candidate["rrf_score"]


def main() -> None:
    test_bm25_recalls_exact_error_code()
    test_metadata_filter_is_applied_before_retrieval()
    test_rrf_keeps_all_query_and_retriever_hits()
    test_simple_query_does_not_call_analysis_llm()
    test_complex_query_uses_bounded_multi_query_and_cache()
    test_hyde_is_dense_only_and_never_becomes_evidence()
    test_feature_flag_can_restore_p0_retrieval()
    test_body_coverage_reranks_architecture_over_tables_and_conclusion()
    print("RAG P1 tests passed: 8")


if __name__ == "__main__":
    main()
