from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deskpilot.multimodal.asset_store import AssetStore
from deskpilot.multimodal.image_processor import ImageProcessor
from deskpilot.multimodal.ingestion import MultimodalIngestionService
from deskpilot.multimodal.providers.ocr import OCRRegion
from deskpilot.multimodal.retriever import MultimodalRetriever
from deskpilot.multimodal.vector_store import MultimodalVectorStore
from deskpilot.core.api_clients import local_hash_embedding


class FixtureOCR:
    provider = "fixture"

    def recognize(self, image_path: Path) -> list[OCRRegion]:
        with Image.open(image_path) as image:
            red, _green, blue = image.resize((1, 1)).getpixel((0, 0))
        text = "ATLAS-503 overload recovery" if red > blue else "NEBULA-204 cache policy"
        return [OCRRegion(text, 0.99, (18.0, 12.0, 162.0, 48.0))]


class FixtureVisionEmbedding:
    provider = "fixture"
    embedding_space = "vision/fixture-v1"
    model_id = "fixture"
    model_revision = "1"

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        values = []
        for path in paths:
            with Image.open(path) as image:
                red, _green, blue = image.resize((1, 1)).getpixel((0, 0))
            values.append([1.0, 0.0] if red > blue else [0.0, 1.0])
        return values

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "ATLAS" in text.upper() else [0.0, 1.0] for text in texts]


class FixtureTextEmbedder:
    semantic = False
    embedding_space = "text/local-hash-v1"
    model_id = "deskpilot/local-hash"
    model_revision = "1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [local_hash_embedding(text) for text in texts]


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (180, 120), color).save(path)
    return path


def test_asset_store_deduplicates_content_and_keeps_binary_out_of_catalog() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = _image(root / "first.png", (220, 30, 30))
        duplicate = root / "renamed.png"
        duplicate.write_bytes(source.read_bytes())
        store = AssetStore(root / "assets")

        first = store.ingest_file(source)
        second = store.ingest_file(duplicate)

        assert first.asset_id == second.asset_id
        assert Path(first.thumbnail_path).is_file()
        assert "base64" not in store.database.read_bytes().decode("latin-1", errors="ignore").casefold()


def test_image_processor_rejects_fake_and_oversized_images() -> None:
    with pytest.raises(ValueError, match="有效图片"):
        ImageProcessor().process_bytes(b"not-an-image")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = _image(Path(temp_dir) / "large.png", (10, 20, 30))
        with pytest.raises(ValueError, match="像素数"):
            ImageProcessor(max_pixels=100).process_file(path)


def test_vector_store_never_compares_different_spaces_or_dimensions() -> None:
    from deskpilot.multimodal.models import VectorRecord

    with tempfile.TemporaryDirectory() as temp_dir:
        store = MultimodalVectorStore(Path(temp_dir) / "vectors.sqlite3")
        store.upsert_vector(VectorRecord("one", "asset", "a", "image", "vision/a", "a", "1", 2, [1, 0]))
        store.upsert_vector(VectorRecord("two", "asset", "b", "image", "vision/b", "b", "1", 2, [1, 0]))
        store.upsert_vector(VectorRecord("three", "asset", "c", "image", "vision/a", "a", "1", 3, [1, 0, 0]))

        assert store.search([1, 0], "vision/a") == [("a", 1.0)]


def test_text_search_tokenizes_chinese_query_without_spaces() -> None:
    from deskpilot.multimodal.models import MultimodalChunk

    with tempfile.TemporaryDirectory() as temp_dir:
        store = MultimodalVectorStore(Path(temp_dir) / "vectors.sqlite3")
        store.upsert_chunk(MultimodalChunk(
            "target", "asset", "ocr", "活动截止日期为2026年9月30日，主办方电话400-636-6060。", "poster.jpg",
        ))
        store.upsert_chunk(MultimodalChunk(
            "noise", "asset2", "ocr", "Camouflaged object detection benchmark and segmentation.", "paper.pdf",
        ))

        results = store.text_search("活动的截止日期和主办方电话是什么？")

        assert results
        assert results[0][0] == "target"


def test_multimodal_ingestion_and_rrf_retrieve_ocr_and_visual_evidence() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        red = _image(root / "atlas.png", (230, 20, 20))
        blue = _image(root / "nebula.png", (20, 20, 230))
        assets = AssetStore(root / "assets")
        vectors = MultimodalVectorStore(root / "index.sqlite3")
        vision = FixtureVisionEmbedding()
        text = FixtureTextEmbedder()
        ingestion = MultimodalIngestionService(assets, vectors, FixtureOCR(), vision, text)
        ingestion.ingest_image(red)
        ingestion.ingest_image(blue)

        evidence = MultimodalRetriever(vectors, assets, vision, text).search("ATLAS-503 如何恢复", top_k=2)

        assert evidence
        assert evidence[0].source_label == "atlas.png"
        assert "ATLAS-503" in evidence[0].text
        assert evidence[0].metadata["channel_hits"] >= 2
        assert "ocr_lexical" in evidence[0].metadata["channel_names"]
        region = vectors.get_chunk(f"mm_{assets.ingest_file(red).sha256[:20]}_ocr_1")
        assert region is not None
        assert region.bbox == pytest.approx((0.1, 0.1, 0.9, 0.4))


def test_repeated_multimodal_ingestion_reuses_ocr_and_visual_vectors() -> None:
    class CountingOCR(FixtureOCR):
        calls = 0

        def recognize(self, image_path: Path) -> list[OCRRegion]:
            self.calls += 1
            return super().recognize(image_path)

    class CountingVision(FixtureVisionEmbedding):
        calls = 0

        def embed_images(self, paths: list[Path]) -> list[list[float]]:
            self.calls += 1
            return super().embed_images(paths)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = _image(root / "atlas.png", (230, 20, 20))
        ocr = CountingOCR()
        vision = CountingVision()
        ingestion = MultimodalIngestionService(
            AssetStore(root / "assets"), MultimodalVectorStore(root / "index.sqlite3"),
            ocr, vision, FixtureTextEmbedder(),
        )

        first = ingestion.ingest_image(image_path)
        second = ingestion.ingest_image(image_path)

        assert first["ocr_cached"] is False
        assert second["ocr_cached"] is True
        assert second["embedding_cached"] is True
        assert ocr.calls == 1
        assert vision.calls == 1


def test_low_memory_retrieval_skips_vision_model_when_ocr_matches(monkeypatch) -> None:
    class VisionMustNotLoad(FixtureVisionEmbedding):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("OCR 字面命中时不应加载视觉模型")

    monkeypatch.setenv("MULTIMODAL_LOW_MEMORY_MODE", "true")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = _image(root / "atlas.png", (230, 20, 20))
        assets = AssetStore(root / "assets")
        vectors = MultimodalVectorStore(root / "index.sqlite3")
        text = FixtureTextEmbedder()
        ingestion = MultimodalIngestionService(
            assets, vectors, FixtureOCR(), FixtureVisionEmbedding(), text
        )
        ingestion.ingest_image(image_path)

        evidence = MultimodalRetriever(
            vectors, assets, VisionMustNotLoad(), text
        ).search("ATLAS-503", top_k=2)

        assert evidence
        assert "ocr_lexical" in evidence[0].metadata["channel_names"]


def test_ingestion_keeps_ocr_index_when_optional_vision_embedding_fails() -> None:
    class FailingVision(FixtureVisionEmbedding):
        def embed_images(self, paths: list[Path]) -> list[list[float]]:
            raise RuntimeError("os error 1455")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = _image(root / "atlas.png", (230, 20, 20))
        assets = AssetStore(root / "assets")
        vectors = MultimodalVectorStore(root / "index.sqlite3")

        result = MultimodalIngestionService(
            assets, vectors, FixtureOCR(), FailingVision(), FixtureTextEmbedder()
        ).ingest_image(image_path)

        assert result["ocr_regions"] == 1
        assert result["embedding_status"] == "failed"
        assert "页面文件不足" in result["warning"]
        assert vectors.text_search("ATLAS-503")
        assert vectors.stats() == {"assets": 1, "chunks": 2, "vectors": 2}


def test_multimodal_dataset_has_unique_ids_and_required_subsets() -> None:
    path = Path(__file__).resolve().parents[1] / "eval" / "dataset" / "multimodal_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    assert len(rows) == len({row["id"] for row in rows})
    assert {row["subset"] for row in rows} >= {"Multimodal-QA", "Multimodal-RAG", "Multimodal-Safety"}


def test_siglip_normalization_accepts_transformers_v5_model_output() -> None:
    torch = pytest.importorskip("torch")
    from deskpilot.multimodal.providers.vision_embedding import SigLIP2EmbeddingProvider

    class Output:
        pooler_output = torch.tensor([[3.0, 4.0]])

    assert SigLIP2EmbeddingProvider._normalize(Output())[0] == pytest.approx([0.6, 0.8])


def test_siglip_isolated_worker_returns_vectors_without_loading_parent_model(monkeypatch) -> None:
    from deskpilot.multimodal.providers.vision_embedding import SigLIP2EmbeddingProvider

    provider = SigLIP2EmbeddingProvider("fixture/model")
    monkeypatch.setattr(
        SigLIP2EmbeddingProvider, "_available_commit_bytes", staticmethod(lambda: None)
    )

    def fake_run(command, **kwargs):
        request = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
        assert request == {"mode": "texts", "values": ["测试文本"]}
        Path(command[-1]).write_text(
            json.dumps({"vectors": [[0.6, 0.8]]}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    vectors = provider._embed_isolated("texts", ["测试文本"])

    assert vectors == [[0.6, 0.8]]
    assert provider._model is None


def test_siglip_preflight_rejects_insufficient_windows_commit(monkeypatch) -> None:
    from deskpilot.multimodal.providers.vision_embedding import SigLIP2EmbeddingProvider

    monkeypatch.setenv("IMAGE_EMBEDDING_MIN_AVAILABLE_COMMIT_GB", "5")
    monkeypatch.setattr(
        SigLIP2EmbeddingProvider, "_available_commit_bytes", staticmethod(lambda: 4 * 1024 ** 3)
    )

    with pytest.raises(RuntimeError, match="可用提交内存不足"):
        SigLIP2EmbeddingProvider._ensure_resource_budget()


def test_paddleocr_v3_result_is_normalized_to_regions() -> None:
    from deskpilot.multimodal.providers.ocr import PaddleOCRProvider

    rows = PaddleOCRProvider._parse_v3_results([{
        "rec_texts": ["配料表", "净含量 100g"],
        "rec_scores": [0.99, 0.91],
        "rec_boxes": [[10, 20, 110, 50], [12, 60, 180, 90]],
    }])

    assert [row.text for row in rows] == ["配料表", "净含量 100g"]
    assert rows[0].confidence == pytest.approx(0.99)
    assert rows[0].bbox == (10.0, 20.0, 110.0, 50.0)


def test_paddleocr_native_crash_retries_once_with_conservative_profile(monkeypatch) -> None:
    from deskpilot.multimodal.providers.ocr import PaddleOCRProvider

    provider = PaddleOCRProvider()
    calls: list[dict[str, str]] = []

    def fake_worker(image_path, output_path, environment, timeout):
        calls.append(dict(environment))
        if len(calls) == 1:
            return subprocess.CompletedProcess([], 3221225477, "", "native crash")
        output_path.write_text(
            json.dumps([{"text": "活动日期", "confidence": 0.98, "bbox": [1, 2, 3, 4]}]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(provider, "_run_worker", fake_worker)
    rows = provider._recognize_isolated(Path("fixture.jpg"))

    assert [row.text for row in rows] == ["活动日期"]
    assert len(calls) == 2
    assert calls[1]["OCR_CPU_THREADS"] == "1"
    assert calls[1]["OCR_USE_TEXTLINE_ORIENTATION"] == "false"


def test_paddleocr_normal_worker_failure_is_not_retried(monkeypatch) -> None:
    from deskpilot.multimodal.providers.ocr import PaddleOCRProvider

    provider = PaddleOCRProvider()
    calls = 0

    def fake_worker(image_path, output_path, environment, timeout):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 2, "", "configuration failed")

    monkeypatch.setattr(provider, "_run_worker", fake_worker)
    with pytest.raises(RuntimeError, match="configuration failed"):
        provider._recognize_isolated(Path("fixture.jpg"))

    assert calls == 1


def test_multimodal_answer_prefers_lexical_evidence_over_multi_channel_false_positive() -> None:
    from deskpilot.multimodal.models import VisualEvidence
    from deskpilot.multimodal.service import MultimodalService

    strong = VisualEvidence(
        "one", "asset_one", "配料表.jpg", "ocr", 0.04,
        metadata={
            "channel_hits": 3,
            "channel_names": ["ocr_lexical", "ocr_dense", "vision_text"],
        },
    )
    weak = VisualEvidence(
        "two", "asset_two", "unrelated.pdf#page-1", "ocr", 0.02,
        metadata={
            "channel_hits": 2,
            "channel_names": ["ocr_dense", "vision_text"],
        },
    )

    assert MultimodalService._filter_answer_evidence([strong, weak]) == [strong]
    assert MultimodalService._filter_answer_evidence([weak]) == [weak]


def test_multimodal_answer_keeps_visual_fallback_without_lexical_hit() -> None:
    from deskpilot.multimodal.models import VisualEvidence
    from deskpilot.multimodal.service import MultimodalService

    visual = VisualEvidence(
        "one", "asset_one", "dog.png", "image", 0.04,
        metadata={"channel_hits": 1, "channel_names": ["vision_text"]},
    )

    assert MultimodalService._filter_answer_evidence([visual]) == [visual]


def test_multimodal_presenters_do_not_render_empty_numbered_items() -> None:
    from deskpilot.core.agent import DocumentQAAgent

    indexed = DocumentQAAgent._present_multimodal_index_result([{
        "asset": {"metadata": {"source_name": "配料表.jpg"}},
        "ocr_regions": 64,
        "embedding_status": "success",
        "ocr_cached": True,
    }])
    searched = DocumentQAAgent._present_multimodal_search_result([{
        "source_label": "配料表.jpg#region-1",
        "text": "活动截止日期为2026年9月30日",
    }])

    assert "配料表.jpg" in indexed
    assert "64" in indexed
    assert "复用已有索引" in indexed
    assert "配料表.jpg#region-1" in searched
    assert "2026年9月30日" in searched
    assert indexed.strip() != "1."


def test_clipboard_image_requires_in_chat_cloud_confirmation(monkeypatch, tmp_path: Path) -> None:
    from io import BytesIO

    from deskpilot.core.agent import DocumentQAAgent
    from deskpilot.multimodal.service import MultimodalService
    from deskpilot.rag.vector_index import DocumentIndex
    from deskpilot.tools.tool_registry import build_default_tool_registry

    buffer = BytesIO()
    Image.new("RGB", (24, 24), (20, 40, 60)).save(buffer, format="PNG")
    monkeypatch.setenv("SENSITIVE_IMAGE_POLICY", "confirm")
    service = MultimodalService(tmp_path / "multimodal")
    service.vision_llm.allow_cloud = True
    service.vision_llm.api_key = "fixture-key"
    asset = service.add_chat_attachment_bytes(buffer.getvalue(), "clipboard.png")
    assert asset.sensitivity == "sensitive"

    index = DocumentIndex(tmp_path / "index" / "index.json")
    agent = DocumentQAAgent(index)
    agent.multimodal = service
    agent.tool_registry = build_default_tool_registry(
        index, agent.web_research_agent, workspace_root=tmp_path, multimodal_service=service
    )
    result = agent.answer_multimodal("图片里有什么？", [asset.asset_id])

    assert result.pending_action is not None
    assert result.pending_action["tool_name"] == "vision.answer_attachments"
    assert any(step.status == "waiting_human" for step in result.steps)
    assert "人工确认" in result.answer


def test_sensitive_policy_block_prevents_cloud_upload(monkeypatch, tmp_path: Path) -> None:
    from io import BytesIO

    from deskpilot.multimodal.service import MultimodalService

    buffer = BytesIO()
    Image.new("RGB", (24, 24), (20, 40, 60)).save(buffer, format="PNG")
    monkeypatch.setenv("SENSITIVE_IMAGE_POLICY", "block")
    service = MultimodalService(tmp_path / "multimodal")
    service.vision_llm.allow_cloud = True
    service.vision_llm.api_key = "fixture-key"
    asset = service.add_chat_attachment_bytes(buffer.getvalue())

    with pytest.raises(PermissionError, match="只允许本地 OCR"):
        service.answer("描述图片", [asset.asset_id], confirm_sensitive=True)


def test_vision_llm_streams_tokens(monkeypatch, tmp_path: Path) -> None:
    import urllib.request

    from deskpilot.multimodal.providers.vision_llm import VisionLLMProvider

    image_path = _image(tmp_path / "stream.png", (30, 40, 50))

    class StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello "}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"vision"}}]}\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n'
            yield b"data: [DONE]\n"

    provider = VisionLLMProvider()
    provider.allow_cloud = True
    provider.api_key = "fixture-key"
    events: list[dict] = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: StreamResponse())

    answer = provider.answer("描述", [image_path], event_callback=events.append)

    assert answer == "hello vision"
    assert "".join(item["text"] for item in events if item["type"] == "token") == answer
    assert provider.last_usage == {
        "prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14, "reported_calls": 1,
    }


def test_pdf_ingestion_batches_ocr_and_vision_once(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    class BatchOCR(FixtureOCR):
        calls = 0

        def recognize_many(self, image_paths: list[Path]) -> list[list[OCRRegion]]:
            self.calls += 1
            return [[OCRRegion(f"page-{index}", 1.0)] for index, _path in enumerate(image_paths, 1)]

    class BatchVision(FixtureVisionEmbedding):
        calls = 0

        def embed_images(self, paths: list[Path]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0] for _path in paths]

    pdf_path = tmp_path / "batch.pdf"
    document = fitz.open()
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"fixture page {page_number}")
    document.save(pdf_path)
    document.close()
    ocr = BatchOCR()
    vision = BatchVision()
    ingestion = MultimodalIngestionService(
        AssetStore(tmp_path / "assets"), MultimodalVectorStore(tmp_path / "vectors.sqlite3"),
        ocr, vision, FixtureTextEmbedder(),
    )

    results = ingestion.ingest_pdf(pdf_path)

    assert len(results) == 3
    assert ocr.calls == 1
    assert vision.calls == 1
    assert all(item["embedding_status"] == "success" for item in results)
    assert [item["asset"]["metadata"]["source_name"] for item in results] == [
        "batch.pdf#page-1", "batch.pdf#page-2", "batch.pdf#page-3",
    ]


def test_pdf_ingestion_rejects_incomplete_batch_results(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    class IncompleteOCR(FixtureOCR):
        def recognize_many(self, image_paths: list[Path]) -> list[list[OCRRegion]]:
            return [[] for _path in image_paths[:-1]]

    pdf_path = tmp_path / "incomplete.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()
    ingestion = MultimodalIngestionService(
        AssetStore(tmp_path / "assets"), MultimodalVectorStore(tmp_path / "vectors.sqlite3"),
        IncompleteOCR(), FixtureVisionEmbedding(), FixtureTextEmbedder(),
    )

    with pytest.raises(RuntimeError, match="OCR 批处理返回数量不一致"):
        ingestion.ingest_pdf(pdf_path)


def test_multimodal_sqlite_stores_use_wal_and_busy_timeout(tmp_path: Path) -> None:
    assets = AssetStore(tmp_path / "assets")
    vectors = MultimodalVectorStore(tmp_path / "vectors.sqlite3")
    with assets._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    with vectors._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
