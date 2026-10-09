from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .asset_store import AssetStore
from .models import MediaAsset, MultimodalChunk, VectorRecord
from .providers.ocr import OCRProvider, build_ocr_provider
from .providers.vision_embedding import VisionEmbeddingProvider, build_vision_embedding_provider
from .vector_store import MultimodalVectorStore
from .text_embedding import MultimodalTextEmbedder


class MultimodalIngestionService:
    def __init__(
        self, asset_store: AssetStore, vector_store: MultimodalVectorStore,
        ocr: OCRProvider | None = None, vision: VisionEmbeddingProvider | None = None,
        text_embedder: MultimodalTextEmbedder | None = None,
    ) -> None:
        self.assets = asset_store
        self.vectors = vector_store
        self.ocr = ocr or build_ocr_provider()
        self.vision = vision or build_vision_embedding_provider()
        self.text_embedder = text_embedder or MultimodalTextEmbedder()
        self._vision_available = True
        self._vision_error = ""

    @property
    def vision_available(self) -> bool:
        """返回当前进程内视觉编码器是否仍可用。"""
        return self._vision_available

    @property
    def vision_error(self) -> str:
        """返回视觉编码器首次失败原因，避免调用方访问内部状态。"""
        return self._vision_error

    def ingest_image(
        self, path: Path, *, source_kind: str = "knowledge", parent_doc_id: str | None = None,
        page_number: int | None = None, source_label: str | None = None,
        _asset: MediaAsset | None = None, _regions: list | None = None,
        _vision_vector: list[float] | None = None, _vision_error: str = "",
        _providers_precomputed: bool = False,
    ) -> dict:
        low_memory = os.getenv("MULTIMODAL_LOW_MEMORY_MODE", "true").casefold() in {
            "1", "true", "yes", "on",
        }
        if low_memory and hasattr(self.vision, "release"):
            self.vision.release()
        asset = _asset or self.assets.ingest_file(
            path, source_kind=source_kind, parent_doc_id=parent_doc_id, page_number=page_number,
            metadata={"source_name": path.name},
        )
        label = source_label or path.name
        chunk_id = f"mm_{asset.sha256[:20]}_ocr"
        existing_chunk = self.vectors.get_chunk(chunk_id)
        vision_cached = self.vectors.has_vector(chunk_id, self.vision.embedding_space)
        text_cached = self.vectors.has_vector(chunk_id, self.text_embedder.embedding_space)
        if existing_chunk is not None and not text_cached:
            self._backfill_text_vectors(asset.asset_id)
            text_cached = True
        if existing_chunk is not None and vision_cached and text_cached:
            return {
                "asset": asset.to_dict(), "chunk_id": chunk_id,
                "ocr_text": existing_chunk.text,
                "ocr_regions": int(existing_chunk.metadata.get("region_count", 0) or 0),
                "embedding_space": self.vision.embedding_space,
                "embedding_status": "success", "warning": "",
                "ocr_cached": True, "embedding_cached": True,
            }
        if (
            existing_chunk is not None
            and not vision_cached
            and os.getenv("MULTIMODAL_RETRY_MISSING_VISION", "false").casefold()
            not in {"1", "true", "yes", "on"}
        ):
            return {
                "asset": asset.to_dict(), "chunk_id": chunk_id,
                "ocr_text": existing_chunk.text,
                "ocr_regions": int(existing_chunk.metadata.get("region_count", 0) or 0),
                "embedding_space": self.vision.embedding_space,
                "embedding_status": "failed",
                "warning": (
                    "已复用 OCR 索引；视觉向量此前未完成。增大页面文件后设置 "
                    "MULTIMODAL_RETRY_MISSING_VISION=true 并重新索引可补齐。"
                ),
                "ocr_cached": True, "embedding_cached": False,
            }
        regions = list(_regions) if _regions is not None else self.ocr.recognize(Path(asset.original_path))
        ocr_text = "\n".join(region.text for region in regions)
        chunk = MultimodalChunk(
            chunk_id=chunk_id, asset_id=asset.asset_id,
            doc_id=parent_doc_id, modality="ocr", text=ocr_text, source_label=label,
            page_number=page_number,
            metadata={"ocr_provider": self.ocr.provider, "region_count": len(regions)},
        )
        self.vectors.upsert_chunk(chunk)
        text_records: list[tuple[str, str, str, str, dict]] = []
        text_space_id = hashlib.sha1(self.text_embedder.embedding_space.encode()).hexdigest()[:8]
        if ocr_text:
            text_records.append((
                f"vec_{chunk.chunk_id}_text_{text_space_id}", "media_chunk", chunk.chunk_id, ocr_text, {},
            ))
        for index, region in enumerate(regions, start=1):
            bbox = None
            if region.bbox:
                left, top, right, bottom = region.bbox
                bbox = (
                    max(0.0, min(1.0, left / asset.width)),
                    max(0.0, min(1.0, top / asset.height)),
                    max(0.0, min(1.0, right / asset.width)),
                    max(0.0, min(1.0, bottom / asset.height)),
                )
            region_chunk = MultimodalChunk(
                chunk_id=f"mm_{asset.sha256[:20]}_ocr_{index}", asset_id=asset.asset_id,
                doc_id=parent_doc_id, modality="image_region", text=region.text,
                source_label=f"{label}#region-{index}", page_number=page_number, bbox=bbox,
                metadata={"ocr_provider": self.ocr.provider, "ocr_confidence": region.confidence},
            )
            self.vectors.upsert_chunk(region_chunk)
            text_records.append((
                f"vec_{region_chunk.chunk_id}_text_{text_space_id}", "media_region", region_chunk.chunk_id,
                region.text, {"bbox": bbox, "ocr_confidence": region.confidence},
            ))
        if text_records:
            vectors = self._embed_texts([item[3] for item in text_records])
            for item, text_vector in zip(text_records, vectors):
                vector_id, owner_type, owner_id, _text, metadata = item
                self.vectors.upsert_vector(VectorRecord(
                    vector_id=vector_id, owner_type=owner_type, owner_id=owner_id,
                    modality="ocr" if owner_type == "media_chunk" else "image_region",
                    embedding_space=self.text_embedder.embedding_space,
                    model_id=self.text_embedder.model_id,
                    model_revision=self.text_embedder.model_revision,
                    dimension=len(text_vector), vector=text_vector, metadata=metadata,
                ))
        if low_memory and hasattr(self.ocr, "release"):
            self.ocr.release()
        embedding_status = "success"
        embedding_warning = ""
        try:
            if _vision_error:
                raise RuntimeError(_vision_error)
            if _providers_precomputed and _vision_vector is None:
                raise RuntimeError("批处理视觉向量未生成")
            if not _providers_precomputed and not self._vision_available:
                raise RuntimeError(self._vision_error or "视觉编码器在本次运行中不可用")
            vision_vector = _vision_vector or self.vision.embed_images([Path(asset.original_path)])[0]
            self.vectors.upsert_vector(VectorRecord(
                vector_id=f"vec_{asset.asset_id}_{hashlib.sha1(self.vision.embedding_space.encode()).hexdigest()[:8]}",
                owner_type="media_asset", owner_id=chunk.chunk_id, modality="image",
                embedding_space=self.vision.embedding_space, model_id=self.vision.model_id,
                model_revision=self.vision.model_revision, dimension=len(vision_vector), vector=vision_vector,
                metadata={"asset_id": asset.asset_id},
            ))
        except (RuntimeError, OSError, MemoryError) as exc:
            # OCR 与视觉向量是独立检索通道。可选视觉模型资源不足时保留已完成的
            # OCR 索引，避免把可用资产错误报告成完全失败。
            embedding_status = "failed"
            detail = str(exc)
            self._vision_available = False
            self._vision_error = detail
            if "os error 1455" in detail:
                detail = "Windows 页面文件不足，SigLIP2 暂时无法加载"
            embedding_warning = (
                f"OCR 索引已完成，但视觉向量生成失败：{detail}。"
                "增大 Windows 页面文件后重新索引可补齐视觉向量。"
            )
        return {
            "asset": asset.to_dict(), "chunk_id": chunk.chunk_id, "ocr_text": ocr_text,
            "ocr_regions": len(regions), "embedding_space": self.vision.embedding_space,
            "embedding_status": embedding_status, "warning": embedding_warning,
            "ocr_cached": False, "embedding_cached": False,
        }

    def _backfill_text_vectors(self, asset_id: str) -> None:
        chunks = [item for item in self.vectors.list_chunks_for_asset(asset_id) if item.text.strip()]
        if not chunks:
            return
        vectors = self._embed_texts([item.text for item in chunks])
        for chunk, vector in zip(chunks, vectors):
            self.vectors.upsert_vector(VectorRecord(
                vector_id=f"vec_{chunk.chunk_id}_text_{hashlib.sha1(self.text_embedder.embedding_space.encode()).hexdigest()[:8]}",
                owner_type="media_chunk" if chunk.modality == "ocr" else "media_region",
                owner_id=chunk.chunk_id, modality=chunk.modality,
                embedding_space=self.text_embedder.embedding_space,
                model_id=self.text_embedder.model_id,
                model_revision=self.text_embedder.model_revision,
                dimension=len(vector), vector=vector,
                metadata={"bbox": chunk.bbox} if chunk.bbox else {},
            ))

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, min(int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16")), 128))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.text_embedder.embed(texts[start:start + batch_size]))
        if len(vectors) != len(texts):
            raise RuntimeError("OCR 文本向量数量与输入不一致")
        return vectors

    def ingest_pdf(self, path: Path) -> list[dict]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF 页图索引需要 PyMuPDF") from exc
        path = path.expanduser().resolve()
        max_pages = max(1, min(int(os.getenv("MULTIMODAL_PDF_MAX_PAGES", "80")), 500))
        doc_id = f"pdf_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
        rendered: list[tuple[Path, MediaAsset, int, str]] = []
        try:
            # 渲染也必须处于清理范围内，否则中途解析失败会遗留临时页图。
            with fitz.open(path) as document:
                for page_index, page in enumerate(document):
                    if page_index >= max_pages:
                        break
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                        temporary.write(pixmap.tobytes("png"))
                        temp_path = Path(temporary.name)
                    page_number = page_index + 1
                    label = f"{path.name}#page-{page_number}"
                    asset = self.assets.ingest_file(
                        temp_path, source_kind="pdf_page", parent_doc_id=doc_id,
                        page_number=page_number, metadata={"source_name": label},
                    )
                    rendered.append((temp_path, asset, page_number, label))

            # 一次 worker 处理整批页面，避免每页重复加载 Paddle/Torch。
            provider_paths = [Path(asset.original_path) for _, asset, _, _ in rendered]
            recognize_many = getattr(self.ocr, "recognize_many", None)
            region_groups = recognize_many(provider_paths) if callable(recognize_many) else [
                self.ocr.recognize(item) for item in provider_paths
            ]
            if len(region_groups) != len(provider_paths):
                raise RuntimeError(
                    f"OCR 批处理返回数量不一致：期望 {len(provider_paths)}，实际 {len(region_groups)}"
                )
            if hasattr(self.ocr, "release"):
                self.ocr.release()
            vision_vectors: list[list[float] | None] = [None] * len(provider_paths)
            vision_error = ""
            try:
                if not self._vision_available:
                    raise RuntimeError(self._vision_error or "视觉编码器在本次运行中不可用")
                generated = self.vision.embed_images(provider_paths) if provider_paths else []
                vision_vectors = list(generated)
                if len(vision_vectors) != len(provider_paths):
                    raise RuntimeError(
                        f"视觉向量批处理返回数量不一致：期望 {len(provider_paths)}，实际 {len(vision_vectors)}"
                    )
            except (RuntimeError, OSError, MemoryError) as exc:
                vision_error = str(exc)
                self._vision_available = False
                self._vision_error = vision_error
            results: list[dict] = []
            for item, regions, vector in zip(rendered, region_groups, vision_vectors):
                temp_path, asset, page_number, label = item
                results.append(self.ingest_image(
                    temp_path, source_kind="pdf_page", parent_doc_id=doc_id,
                    page_number=page_number, source_label=label, _asset=asset,
                    _regions=regions, _vision_vector=vector, _vision_error=vision_error,
                    _providers_precomputed=True,
                ))
            return results
        finally:
            for temp_path, _asset, _page, _label in rendered:
                temp_path.unlink(missing_ok=True)

    def ingest_file(self, path: Path) -> list[dict]:
        if path.suffix.casefold() == ".pdf":
            return self.ingest_pdf(path)
        return [self.ingest_image(path)]
