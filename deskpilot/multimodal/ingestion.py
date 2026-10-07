from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from ..core.api_clients import local_hash_embedding
from .asset_store import AssetStore
from .models import MediaAsset, MultimodalChunk, VectorRecord
from .providers.ocr import OCRProvider, build_ocr_provider
from .providers.vision_embedding import VisionEmbeddingProvider, build_vision_embedding_provider
from .vector_store import MultimodalVectorStore


class MultimodalIngestionService:
    def __init__(
        self, asset_store: AssetStore, vector_store: MultimodalVectorStore,
        ocr: OCRProvider | None = None, vision: VisionEmbeddingProvider | None = None,
    ) -> None:
        self.assets = asset_store
        self.vectors = vector_store
        self.ocr = ocr or build_ocr_provider()
        self.vision = vision or build_vision_embedding_provider()

    def ingest_image(
        self, path: Path, *, source_kind: str = "knowledge", parent_doc_id: str | None = None,
        page_number: int | None = None, source_label: str | None = None,
    ) -> dict:
        low_memory = os.getenv("MULTIMODAL_LOW_MEMORY_MODE", "true").casefold() in {
            "1", "true", "yes", "on",
        }
        if low_memory and hasattr(self.vision, "release"):
            self.vision.release()
        asset = self.assets.ingest_file(
            path, source_kind=source_kind, parent_doc_id=parent_doc_id, page_number=page_number,
            metadata={"source_name": path.name},
        )
        label = source_label or path.name
        chunk_id = f"mm_{asset.sha256[:20]}_ocr"
        existing_chunk = self.vectors.get_chunk(chunk_id)
        vision_cached = self.vectors.has_vector(chunk_id, self.vision.embedding_space)
        if existing_chunk is not None and vision_cached:
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
        regions = self.ocr.recognize(Path(asset.original_path))
        ocr_text = "\n".join(region.text for region in regions)
        chunk = MultimodalChunk(
            chunk_id=chunk_id, asset_id=asset.asset_id,
            doc_id=parent_doc_id, modality="ocr", text=ocr_text, source_label=label,
            page_number=page_number,
            metadata={"ocr_provider": self.ocr.provider, "region_count": len(regions)},
        )
        self.vectors.upsert_chunk(chunk)
        if ocr_text:
            text_vector = local_hash_embedding(ocr_text)
            self.vectors.upsert_vector(VectorRecord(
                vector_id=f"vec_{chunk.chunk_id}_text", owner_type="media_chunk", owner_id=chunk.chunk_id,
                modality="ocr", embedding_space="text/local-hash-v1", model_id="deskpilot/local-hash",
                model_revision="1", dimension=len(text_vector), vector=text_vector,
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
            region_vector = local_hash_embedding(region.text)
            self.vectors.upsert_vector(VectorRecord(
                vector_id=f"vec_{region_chunk.chunk_id}_text", owner_type="media_region",
                owner_id=region_chunk.chunk_id, modality="image_region",
                embedding_space="text/local-hash-v1", model_id="deskpilot/local-hash",
                model_revision="1", dimension=len(region_vector), vector=region_vector,
                metadata={"bbox": bbox, "ocr_confidence": region.confidence},
            ))
        if low_memory and hasattr(self.ocr, "release"):
            self.ocr.release()
        embedding_status = "success"
        embedding_warning = ""
        try:
            vision_vector = self.vision.embed_images([Path(asset.original_path)])[0]
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

    def ingest_pdf(self, path: Path) -> list[dict]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF 页图索引需要 PyMuPDF") from exc
        path = path.expanduser().resolve()
        max_pages = max(1, min(int(os.getenv("MULTIMODAL_PDF_MAX_PAGES", "80")), 500))
        doc_id = f"pdf_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
        results: list[dict] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                if page_index >= max_pages:
                    break
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                    temporary.write(pixmap.tobytes("png"))
                    temp_path = Path(temporary.name)
                try:
                    results.append(self.ingest_image(
                        temp_path, source_kind="pdf_page", parent_doc_id=doc_id,
                        page_number=page_index + 1, source_label=f"{path.name}#page-{page_index + 1}",
                    ))
                finally:
                    temp_path.unlink(missing_ok=True)
        return results

    def ingest_file(self, path: Path) -> list[dict]:
        if path.suffix.casefold() == ".pdf":
            return self.ingest_pdf(path)
        return [self.ingest_image(path)]
