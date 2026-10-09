from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from .asset_store import AssetStore
from .models import VisualEvidence
from .providers.vision_embedding import VisionEmbeddingProvider
from .vector_store import MultimodalVectorStore
from .text_embedding import MultimodalTextEmbedder


class MultimodalRetriever:
    """OCR、跨模态和图像相似度多路召回，使用 RRF 融合不可比的原始分数。"""

    def __init__(
        self, store: MultimodalVectorStore, assets: AssetStore,
        vision: VisionEmbeddingProvider, text_embedder: MultimodalTextEmbedder | None = None,
    ) -> None:
        self.store = store
        self.assets = assets
        self.vision = vision
        self.text_embedder = text_embedder or MultimodalTextEmbedder()
        self._vision_available = True

    def search(self, query: str = "", image_path: Path | None = None, top_k: int = 5) -> list[VisualEvidence]:
        channels: list[tuple[str, list[tuple[str, float]]]] = []
        if query.strip():
            lexical_results = self.store.text_search(query, limit=max(20, top_k * 4))
            channels.append((
                "ocr_lexical",
                lexical_results,
            ))
            query_vector = self.text_embedder.embed([query])[0]
            channels.append((
                "ocr_semantic" if self.text_embedder.semantic else "ocr_dense_fallback",
                self.store.search(
                    query_vector, self.text_embedder.embedding_space, limit=max(20, top_k * 4)
                ),
            ))
            low_memory = os.getenv("MULTIMODAL_LOW_MEMORY_MODE", "true").casefold() in {
                "1", "true", "yes", "on",
            }
            # OCR 已直接命中时，低内存模式无需再加载大型视觉编码器。
            if lexical_results and low_memory:
                text_vector = []
            elif self._vision_available:
                try:
                    text_vector = self.vision.embed_texts([query])[0]
                except RuntimeError:
                    self._vision_available = False
                    text_vector = []
            else:
                text_vector = []
            if text_vector:
                channels.append((
                    "vision_text",
                    self.store.search(
                        text_vector, self.vision.embedding_space, limit=max(20, top_k * 4)
                    ),
                ))
        if image_path is not None:
            image_vector = self.vision.embed_images([image_path])[0]
            channels.append((
                "vision_image",
                self.store.search(
                    image_vector, self.vision.embedding_space, limit=max(20, top_k * 4)
                ),
            ))
        scores: dict[str, float] = defaultdict(float)
        hits: dict[str, int] = defaultdict(int)
        channel_names: dict[str, set[str]] = defaultdict(set)
        for channel_name, values in channels:
            for rank, (chunk_id, _raw_score) in enumerate(values, start=1):
                scores[chunk_id] += 1.0 / (60 + rank)
                hits[chunk_id] += 1
                channel_names[chunk_id].add(channel_name)
        ranked = sorted(scores, key=lambda key: (scores[key], hits[key]), reverse=True)[:max(1, top_k)]
        evidences: list[VisualEvidence] = []
        for index, chunk_id in enumerate(ranked, start=1):
            chunk = self.store.get_chunk(chunk_id)
            if not chunk:
                continue
            evidences.append(VisualEvidence(
                evidence_id=f"image_ev_{index}", asset_id=chunk.asset_id,
                doc_id=chunk.doc_id, page_number=chunk.page_number, bbox=chunk.bbox,
                source_label=chunk.source_label, modality=chunk.modality,
                score=scores[chunk_id], text=chunk.text,
                metadata={
                    "channel_hits": hits[chunk_id],
                    "channel_names": sorted(channel_names[chunk_id]),
                },
            ))
        return evidences
