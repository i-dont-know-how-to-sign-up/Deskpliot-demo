from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.config import DATA_DIR, ROOT_DIR, _load_dotenv
from .asset_store import AssetStore
from .ingestion import MultimodalIngestionService
from .models import MediaAsset, VisualEvidence
from .providers.ocr import build_ocr_provider
from .providers.vision_embedding import build_vision_embedding_provider
from .providers.vision_llm import VisionLLMProvider
from .retriever import MultimodalRetriever
from .vector_store import MultimodalVectorStore


class MultimodalService:
    """P0/P1 门面：Agent、工具和 UI 不直接依赖具体模型实现。"""

    def __init__(self, data_root: Path | None = None) -> None:
        # Provider 在首次加载模型前必须看到缓存目录配置，防止大型权重回落到系统盘。
        _load_dotenv(ROOT_DIR / ".env")
        root = (data_root or DATA_DIR).resolve()
        self.assets = AssetStore(root / "workspace" / "assets")
        self.vector_store = MultimodalVectorStore(root / "index" / "multimodal_catalog.sqlite3")
        self.ocr = build_ocr_provider()
        self.vision_embedding = build_vision_embedding_provider()
        self.vision_llm = VisionLLMProvider()
        self.ingestion = MultimodalIngestionService(
            self.assets, self.vector_store, self.ocr, self.vision_embedding
        )
        self.retriever = MultimodalRetriever(self.vector_store, self.assets, self.vision_embedding)

    def add_chat_attachment(self, path: Path) -> MediaAsset:
        return self.assets.ingest_file(path, source_kind="chat", metadata={"source_name": path.name})

    def add_chat_attachment_bytes(self, content: bytes, source_name: str = "clipboard.png") -> MediaAsset:
        return self.assets.ingest_bytes(
            content, source_path=None, source_kind="chat", metadata={"source_name": source_name}
        )

    def inspect_image(self, path: Path, include_ocr: bool = True) -> dict[str, Any]:
        asset = self.add_chat_attachment(path)
        regions = self.ocr.recognize(Path(asset.original_path)) if include_ocr else []
        return {
            "asset": asset.to_dict(), "ocr_provider": self.ocr.provider,
            "ocr_text": "\n".join(region.text for region in regions),
            "regions": [region.__dict__ for region in regions],
        }

    def answer(self, question: str, asset_ids: list[str]) -> str:
        limit = max(1, min(int(__import__("os").getenv("IMAGE_MAX_COUNT_PER_TURN", "4")), 12))
        unique_ids = list(dict.fromkeys(str(value) for value in asset_ids))
        if not unique_ids:
            raise ValueError("图文问答至少需要一张图片")
        if len(unique_ids) > limit:
            raise ValueError(f"单轮最多允许 {limit} 张图片")
        assets = [self.assets.get(asset_id) for asset_id in unique_ids]
        if any(asset is None for asset in assets):
            missing = [asset_id for asset_id, asset in zip(unique_ids, assets) if asset is None]
            raise FileNotFoundError(f"找不到图片资产：{', '.join(missing)}")
        answer = self.vision_llm.answer(question, [Path(asset.original_path) for asset in assets if asset])
        if not answer:
            raise RuntimeError("视觉模型没有返回内容")
        # 至少保留图片级来源；模型遗漏引用时追加，不伪造区域定位。
        if not re.search(r"\[图片\d+\]", answer):
            answer += "\n\n图片来源：" + " ".join(f"[图片{index}]" for index in range(1, len(assets) + 1))
        return answer

    def index_file(self, path: Path) -> list[dict]:
        return self.ingestion.ingest_file(path)

    def search(self, query: str = "", image_path: Path | None = None, top_k: int = 5) -> list[VisualEvidence]:
        if not query.strip() and image_path is None:
            raise ValueError("多模态检索需要文本或图片查询")
        return self.retriever.search(query, image_path, top_k)

    def answer_from_index(self, query: str, top_k: int = 4) -> dict[str, Any]:
        """先检索有限图片，再把命中的原图交给 VLM，避免发送整个图片库。"""
        evidence = self.search(query=query, top_k=max(1, min(top_k, 8)))
        evidence = self._filter_answer_evidence(evidence)
        selected: list[VisualEvidence] = []
        seen_assets: set[str] = set()
        for item in evidence:
            if item.asset_id in seen_assets:
                continue
            selected.append(item)
            seen_assets.add(item.asset_id)
            if len(selected) >= 4:
                break
        if not selected:
            return {"answer": "当前多模态索引中没有足够的相关图片证据。", "evidences": []}
        mapping = "\n".join(
            f"图片{index}的可核对来源是 {item.source_label}；OCR 摘要：{item.text[:600]}"
            for index, item in enumerate(selected, start=1)
        )
        answer = self.vision_llm.answer(
            f"请结合图片本身和以下来源映射回答问题。不要使用未检索到的资料。\n{mapping}\n问题：{query}",
            [Path(self.assets.get(item.asset_id).original_path) for item in selected if self.assets.get(item.asset_id)],
        )
        return {"answer": answer, "evidences": [item.to_dict() for item in selected]}

    @staticmethod
    def _filter_answer_evidence(evidence: list[VisualEvidence]) -> list[VisualEvidence]:
        """有 OCR/文本交叉支持时，过滤仅凭视觉相似度进入的弱候选。"""
        lexical = [
            item for item in evidence
            if "ocr_lexical" in set(item.metadata.get("channel_names") or [])
        ]
        if lexical:
            return lexical

        # 新检索结果没有字面命中时保留跨模态语义回退能力。
        named = [item for item in evidence if item.metadata.get("channel_names")]
        if named:
            return named

        # 兼容升级前或第三方构造的、尚未记录通道名称的证据。
        supported = [item for item in evidence if int(item.metadata.get("channel_hits", 0)) >= 2]
        return supported or evidence

    def attachment_view(self, asset_id: str) -> dict[str, Any] | None:
        asset = self.assets.get(asset_id)
        if not asset:
            return None
        return {
            "assetId": asset.asset_id,
            "name": str(asset.metadata.get("source_name") or Path(asset.source_path or asset.original_path).name),
            "thumbnail": Path(asset.thumbnail_path).as_uri(),
            "original": Path(asset.original_path).as_uri(),
            "width": asset.width,
            "height": asset.height,
            "size": asset.size_bytes,
        }

    def stats(self) -> dict[str, int]:
        return self.vector_store.stats()
