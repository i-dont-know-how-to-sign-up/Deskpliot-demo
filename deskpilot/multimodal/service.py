from __future__ import annotations

import re
import os
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
from .text_embedding import MultimodalTextEmbedder
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
        self.text_embedder = MultimodalTextEmbedder()
        self.ingestion = MultimodalIngestionService(
            self.assets, self.vector_store, self.ocr, self.vision_embedding, self.text_embedder
        )
        self.retriever = MultimodalRetriever(
            self.vector_store, self.assets, self.vision_embedding, self.text_embedder
        )

    def add_chat_attachment(self, path: Path) -> MediaAsset:
        return self.assets.ingest_file(path, source_kind="chat", metadata={"source_name": path.name})

    def add_chat_attachment_bytes(self, content: bytes, source_name: str = "clipboard.png") -> MediaAsset:
        return self.assets.ingest_bytes(
            content, source_path=None, source_kind="chat", sensitivity="sensitive",
            metadata={"source_name": source_name, "source": "clipboard"},
        )

    def inspect_image(self, path: Path, include_ocr: bool = True) -> dict[str, Any]:
        asset = self.add_chat_attachment(path)
        regions = self.ocr.recognize(Path(asset.original_path)) if include_ocr else []
        return {
            "asset": asset.to_dict(), "ocr_provider": self.ocr.provider,
            "ocr_text": "\n".join(region.text for region in regions),
            "regions": [region.__dict__ for region in regions],
        }

    def answer(
        self, question: str, asset_ids: list[str], *, confirm_sensitive: bool = False,
        event_callback=None,
    ) -> str:
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
        self._ensure_cloud_ready()
        sensitive = [asset for asset in assets if asset and asset.sensitivity != "normal"]
        self._enforce_sensitive_policy(sensitive, confirm_sensitive=confirm_sensitive)
        answer = self.vision_llm.answer(
            question, [Path(asset.original_path) for asset in assets if asset],
            event_callback=event_callback,
        )
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

    def answer_from_index(
        self, query: str, top_k: int = 4, *, confirm_sensitive: bool = False,
        event_callback=None,
    ) -> dict[str, Any]:
        """先检索有限图片，再把命中的原图交给 VLM，避免发送整个图片库。"""
        self._ensure_cloud_ready()
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
        selected_assets = [self.assets.get(item.asset_id) for item in selected]
        sensitive = [asset for asset in selected_assets if asset and asset.sensitivity != "normal"]
        if sensitive and not confirm_sensitive and self._sensitive_policy() == "confirm":
            return {
                "answer": "",
                "evidences": [item.to_dict() for item in selected],
                "permission": self.sensitive_permission(sensitive),
            }
        self._enforce_sensitive_policy(sensitive, confirm_sensitive=confirm_sensitive)
        mapping = "\n".join(
            f"图片{index}的可核对来源是 {item.source_label}；OCR 摘要：{item.text[:600]}"
            for index, item in enumerate(selected, start=1)
        )
        answer = self.vision_llm.answer(
            f"请结合图片本身和以下来源映射回答问题。不要使用未检索到的资料。\n{mapping}\n问题：{query}",
            [Path(asset.original_path) for asset in selected_assets if asset],
            event_callback=event_callback,
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

    def delete_asset(self, asset_id: str) -> bool:
        self.vector_store.delete_asset(asset_id)
        return self.assets.delete(asset_id)

    def stats(self) -> dict[str, Any]:
        values: dict[str, Any] = self.vector_store.stats()
        values["ocr_provider"] = self.ocr.provider
        values["ocr_enabled"] = self.ocr.provider != "disabled"
        values["vision_embedding_provider"] = self.vision_embedding.provider
        values["vision_embedding_available"] = self.ingestion.vision_available
        values["vision_embedding_error"] = self.ingestion.vision_error
        values["vision_llm_available"] = self.vision_llm.available
        values["cloud_upload_enabled"] = self.vision_llm.allow_cloud
        values["text_embedding_space"] = self.text_embedder.embedding_space
        values["semantic_text_embedding"] = self.text_embedder.semantic
        values["sensitive_image_policy"] = self._sensitive_policy()
        return values

    def sensitive_assets(self, asset_ids: list[str]) -> list[MediaAsset]:
        assets = [self.assets.get(asset_id) for asset_id in dict.fromkeys(asset_ids)]
        return [asset for asset in assets if asset and asset.sensitivity != "normal"]

    @staticmethod
    def sensitive_permission(assets: list[MediaAsset]) -> dict[str, Any]:
        names = [str(item.metadata.get("source_name") or Path(item.original_path).name) for item in assets]
        return {
            "requires_confirmation": True,
            "risk_level": "high",
            "reasons": [
                "图片被标记为敏感或来自剪贴板/截图。",
                "确认后图片内容将发送到云端视觉模型。",
                "涉及图片：" + "、".join(names[:8]),
            ],
        }

    def _ensure_cloud_ready(self) -> None:
        if not self.vision_llm.allow_cloud:
            raise PermissionError("云端图片上传未启用，请在 .env 设置 ALLOW_CLOUD_IMAGE_UPLOAD=true")
        if not self.vision_llm.available:
            raise RuntimeError("视觉模型不可用，请检查 VISION_PROVIDER、VISION_MODEL 和 API Key")

    @staticmethod
    def _sensitive_policy() -> str:
        value = os.getenv("SENSITIVE_IMAGE_POLICY", "confirm").strip().casefold()
        return value if value in {"block", "confirm", "allow"} else "confirm"

    def _enforce_sensitive_policy(
        self, assets: list[MediaAsset], *, confirm_sensitive: bool,
    ) -> None:
        if not assets:
            return
        policy = self._sensitive_policy()
        if policy == "block":
            raise PermissionError("敏感图片策略为 block，只允许本地 OCR，不允许发送到云端 VLM")
        if policy == "confirm" and not confirm_sensitive:
            raise PermissionError("敏感图片发送到云端视觉模型前需要人工确认")
