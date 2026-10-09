from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .memory_models import MemoryItem
from .memory_store import MemoryStore
from ..multimodal.models import MediaAsset
from ..multimodal.service import MultimodalService


@dataclass
class VisualMemoryContext:
    """进入模型上下文的视觉记忆引用；不携带图片二进制或 Base64。"""

    memories: list[MemoryItem] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    total_pixels: int = 0
    dropped_assets: list[str] = field(default_factory=list)


class VisualMemoryManager:
    """P2 多模态记忆门面：策略、引用检索、预算选择和级联删除。"""

    _SENSITIVE_PATTERNS = (
        re.compile(r"(?:password|passwd|api[_ -]?key|access[_ -]?token|bearer)\s*[:=]", re.I),
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        re.compile(r"\b\d{17}[0-9Xx]\b"),
        re.compile(r"\b(?:\d[ -]?){16,19}\b"),
        re.compile(r"(?:验证码|动态码|短信码|OTP)\D{0,6}\d{4,8}", re.I),
    )

    def __init__(self, memory_store: MemoryStore, multimodal: MultimodalService) -> None:
        self.memory_store = memory_store
        self.multimodal = multimodal

    def remember(
        self,
        content: str,
        asset_ids: list[str],
        *,
        source_session_id: str | None = None,
        source_message_ids: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.9,
    ) -> MemoryItem:
        """创建待审批视觉记忆；显式记忆也不能绕过敏感信息策略。"""
        normalized_assets = list(dict.fromkeys(str(item) for item in asset_ids if str(item)))
        if not normalized_assets:
            raise ValueError("视觉记忆至少需要一个 asset_id")
        assets = self._load_assets(normalized_assets)
        self._ensure_assets_indexed(assets)
        reason = self._sensitive_reason(content, assets)
        if reason:
            raise PermissionError(reason)
        for asset_id in normalized_assets:
            for existing in self.memory_store.memories_referencing_asset(asset_id):
                if self._normalize_fact(existing.content) == self._normalize_fact(content):
                    return existing
        item = self.memory_store.add_memory(
            scope="user",
            memory_type="artifact",
            content=content,
            source_session_id=source_session_id,
            source_message_ids=source_message_ids,
            confidence=confidence,
            status="pending",
            tags=list(dict.fromkeys(["visual", *(tags or [])])),
            asset_refs=normalized_assets,
            evidence_refs=evidence_refs or [],
        )
        if item is None:
            raise PermissionError("视觉记忆内容触发敏感信息策略，未写入长期记忆")
        return item

    def approve(self, memory_id: str) -> bool:
        item = self.memory_store.get_memory(memory_id)
        if item is None or item.status != "pending" or not item.asset_refs:
            return False
        assets = self._load_assets(item.asset_refs)
        if self._sensitive_reason(item.content, assets):
            return False
        return self.memory_store.approve_memory(memory_id)

    def search(
        self,
        query: str = "",
        *,
        query_asset_ids: list[str] | None = None,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """融合文本记忆召回与相似图片关联召回，并按 memory_id 去重。"""
        ranked: dict[str, MemoryItem] = {}
        if query.strip():
            for item in self.memory_store.search(query, session_id=session_id, top_k=max(top_k * 2, 8)):
                if item.asset_refs:
                    ranked[item.memory_id] = item
        visual_asset_ids: list[str] = []
        for asset_id in dict.fromkeys(query_asset_ids or []):
            asset = self.multimodal.assets.get(asset_id)
            if asset is None:
                continue
            visual_asset_ids.append(asset_id)
            try:
                evidences = self.multimodal.search(
                    query="", image_path=Path(asset.original_path), top_k=max(top_k * 3, 10),
                )
                visual_asset_ids.extend(item.asset_id for item in evidences)
            except (RuntimeError, ValueError, OSError):
                # 视觉编码器不可用时仍保留同 asset ID 精确关联召回。
                pass
        for item in self.memory_store.search_by_asset_refs(
            list(dict.fromkeys(visual_asset_ids)), session_id=session_id, top_k=max(top_k * 2, 8),
        ):
            previous = ranked.get(item.memory_id)
            if previous is None or item.retrieval_score > previous.retrieval_score:
                ranked[item.memory_id] = item
        return sorted(ranked.values(), key=lambda item: item.retrieval_score, reverse=True)[:max(1, top_k)]

    def select_context(self, memories: list[MemoryItem]) -> VisualMemoryContext:
        """按图片数量和总像素预算选择视觉引用，文本 token 仍由 ContextBuilder 管理。"""
        max_images = max(1, min(int(os.getenv("VISUAL_MEMORY_MAX_IMAGES", "4")), 12))
        max_pixels = max(1, int(os.getenv("VISUAL_MEMORY_MAX_TOTAL_PIXELS", "12000000")))
        selected_memories: list[MemoryItem] = []
        asset_ids: list[str] = []
        dropped: list[str] = []
        total_pixels = 0
        for memory in memories:
            accepted_refs: list[str] = []
            for asset_id in memory.asset_refs:
                if asset_id in asset_ids:
                    accepted_refs.append(asset_id)
                    continue
                asset = self.multimodal.assets.get(asset_id)
                if asset is None:
                    continue
                pixels = max(1, int(asset.width) * int(asset.height))
                if len(asset_ids) >= max_images or total_pixels + pixels > max_pixels:
                    dropped.append(asset_id)
                    continue
                asset_ids.append(asset_id)
                total_pixels += pixels
                accepted_refs.append(asset_id)
            if accepted_refs:
                # 上下文副本只暴露真正通过图片预算的引用，原始记忆记录保持不变。
                selected_memories.append(replace(memory, asset_refs=accepted_refs))
        return VisualMemoryContext(selected_memories, asset_ids, total_pixels, dropped)

    def delete_asset(
        self, asset_id: str, *, confirm: bool = False, cascade_memories: bool = False,
    ) -> dict[str, Any]:
        references = self.memory_store.memories_referencing_asset(asset_id)
        if references and not confirm:
            return {
                "deleted": False,
                "requires_confirmation": True,
                "asset_id": asset_id,
                "memory_ids": [item.memory_id for item in references],
                "reason": "图片仍被长期或待审批记忆引用",
            }
        if references and not cascade_memories:
            raise PermissionError("图片仍被记忆引用；确认后还需设置 cascade_memories=true")
        for item in references:
            self.memory_store.delete_memory(item.memory_id)
        deleted = self.multimodal.delete_asset(asset_id)
        return {
            "deleted": deleted,
            "requires_confirmation": False,
            "asset_id": asset_id,
            "deleted_memory_ids": [item.memory_id for item in references],
        }

    def _load_assets(self, asset_ids: list[str]) -> list[MediaAsset]:
        assets = [self.multimodal.assets.get(asset_id) for asset_id in asset_ids]
        missing = [asset_id for asset_id, asset in zip(asset_ids, assets) if asset is None]
        if missing:
            raise FileNotFoundError("找不到图片资产：" + "、".join(missing))
        return [asset for asset in assets if asset is not None]

    def _ensure_assets_indexed(self, assets: list[MediaAsset]) -> None:
        """显式记忆前按需生成 OCR/视觉索引，保证敏感检查和相似图召回可用。"""
        index_file = getattr(self.multimodal, "index_file", None)
        if not callable(index_file):
            return
        for asset in assets:
            if self.multimodal.vector_store.list_chunks_for_asset(asset.asset_id):
                continue
            index_file(Path(asset.original_path))

    def _sensitive_reason(self, content: str, assets: list[MediaAsset]) -> str:
        if any(asset.sensitivity != "normal" for asset in assets):
            return "敏感图片、截图或剪贴板图片禁止进入长期视觉记忆"
        searchable = content + "\n" + "\n".join(self._asset_ocr_text(asset.asset_id) for asset in assets)
        if any(pattern.search(searchable) for pattern in self._SENSITIVE_PATTERNS):
            return "图片或记忆摘要包含凭证、验证码或身份/银行卡信息，禁止长期保存"
        return ""

    def _asset_ocr_text(self, asset_id: str) -> str:
        return "\n".join(
            chunk.text for chunk in self.multimodal.vector_store.list_chunks_for_asset(asset_id)
            if chunk.text.strip()
        )

    @staticmethod
    def _normalize_fact(content: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", content.casefold())
