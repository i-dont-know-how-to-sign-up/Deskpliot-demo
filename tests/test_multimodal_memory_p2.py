from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from deskpilot.memory.memory_models import MemoryItem, SessionMessage
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.memory.visual_memory import VisualMemoryManager
from deskpilot.multimodal.models import MediaAsset, VisualEvidence
from deskpilot.tools.tool_registry import build_default_tool_registry


class FakeAssets:
    def __init__(self, assets: list[MediaAsset]) -> None:
        self.values = {item.asset_id: item for item in assets}

    def get(self, asset_id: str) -> MediaAsset | None:
        return self.values.get(asset_id)


class FakeVectors:
    def __init__(self, ocr: dict[str, str] | None = None) -> None:
        self.ocr = ocr or {}

    def list_chunks_for_asset(self, asset_id: str):
        text = self.ocr.get(asset_id, "")
        return [SimpleNamespace(text=text)] if text else []


class FakeMultimodal:
    def __init__(
        self, assets: list[MediaAsset], *, similar: dict[str, list[str]] | None = None,
        ocr: dict[str, str] | None = None,
    ) -> None:
        self.assets = FakeAssets(assets)
        self.vector_store = FakeVectors(ocr)
        self.similar = similar or {}
        self.deleted: list[str] = []
        self.indexed: list[str] = []

    def search(self, query: str = "", image_path: Path | None = None, top_k: int = 5):
        source = next(
            (item for item in self.assets.values.values() if Path(item.original_path) == image_path),
            None,
        )
        return [
            VisualEvidence(f"ev-{asset_id}", asset_id, f"{asset_id}.png", "image", 0.9)
            for asset_id in self.similar.get(source.asset_id if source else "", [])[:top_k]
        ]

    def delete_asset(self, asset_id: str) -> bool:
        self.deleted.append(asset_id)
        return self.assets.values.pop(asset_id, None) is not None

    def index_file(self, path: Path) -> list[dict]:
        self.indexed.append(str(path))
        return []

    def sensitive_assets(self, asset_ids: list[str]) -> list[MediaAsset]:
        return [
            asset for asset_id in asset_ids
            if (asset := self.assets.get(asset_id)) is not None and asset.sensitivity != "normal"
        ]


def make_asset(
    asset_id: str, tmp_path: Path, *, width: int = 100, height: int = 100,
    sensitivity: str = "normal",
) -> MediaAsset:
    original = tmp_path / f"{asset_id}.png"
    original.write_bytes(b"fixture")
    return MediaAsset(
        asset_id=asset_id, sha256=asset_id, media_type="image/png",
        original_path=str(original), source_path=str(original), width=width, height=height,
        size_bytes=7, thumbnail_path=str(tmp_path / f"{asset_id}.thumb.png"),
        sensitivity=sensitivity,
    )


def make_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(
        tmp_path / "memory" / "catalog.sqlite", tmp_path / "workspace",
        vector_provider="sqlite",
    )
    store._embed_text = lambda text: [float(len(text)), 1.0]
    return store


def test_session_message_migrates_legacy_attachment_ids() -> None:
    message = SessionMessage.from_dict({
        "message_id": "m1", "session_id": "s1", "role": "user", "content": "看图",
        "metadata": {"attachment_ids": ["asset_one"]},
    })
    assert message.attachment_ids == ["asset_one"]
    assert SessionMessage.from_dict({
        "message_id": "m2", "session_id": "s1", "role": "user", "content": "看图",
        "metadata": {"attachment_ids": "invalid"},
    }).attachment_ids == []


def test_session_store_writes_attachment_ids_outside_metadata(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session()
    message = store.append_message(session.session_id, "user", "看图", attachment_ids=["a", "a"])
    raw = (tmp_path / "sessions" / session.session_id / "messages.jsonl").read_text(encoding="utf-8")
    assert message.attachment_ids == ["a"]
    assert json.loads(raw)["attachment_ids"] == ["a"]
    assert "base64" not in raw.casefold()


def test_memory_store_migrates_old_schema_and_round_trips_visual_refs(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "catalog.sqlite"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.execute(
            """create table memories (
                memory_id text primary key, scope text not null, memory_type text not null,
                content text not null, source_session_id text, source_message_ids text not null,
                confidence real not null, status text not null, tags text not null,
                created_at text not null, updated_at text not null, expires_at text,
                embedding text not null
            )"""
        )
    store = make_store(tmp_path)
    item = store.add_memory(
        "user", "artifact", "白板中的发布计划", status="pending",
        asset_refs=["asset_a"], evidence_refs=["asset_a#region-1"],
    )
    assert item is not None
    loaded = store.get_memory(item.memory_id)
    assert loaded is not None
    assert loaded.asset_refs == ["asset_a"]
    assert loaded.evidence_refs == ["asset_a#region-1"]


def test_visual_memory_is_pending_then_approved_and_cross_session_searchable(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    asset = make_asset("asset_a", tmp_path)
    manager = VisualMemoryManager(store, FakeMultimodal([asset]))
    item = manager.remember("白板上的项目代号是 Aurora", [asset.asset_id], source_session_id="session_a")
    assert item.status == "pending"
    assert manager.multimodal.indexed == [asset.original_path]
    assert store.search("Aurora", session_id="session_b") == []
    assert manager.approve(item.memory_id) is True
    result = manager.search("Aurora 项目代号", session_id="session_b")
    assert [value.memory_id for value in result] == [item.memory_id]


@pytest.mark.parametrize(
    ("sensitivity", "ocr"),
    [("sensitive", "普通截图"), ("normal", "验证码 123456"), ("normal", "api_key=sk-1234567890123456")],
)
def test_sensitive_visual_memory_is_rejected(
    tmp_path: Path, sensitivity: str, ocr: str,
) -> None:
    store = make_store(tmp_path)
    asset = make_asset("asset_sensitive", tmp_path, sensitivity=sensitivity)
    manager = VisualMemoryManager(store, FakeMultimodal([asset], ocr={asset.asset_id: ocr}))
    with pytest.raises(PermissionError):
        manager.remember("请记住这张图片", [asset.asset_id])
    assert store.list_pending_memories() == []


def test_duplicate_clipboard_ingest_only_raises_asset_sensitivity(tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    from deskpilot.multimodal.asset_store import AssetStore

    buffer = BytesIO()
    Image.new("RGB", (16, 16), (20, 30, 40)).save(buffer, format="PNG")
    assets = AssetStore(tmp_path / "assets")
    normal = assets.ingest_bytes(buffer.getvalue(), sensitivity="normal")
    sensitive = assets.ingest_bytes(buffer.getvalue(), sensitivity="sensitive")
    assert sensitive.asset_id == normal.asset_id
    assert sensitive.sensitivity == "sensitive"
    assert assets.get(normal.asset_id).sensitivity == "sensitive"


def test_image_query_recalls_memory_associated_with_similar_asset(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    query = make_asset("query", tmp_path)
    remembered = make_asset("remembered", tmp_path)
    manager = VisualMemoryManager(
        store, FakeMultimodal([query, remembered], similar={"query": ["remembered"]}),
    )
    item = manager.remember("这是 Aurora 项目的架构白板", [remembered.asset_id])
    assert manager.approve(item.memory_id)
    result = manager.search(query_asset_ids=[query.asset_id])
    assert [value.memory_id for value in result] == [item.memory_id]


def test_visual_context_obeys_image_count_and_pixel_budgets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VISUAL_MEMORY_MAX_IMAGES", "2")
    monkeypatch.setenv("VISUAL_MEMORY_MAX_TOTAL_PIXELS", "18000")
    store = make_store(tmp_path)
    assets = [make_asset(f"asset_{index}", tmp_path, width=100, height=100) for index in range(3)]
    manager = VisualMemoryManager(store, FakeMultimodal(assets))
    memory = MemoryItem(
        "m1", "user", "artifact", "三张设计图", confidence=0.9,
        status="active", asset_refs=[item.asset_id for item in assets],
    )
    selected = manager.select_context([memory])
    assert selected.asset_ids == ["asset_0"]
    assert selected.total_pixels == 10000
    assert selected.memories[0].asset_refs == ["asset_0"]
    assert selected.dropped_assets == ["asset_1", "asset_2"]


def test_referenced_asset_requires_confirmation_and_cascade(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    asset = make_asset("asset_a", tmp_path)
    multimodal = FakeMultimodal([asset])
    manager = VisualMemoryManager(store, multimodal)
    item = manager.remember("需要保留的设计图", [asset.asset_id])
    assert manager.approve(item.memory_id)
    preview = manager.delete_asset(asset.asset_id)
    assert preview["requires_confirmation"] is True
    with pytest.raises(PermissionError):
        manager.delete_asset(asset.asset_id, confirm=True, cascade_memories=False)
    result = manager.delete_asset(asset.asset_id, confirm=True, cascade_memories=True)
    assert result["deleted"] is True
    assert store.get_memory(item.memory_id).status == "deleted"
    assert multimodal.deleted == [asset.asset_id]


def test_visual_memory_tools_expose_pending_and_human_approval(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    asset = make_asset("asset_a", tmp_path)
    manager = VisualMemoryManager(store, FakeMultimodal([asset]))
    registry = build_default_tool_registry(None, workspace_root=tmp_path, visual_memory=manager)
    created = registry.call(
        "memory.remember_visual", content="记住白板上的路线图", asset_ids=[asset.asset_id],
    )
    assert created.ok and created.output["status"] == "pending"
    memory_id = created.output["memory"]["memory_id"]
    waiting = registry.call("memory.approve_visual", memory_id=memory_id)
    assert waiting.status == "waiting_human"
    approved = registry.call_approved("memory.approve_visual", memory_id=memory_id)
    assert approved.ok and approved.output["approved"] is True
    delete_waiting = registry.call(
        "assets.delete", asset_id=asset.asset_id, cascade_memories=True,
    )
    assert delete_waiting.status == "waiting_human"


def test_attachment_intent_router_can_create_pending_visual_memory(tmp_path: Path) -> None:
    from deskpilot.core.agent import DocumentQAAgent
    from deskpilot.intent.schemas import IntentDecision
    from deskpilot.rag.vector_index import DocumentIndex

    index = DocumentIndex(tmp_path / "index" / "index.json")
    agent = DocumentQAAgent(index)
    agent.session_store = SessionStore(tmp_path / "sessions")
    agent.memory_store = make_store(tmp_path / "isolated")
    asset = make_asset("asset_a", tmp_path)
    agent.multimodal = FakeMultimodal([asset])
    agent.visual_memory = VisualMemoryManager(agent.memory_store, agent.multimodal)
    agent.tool_registry = build_default_tool_registry(
        index, workspace_root=tmp_path, visual_memory=agent.visual_memory,
    )
    agent.intent_router.route = lambda *_args, **_kwargs: IntentDecision(
        mode="tool_call", tool_name="memory.remember_visual",
        reason="用户显式要求记住图片", arguments={"content": "这是 Aurora 项目设计图"},
    )

    result = agent.answer_multimodal("请记住这张设计图", [asset.asset_id])
    pending = agent.memory_store.list_pending_memories(session_id=result.session_id)
    messages = agent.session_store.read_messages(result.session_id)
    assert len(pending) == 1
    assert pending[0].asset_refs == [asset.asset_id]
    assert messages[0].attachment_ids == [asset.asset_id]
    assert any(step.name == "remember_visual" for step in result.steps)

