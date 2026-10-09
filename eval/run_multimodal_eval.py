from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from deskpilot.multimodal.asset_store import AssetStore
from deskpilot.multimodal.image_processor import ImageProcessor
from deskpilot.multimodal.ingestion import MultimodalIngestionService
from deskpilot.multimodal.models import VectorRecord
from deskpilot.multimodal.providers.ocr import OCRRegion
from deskpilot.multimodal.retriever import MultimodalRetriever
from deskpilot.multimodal.service import MultimodalService
from deskpilot.multimodal.vector_store import MultimodalVectorStore
from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.visual_memory import VisualMemoryManager
from deskpilot import __version__


DATASET = Path(__file__).parent / "dataset" / "multimodal_cases.jsonl"
SUITES_DIR = Path(__file__).parent / "suites"


class FixtureOCR:
    provider = "eval-fixture"

    def recognize(self, image_path: Path) -> list[OCRRegion]:
        with Image.open(image_path) as image:
            red, _green, blue = image.resize((1, 1)).getpixel((0, 0))
        value = "ATLAS-503 overload recovery" if red > blue else "NEBULA-204 cache policy"
        return [OCRRegion(value, 1.0, (0.08, 0.2, 0.92, 0.5))]


class FixtureVision:
    provider = "eval-fixture"
    embedding_space = "vision/eval-fixture-v1"
    model_id = "deskpilot/eval-fixture"
    model_revision = "1"

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        result = []
        for path in paths:
            with Image.open(path) as image:
                red, _green, blue = image.resize((1, 1)).getpixel((0, 0))
            result.append([1.0, 0.0] if red > blue else [0.0, 1.0])
        return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "ATLAS" in text.upper() else [0.0, 1.0] for text in texts]


class FixtureTextEmbedder:
    semantic = False
    embedding_space = "text/local-hash-v1"
    model_id = "deskpilot/local-hash"
    model_revision = "1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [local_hash_embedding(text) for text in texts]


class FixtureVisualMemoryService:
    """复用真实资产、向量和检索实现，不加载 OCR/SigLIP2 大模型。"""

    def __init__(
        self, assets: AssetStore, vectors: MultimodalVectorStore,
        retriever: MultimodalRetriever,
    ) -> None:
        self.assets = assets
        self.vector_store = vectors
        self.retriever = retriever

    def search(self, query: str = "", image_path: Path | None = None, top_k: int = 5):
        return self.retriever.search(query, image_path, top_k)

    def delete_asset(self, asset_id: str) -> bool:
        self.vector_store.delete_asset(asset_id)
        return self.assets.delete(asset_id)


def load_cases(
    count: int | None = None, case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if case_ids:
        by_id = {str(item["id"]): item for item in rows}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown multimodal case IDs: {', '.join(missing)}")
        rows = [by_id[case_id] for case_id in dict.fromkeys(case_ids)]
    return rows[:count] if count is not None else rows


def load_suite(name: str) -> dict[str, Any]:
    path = SUITES_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown multimodal evaluation suite: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("case_ids"), list):
        raise ValueError(f"Invalid multimodal evaluation suite: {path}")
    return data


def make_fixture(path: Path, color: tuple[int, int, int], label: str) -> Path:
    image = Image.new("RGB", (480, 260), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 80, 455, 180), fill=(250, 250, 250))
    draw.text((50, 115), label, fill=(15, 23, 42))
    image.save(path)
    return path


def evaluate_case(case: dict[str, Any], root: Path, atlas: Path, nebula: Path) -> dict[str, Any]:
    started = time.perf_counter()
    task = case["task"]
    expected = case["expected"]
    checks: dict[str, bool] = {}
    assets = AssetStore(root / "assets")
    vectors = MultimodalVectorStore(root / "vectors.sqlite3")
    vision = FixtureVision()
    ingestion = MultimodalIngestionService(assets, vectors, FixtureOCR(), vision)
    if task == "asset_dedup":
        first = assets.ingest_file(atlas)
        copy = root / "renamed.png"
        copy.write_bytes(atlas.read_bytes())
        checks["same_asset_id"] = first.asset_id == assets.ingest_file(copy).asset_id
    elif task == "invalid_image":
        try:
            assets.ingest_bytes(b"not an image")
            checks["rejected"] = False
        except ValueError:
            checks["rejected"] = True
    elif task == "pixel_limit":
        try:
            ImageProcessor(max_pixels=100).process_file(atlas)
            checks["rejected"] = False
        except ValueError:
            checks["rejected"] = True
    elif task == "space_isolation":
        vectors.upsert_vector(VectorRecord("a", "asset", "same", "image", "vision/a", "a", "1", 2, [1, 0]))
        vectors.upsert_vector(VectorRecord("b", "asset", "foreign", "image", "vision/b", "b", "1", 2, [1, 0]))
        vectors.upsert_vector(VectorRecord("c", "asset", "dimension", "image", "vision/a", "a", "1", 3, [1, 0, 0]))
        checks["foreign_vectors_excluded"] = vectors.search([1, 0], "vision/a") == [("same", 1.0)]
    elif task == "catalog_privacy":
        assets.ingest_file(atlas)
        checks["base64_absent"] = "base64" not in assets.database.read_bytes().decode("latin-1", errors="ignore").casefold()
    elif task == "clipboard_sensitivity":
        service = MultimodalService(root)
        asset = service.add_chat_attachment_bytes(atlas.read_bytes(), "clipboard.png")
        checks["sensitive"] = asset.sensitivity == "sensitive"
        checks["confirmation_required"] = service.sensitive_permission([asset])["requires_confirmation"] is True
    elif task == "sqlite_wal":
        with assets._connect() as connection:
            checks["asset_wal"] = connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        with vectors._connect() as connection:
            checks["vector_wal"] = connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
    elif task == "asset_lifecycle":
        service = MultimodalService(root)
        asset = service.add_chat_attachment(atlas)
        original = Path(asset.original_path)
        thumbnail = Path(asset.thumbnail_path)
        checks["deleted"] = service.delete_asset(asset.asset_id)
        checks["files_removed"] = not original.exists() and not thumbnail.exists()
    elif task.startswith("visual_memory_"):
        text = FixtureTextEmbedder()
        ingestion = MultimodalIngestionService(assets, vectors, FixtureOCR(), vision, text)
        ingestion.ingest_image(atlas)
        ingestion.ingest_image(nebula)
        atlas_asset = assets.ingest_file(atlas)
        nebula_asset = assets.ingest_file(nebula)
        retriever = MultimodalRetriever(vectors, assets, vision, text)
        facade = FixtureVisualMemoryService(assets, vectors, retriever)
        memory = MemoryStore(
            root / "memory" / "catalog.sqlite", root / "memory_workspace",
            vector_provider="sqlite",
        )
        memory._embed_text = local_hash_embedding
        manager = VisualMemoryManager(memory, facade)
        if task == "visual_memory_pending":
            item = manager.remember("ATLAS-503 是红色告警面板", [atlas_asset.asset_id])
            checks["pending"] = item.status == "pending"
            checks["excluded_before_approval"] = manager.search("ATLAS-503") == []
        elif task == "visual_memory_cross_session":
            item = manager.remember(
                "ATLAS-503 是红色告警面板", [atlas_asset.asset_id],
                source_session_id="session_a",
            )
            manager.approve(item.memory_id)
            recalled = manager.search("ATLAS-503 红色告警", session_id="session_b")
            checks["cross_session_recalled"] = any(value.memory_id == item.memory_id for value in recalled)
        elif task == "visual_memory_image_recall":
            item = manager.remember("这张图对应 ATLAS 过载恢复", [atlas_asset.asset_id])
            manager.approve(item.memory_id)
            recalled = manager.search(query_asset_ids=[atlas_asset.asset_id])
            checks["image_recalled"] = any(value.memory_id == item.memory_id for value in recalled)
        elif task == "visual_memory_sensitive":
            sensitive = assets.ingest_bytes(
                atlas.read_bytes(), source_path=None, source_kind="clipboard",
                sensitivity="sensitive", metadata={"source": "clipboard"},
            )
            try:
                manager.remember("保存剪贴板截图", [sensitive.asset_id])
                checks["rejected"] = False
            except PermissionError:
                checks["rejected"] = True
            checks["not_persisted"] = memory.list_pending_memories() == []
        elif task == "visual_memory_delete_cascade":
            item = manager.remember("ATLAS 图片记忆", [atlas_asset.asset_id])
            manager.approve(item.memory_id)
            preview = manager.delete_asset(atlas_asset.asset_id)
            checks["confirmation_required"] = preview["requires_confirmation"] is True
            deleted = manager.delete_asset(
                atlas_asset.asset_id, confirm=True, cascade_memories=True,
            )
            checks["cascade_deleted"] = bool(deleted["deleted"])
            checks["memory_deleted"] = memory.get_memory(item.memory_id).status == "deleted"
        elif task == "visual_memory_budget":
            previous_images = os.environ.get("VISUAL_MEMORY_MAX_IMAGES")
            previous_pixels = os.environ.get("VISUAL_MEMORY_MAX_TOTAL_PIXELS")
            os.environ["VISUAL_MEMORY_MAX_IMAGES"] = "1"
            os.environ["VISUAL_MEMORY_MAX_TOTAL_PIXELS"] = "200000"
            try:
                item = memory.add_memory(
                    "user", "artifact", "两个系统截图", status="active",
                    asset_refs=[atlas_asset.asset_id, nebula_asset.asset_id],
                )
                selected = manager.select_context([item] if item else [])
                checks["one_asset_selected"] = len(selected.asset_ids) == 1
                checks["one_asset_dropped"] = len(selected.dropped_assets) == 1
                checks["bounded_refs"] = bool(
                    selected.memories and len(selected.memories[0].asset_refs) == 1
                )
            finally:
                if previous_images is None:
                    os.environ.pop("VISUAL_MEMORY_MAX_IMAGES", None)
                else:
                    os.environ["VISUAL_MEMORY_MAX_IMAGES"] = previous_images
                if previous_pixels is None:
                    os.environ.pop("VISUAL_MEMORY_MAX_TOTAL_PIXELS", None)
                else:
                    os.environ["VISUAL_MEMORY_MAX_TOTAL_PIXELS"] = previous_pixels
    else:
        text = FixtureTextEmbedder()
        ingestion = MultimodalIngestionService(assets, vectors, FixtureOCR(), vision, text)
        ingestion.ingest_image(atlas)
        ingestion.ingest_image(nebula)
        retriever = MultimodalRetriever(vectors, assets, vision, text)
        image_query = atlas if task == "image_to_image" else None
        query = case["input"] if task != "image_to_image" else ""
        evidence = retriever.search(query, image_query, top_k=2)
        checks["has_result"] = bool(evidence)
        checks["top_source"] = bool(evidence and evidence[0].source_label == expected.get("top_source", evidence[0].source_label))
        if expected.get("must_include"):
            checks["content"] = bool(evidence and all(value in evidence[0].text for value in expected["must_include"]))
        if expected.get("min_channel_hits"):
            checks["channel_hits"] = bool(
                evidence and evidence[0].metadata.get("channel_hits", 0) >= expected["min_channel_hits"]
            )
    return {
        "case_id": case["id"], "subset": case["subset"],
        "difficulty": case.get("difficulty", "unspecified"),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "token_usage": {}, "usage_events": [],
    }


def evaluate_vlm_case(case: dict[str, Any], root: Path, atlas: Path, nebula: Path) -> dict[str, Any]:
    started = time.perf_counter()
    service = MultimodalService(root)
    first = service.add_chat_attachment(atlas)
    second = service.add_chat_attachment(nebula)
    ids = [first.asset_id] if case["id"] == "mm_qa_001" else [first.asset_id, second.asset_id]
    answer = service.answer(case["input"], ids)
    expected = case["expected"]
    checks = {
        "content": all(value.casefold() in answer.casefold() for value in expected.get("must_include", [])),
        "citations": all(value in answer for value in expected.get("citations", [expected.get("citation", "")]) if value),
    }
    return {
        "case_id": case["id"], "subset": case["subset"],
        "difficulty": case.get("difficulty", "unspecified"),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "answer": answer,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "token_usage": dict(service.vision_llm.last_usage),
        "usage_events": ([{
            "stage": "Vision Answer",
            "success": True,
            "usage_reported": True,
            **service.vision_llm.last_usage,
        }] if service.vision_llm.last_usage else []),
    }


def write_report(
    results: list[dict[str, Any]], path: Path, mode: str, *, version: str, suite: str,
) -> None:
    completed = [item for item in results if item["status"] != "skipped"]
    passed = [item for item in completed if item["status"] == "passed"]
    latencies = [float(item["latency_ms"]) for item in completed]
    lines = [
        f"# DeskPilot {version} 多模态评测报告", "",
        f"- 时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 模式：{mode}", f"- 套件：{suite or '未指定'}", f"- 用例数：{len(results)}",
        f"- 完成/跳过：{len(completed)}/{len(results) - len(completed)}",
        f"- 通过率：{len(passed) / max(1, len(completed)):.2%}",
        f"- 平均响应时间：{statistics.mean(latencies) if latencies else 0.0:.2f} ms", "",
        "| 用例 | 模块 | 难度 | 状态 | 延迟(ms) | Token |", "|---|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['case_id']} | {item['subset']} | {item.get('difficulty', 'unspecified')} | "
        f"{item['status']} | {item.get('latency_ms', 0)} | "
        f"{item.get('token_usage', {}).get('total_tokens', 'N/A')} |"
        for item in results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "project_version": args.project_version,
        "mode": args.mode,
        "suite": args.suite or "",
        "dataset": str(DATASET),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "git_commit": commit or "unknown",
        "result_count": len(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 DeskPilot 多模态 P0/P1/P2 评测。")
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--suite")
    parser.add_argument("--project-version", default=__version__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=Path("eval/reports/multimodal_latest.md"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    selected_ids = list(args.case_id or [])
    if args.suite:
        selected_ids.extend(str(item) for item in load_suite(args.suite)["case_ids"])
    cases = load_cases(args.count, selected_ids)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        atlas = make_fixture(root / "atlas.png", (190, 45, 45), "ATLAS-503")
        nebula = make_fixture(root / "nebula.png", (35, 75, 190), "NEBULA-204")
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
            requires_vlm = bool(case.get("runtime", {}).get("requires_vlm"))
            if requires_vlm and args.mode != "api":
                results.append({
                    "case_id": case["id"], "subset": case["subset"],
                    "difficulty": case.get("difficulty", "unspecified"),
                    "status": "skipped", "latency_ms": 0,
                    "token_usage": {}, "usage_events": [],
                })
                continue
            try:
                results.append(
                    evaluate_vlm_case(case, root / case["id"], atlas, nebula)
                    if requires_vlm else evaluate_case(case, root / case["id"], atlas, nebula)
                )
            except Exception as exc:
                results.append({
                    "case_id": case["id"], "subset": case["subset"],
                    "difficulty": case.get("difficulty", "unspecified"),
                    "status": "error", "error": repr(exc),
                    "latency_ms": 0, "token_usage": {}, "usage_events": [],
                })
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n",
            encoding="utf-8",
        )
    write_report(
        results, args.report, args.mode,
        version=args.project_version, suite=args.suite or "",
    )
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(_metadata(args, results), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    failures = sum(item["status"] in {"failed", "error"} for item in results)
    print(f"评测完成：通过 {sum(item['status'] == 'passed' for item in results)}，失败/错误 {failures}，报告 {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
