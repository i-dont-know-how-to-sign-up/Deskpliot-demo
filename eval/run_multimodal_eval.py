from __future__ import annotations

import argparse
import json
import statistics
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


DATASET = Path(__file__).parent / "dataset" / "multimodal_cases.jsonl"


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


def load_cases(count: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:count] if count is not None else rows


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
    else:
        ingestion.ingest_image(atlas)
        ingestion.ingest_image(nebula)
        retriever = MultimodalRetriever(vectors, assets, vision)
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
        "id": case["id"], "subset": case["subset"], "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
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
        "id": case["id"], "subset": case["subset"], "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "answer": answer,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def write_report(results: list[dict[str, Any]], path: Path, mode: str) -> None:
    completed = [item for item in results if item["status"] != "skipped"]
    passed = [item for item in completed if item["status"] == "passed"]
    latencies = [float(item["latency_ms"]) for item in completed]
    lines = [
        "# DeskPilot 多模态 P0/P1 评测报告", "",
        f"- 时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 模式：{mode}", f"- 用例数：{len(results)}",
        f"- 完成/跳过：{len(completed)}/{len(results) - len(completed)}",
        f"- 通过率：{len(passed) / max(1, len(completed)):.2%}",
        f"- 平均响应时间：{statistics.mean(latencies) if latencies else 0.0:.2f} ms", "",
        "| 用例 | 模块 | 状态 | 延迟(ms) |", "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {item['id']} | {item['subset']} | {item['status']} | {item.get('latency_ms', 0)} |"
        for item in results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DeskPilot 多模态 P0/P1 评测。")
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--report", type=Path, default=Path("eval/reports/multimodal_latest.md"))
    args = parser.parse_args()
    cases = load_cases(args.count)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        atlas = make_fixture(root / "atlas.png", (190, 45, 45), "ATLAS-503")
        nebula = make_fixture(root / "nebula.png", (35, 75, 190), "NEBULA-204")
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
            requires_vlm = bool(case.get("runtime", {}).get("requires_vlm"))
            if requires_vlm and args.mode != "api":
                results.append({"id": case["id"], "subset": case["subset"], "status": "skipped", "latency_ms": 0})
                continue
            results.append(
                evaluate_vlm_case(case, root / case["id"], atlas, nebula)
                if requires_vlm else evaluate_case(case, root / case["id"], atlas, nebula)
            )
    write_report(results, args.report, args.mode)
    failures = sum(item["status"] == "failed" for item in results)
    print(f"评测完成：通过 {sum(item['status'] == 'passed' for item in results)}，失败 {failures}，报告 {args.report}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
