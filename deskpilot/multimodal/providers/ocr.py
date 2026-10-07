from __future__ import annotations

import os
import json
import gc
import re
import subprocess
import sys
import tempfile
from importlib.metadata import version
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class OCRRegion:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


class OCRProvider(Protocol):
    provider: str

    def recognize(self, image_path: Path) -> list[OCRRegion]: ...


class DisabledOCRProvider:
    provider = "disabled"

    def recognize(self, image_path: Path) -> list[OCRRegion]:
        return []


class PaddleOCRProvider:
    """延迟加载 PaddleOCR，避免启动 Demo 时加载重模型。"""

    provider = "paddleocr"

    def __init__(self) -> None:
        self._engine: Any = None
        self._major_version = int(version("paddleocr").split(".", 1)[0])

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "OCR_PROVIDER=paddleocr 需要可选依赖：pip install paddleocr paddlepaddle"
                ) from exc
            if self._major_version >= 3:
                enable_mkldnn = os.getenv("OCR_ENABLE_MKLDNN", "false").casefold() in {
                    "1", "true", "yes", "on",
                }
                detection_model = os.getenv("OCR_DETECTION_MODEL", "PP-OCRv5_mobile_det")
                recognition_model = os.getenv("OCR_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec")
                self._engine = PaddleOCR(
                    text_detection_model_name=detection_model,
                    text_recognition_model_name=recognition_model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=os.getenv(
                        "OCR_USE_TEXTLINE_ORIENTATION", "true"
                    ).casefold() in {"1", "true", "yes", "on"},
                    enable_mkldnn=enable_mkldnn,
                    cpu_threads=max(1, min(int(os.getenv("OCR_CPU_THREADS", "4")), 32)),
                )
            else:
                self._engine = PaddleOCR(
                    use_angle_cls=True, lang=os.getenv("OCR_LANGUAGE", "ch"), show_log=False
                )
        return self._engine

    def release(self) -> None:
        """释放 Paddle Predictor；低虚拟内存机器不能与 SigLIP2 同时常驻。"""
        self._engine = None
        gc.collect()

    def recognize(self, image_path: Path) -> list[OCRRegion]:
        isolated = os.getenv("OCR_ISOLATED_PROCESS", "true").casefold() in {
            "1", "true", "yes", "on",
        }
        if self._major_version >= 3 and isolated:
            return self._recognize_isolated(image_path)
        return self._recognize_in_process(image_path)

    def _recognize_in_process(self, image_path: Path) -> list[OCRRegion]:
        engine = self._load()
        if self._major_version >= 3:
            return self._parse_v3_results(engine.predict(str(image_path)))
        raw = engine.ocr(str(image_path), cls=True)
        regions: list[OCRRegion] = []
        for page in raw or []:
            for item in page or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                points, value = item[0], item[1]
                text = str(value[0]).strip() if value else ""
                confidence = float(value[1]) if value and len(value) > 1 else 0.0
                if not text:
                    continue
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                regions.append(OCRRegion(text, confidence, (min(xs), min(ys), max(xs), max(ys))))
        return regions

    def _recognize_isolated(self, image_path: Path) -> list[OCRRegion]:
        cache_root = Path(os.getenv("PADDLE_PDX_CACHE_HOME", tempfile.gettempdir()))
        temporary_root = cache_root / "deskpilot_ocr_temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".json", dir=temporary_root, delete=False
        ) as temporary:
            output_path = Path(temporary.name)
        environment = dict(os.environ)
        environment["OCR_ISOLATED_PROCESS"] = "false"
        environment["PYTHONUTF8"] = "1"
        timeout = max(30, min(int(os.getenv("OCR_TIMEOUT_SECONDS", "180")), 600))
        try:
            result = self._run_worker(image_path, output_path, environment, timeout)
            retried = False
            if result.returncode != 0 and self._is_native_crash(result.returncode):
                # Paddle 在部分 Windows CPU 上使用“方向分类 + 多线程”时会发生
                # 0xC0000005 原生访问冲突。子进程已隔离，可用保守配置安全重试一次。
                output_path.write_text("", encoding="utf-8")
                fallback_environment = dict(environment)
                fallback_environment["OCR_CPU_THREADS"] = "1"
                fallback_environment["OCR_USE_TEXTLINE_ORIENTATION"] = "false"
                result = self._run_worker(
                    image_path, output_path, fallback_environment, timeout
                )
                retried = True
            if result.returncode != 0:
                detail = self._process_detail(result)[-3000:]
                retry_note = "；保守配置重试仍失败" if retried else ""
                raise RuntimeError(
                    f"PaddleOCR 子进程失败（exit={result.returncode}{retry_note}）：{detail}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            return [OCRRegion(
                text=str(item.get("text", "")), confidence=float(item.get("confidence", 0.0)),
                bbox=tuple(item["bbox"]) if item.get("bbox") else None,
            ) for item in payload]
        finally:
            output_path.unlink(missing_ok=True)

    @staticmethod
    def _run_worker(
        image_path: Path, output_path: Path, environment: dict[str, str], timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable, "-m", "deskpilot.multimodal.providers.ocr_worker",
                str(image_path), str(output_path),
            ],
            cwd=str(Path(__file__).resolve().parents[3]), env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )

    @staticmethod
    def _is_native_crash(returncode: int) -> bool:
        """识别 Windows NTSTATUS 或 POSIX signal，不重试普通 Python 失败。"""
        return returncode > 0x7FFFFFFF or returncode < 0

    @staticmethod
    def _process_detail(result: subprocess.CompletedProcess[str]) -> str:
        detail = result.stderr or result.stdout or "OCR 子进程异常退出"
        # Steps 中不展示 ANSI 颜色控制符，也不把 ccache 探测噪声当成根因。
        detail = re.sub(r"\x1b\[[0-9;]*m", "", detail)
        lines = [
            line for line in detail.splitlines()
            if "No ccache found" not in line
            and "extension_utils.py" not in line
            and "warnings.warn(warning_message)" not in line
            and "Could not find files for the given pattern" not in line
        ]
        return "\n".join(lines).strip() or "OCR 子进程异常退出"

    @classmethod
    def _parse_v3_results(cls, results: Any) -> list[OCRRegion]:
        regions: list[OCRRegion] = []
        for result in results or []:
            payload = cls._result_payload(result)
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            raw_boxes = payload.get("rec_boxes")
            if raw_boxes is None:
                raw_boxes = payload.get("rec_polys")
            boxes = list(raw_boxes) if raw_boxes is not None else []
            for index, raw_text in enumerate(texts):
                text = str(raw_text).strip()
                if not text:
                    continue
                confidence = float(scores[index]) if index < len(scores) else 0.0
                bbox = cls._bbox(boxes[index]) if index < len(boxes) else None
                regions.append(OCRRegion(text, confidence, bbox))
        return regions

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result.get("res", result) if isinstance(result.get("res", result), dict) else result
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        if isinstance(value, dict):
            nested = value.get("res", value)
            return nested if isinstance(nested, dict) else value
        try:
            return dict(result)
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _bbox(value: Any) -> tuple[float, float, float, float] | None:
        try:
            points = value.tolist() if hasattr(value, "tolist") else value
            if len(points) == 4 and not isinstance(points[0], (list, tuple)):
                return tuple(float(item) for item in points)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return min(xs), min(ys), max(xs), max(ys)
        except (TypeError, ValueError, IndexError):
            return None


def build_ocr_provider(name: str | None = None) -> OCRProvider:
    selected = (name or os.getenv("OCR_PROVIDER", "disabled")).strip().casefold()
    if selected in {"paddle", "paddleocr"}:
        return PaddleOCRProvider()
    return DisabledOCRProvider()
