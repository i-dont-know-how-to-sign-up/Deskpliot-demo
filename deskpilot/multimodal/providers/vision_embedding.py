from __future__ import annotations

import math
import os
import gc
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class VisionEmbeddingProvider(Protocol):
    provider: str
    embedding_space: str
    model_id: str
    model_revision: str

    def embed_images(self, paths: list[Path]) -> list[list[float]]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class BasicVisualEmbeddingProvider:
    """无需模型的图像相似度 fallback；仅支持图搜图，不冒充跨模态空间。"""

    provider = "basic"
    embedding_space = "vision/basic-rgb-v1"
    model_id = "deskpilot/basic-rgb"
    model_revision = "1"

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        return [self._embed(path) for path in paths]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("basic 视觉向量不支持文本到图片检索，请启用 IMAGE_EMBEDDING_PROVIDER=siglip2")

    @staticmethod
    def _embed(path: Path) -> list[float]:
        with Image.open(path) as opened:
            ratio = float(opened.width) / max(1.0, float(opened.height))
            image = opened.convert("RGB").resize((32, 32))
            histogram = image.histogram()
        # RGB 各 256 桶压缩成各 16 桶，加入宽高比，适合离线相似图回归。
        vector = [sum(histogram[channel * 256 + start:channel * 256 + start + 16])
                  for channel in range(3) for start in range(0, 256, 16)]
        vector.append(ratio)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [float(value) / norm for value in vector]


class SigLIP2EmbeddingProvider:
    """SigLIP2 统一图文空间；Transformers 与 Torch 仅在首次使用时加载。"""

    provider = "siglip2"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("IMAGE_EMBEDDING_MODEL", "google/siglip2-base-patch16-224")
        self.model_revision = os.getenv("IMAGE_EMBEDDING_REVISION", "main")
        self.embedding_space = f"vision/{self.model_id}@{self.model_revision}"
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "SigLIP2 需要可选依赖：pip install torch transformers safetensors"
                ) from exc
            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.model_revision)
            self._model = AutoModel.from_pretrained(self.model_id, revision=self.model_revision)
            self._model.eval()
        return self._model, self._processor

    def release(self) -> None:
        """释放 Torch 模型，供 OCR 与视觉编码按阶段复用有限内存。"""
        self._model = None
        self._processor = None
        gc.collect()

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        if self._use_isolated_process():
            return self._embed_isolated("images", [str(path.resolve()) for path in paths])
        return self._embed_images_in_process(paths)

    def _embed_images_in_process(self, paths: list[Path]) -> list[list[float]]:
        model, processor = self._load()
        import torch
        images = [Image.open(path).convert("RGB") for path in paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            values = model.get_image_features(**inputs)
        return self._normalize(values)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._use_isolated_process():
            return self._embed_isolated("texts", texts)
        return self._embed_texts_in_process(texts)

    def _embed_texts_in_process(self, texts: list[str]) -> list[list[float]]:
        model, processor = self._load()
        import torch
        inputs = processor(text=texts, padding=True, return_tensors="pt")
        with torch.inference_mode():
            values = model.get_text_features(**inputs)
        return self._normalize(values)

    @staticmethod
    def _use_isolated_process() -> bool:
        return os.getenv("IMAGE_EMBEDDING_ISOLATED_PROCESS", "true").casefold() in {
            "1", "true", "yes", "on",
        }

    def _embed_isolated(self, mode: str, values: list[str]) -> list[list[float]]:
        self._ensure_resource_budget()
        cache_root = Path(os.getenv("HF_HOME", tempfile.gettempdir()))
        temporary_root = cache_root / "deskpilot_embedding_temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        input_path: Path | None = None
        output_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", dir=temporary_root, delete=False, mode="w", encoding="utf-8"
            ) as temporary:
                input_path = Path(temporary.name)
                json.dump({"mode": mode, "values": values}, temporary, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(
                suffix=".json", dir=temporary_root, delete=False
            ) as temporary:
                output_path = Path(temporary.name)
            environment = dict(os.environ)
            environment["IMAGE_EMBEDDING_ISOLATED_PROCESS"] = "false"
            environment["PYTHONUTF8"] = "1"
            timeout = max(
                30, min(int(os.getenv("IMAGE_EMBEDDING_TIMEOUT_SECONDS", "180")), 600)
            )
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "deskpilot.multimodal.providers.vision_embedding_worker",
                    str(input_path), str(output_path),
                ],
                cwd=str(Path(__file__).resolve().parents[3]), env=environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "视觉编码子进程异常退出")[-3000:]
                raise RuntimeError(
                    f"SigLIP2 子进程失败（exit={result.returncode}）：{detail}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            vectors = payload.get("vectors") if isinstance(payload, dict) else None
            if not isinstance(vectors, list) or len(vectors) != len(values):
                raise RuntimeError("SigLIP2 子进程返回的向量数量不正确")
            return vectors
        finally:
            if input_path is not None:
                input_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    @staticmethod
    def _available_commit_bytes() -> int | None:
        """返回 Windows 当前可用提交内存；其他平台或查询失败时不做推断。"""
        if os.name != "nt":
            return None

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
        except (AttributeError, OSError):
            return None
        return int(status.ullAvailPageFile)

    @classmethod
    def _ensure_resource_budget(cls) -> None:
        minimum_gib = max(
            0.0, min(float(os.getenv("IMAGE_EMBEDDING_MIN_AVAILABLE_COMMIT_GB", "5")), 64.0)
        )
        available = cls._available_commit_bytes()
        required = int(minimum_gib * 1024 ** 3)
        if available is not None and required and available < required:
            raise RuntimeError(
                "Windows 可用提交内存不足，跳过 SigLIP2 加载"
                f"（{available / 1024 ** 3:.2f} GiB < {minimum_gib:.2f} GiB）"
            )

    @staticmethod
    def _normalize(values: Any) -> list[list[float]]:
        # Transformers 4.x 通常直接返回 Tensor；5.x 的部分 SigLIP2 实现返回
        # BaseModelOutputWithPooling。统一提取池化特征，避免绑定单一版本。
        if not hasattr(values, "norm"):
            for field in ("image_embeds", "text_embeds", "pooler_output"):
                candidate = getattr(values, field, None)
                if candidate is not None:
                    values = candidate
                    break
        if not hasattr(values, "norm"):
            raise TypeError(f"视觉编码器返回了不支持的特征类型：{type(values).__name__}")
        normalized = values / values.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return normalized.detach().cpu().float().tolist()


def build_vision_embedding_provider(name: str | None = None) -> VisionEmbeddingProvider:
    selected = (name or os.getenv("IMAGE_EMBEDDING_PROVIDER", "basic")).strip().casefold()
    if selected in {"siglip", "siglip2", "local_siglip"}:
        return SigLIP2EmbeddingProvider()
    return BasicVisualEmbeddingProvider()
