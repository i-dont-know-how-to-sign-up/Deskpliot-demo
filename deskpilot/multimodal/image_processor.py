from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - 启动时给出明确依赖提示
    raise RuntimeError("多模态图片能力需要 Pillow：python -m pip install Pillow") from exc


SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


@dataclass(frozen=True)
class ProcessedImage:
    content: bytes
    media_type: str
    extension: str
    width: int
    height: int
    source_size: int


class ImageProcessor:
    """校验真实图片内容并生成方向正确、尺寸受限的安全推理副本。"""

    def __init__(self, max_bytes: int | None = None, max_pixels: int | None = None) -> None:
        self.max_bytes = max_bytes or int(os.getenv("IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))
        self.max_pixels = max_pixels or int(os.getenv("IMAGE_MAX_PIXELS", "40000000"))

    def process_file(self, path: Path) -> ProcessedImage:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")
        return self.process_bytes(path.read_bytes())

    def process_bytes(self, content: bytes) -> ProcessedImage:
        if not content:
            raise ValueError("图片内容为空")
        if len(content) > self.max_bytes:
            raise ValueError(f"图片超过大小限制：{len(content)} > {self.max_bytes} bytes")
        try:
            with Image.open(io.BytesIO(content)) as probe:
                image_format = str(probe.format or "").upper()
                if image_format not in SUPPORTED_FORMATS:
                    raise ValueError(f"不支持的图片格式：{image_format or 'unknown'}")
                if probe.width * probe.height > self.max_pixels:
                    raise ValueError(f"图片像素数超过限制：{probe.width}x{probe.height}")
                probe.verify()
            with Image.open(io.BytesIO(content)) as opened:
                normalized = ImageOps.exif_transpose(opened).convert("RGB")
                output = io.BytesIO()
                # 推理副本统一为高质量 JPEG，避免携带 EXIF 和不可控元数据。
                normalized.save(output, format="JPEG", quality=92, optimize=True)
                return ProcessedImage(
                    output.getvalue(), "image/jpeg", ".jpg", normalized.width, normalized.height, len(content)
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("文件不是有效图片或图片已经损坏") from exc

    @staticmethod
    def thumbnail(content: bytes, max_size: tuple[int, int] = (360, 240)) -> bytes:
        with Image.open(io.BytesIO(content)) as opened:
            image = opened.convert("RGB")
            image.thumbnail(max_size)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            return output.getvalue()
