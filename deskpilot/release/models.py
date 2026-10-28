from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from .profile import profile_home


def install_optional_model(kind: str, home: Path | None = None) -> str:
    """Explicit opt-in prefetch. Model weights are never bundled into the base installer."""
    home = home or profile_home()
    if kind == "siglip2":
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            raise RuntimeError("此安装包未包含 SigLIP2 运行依赖，请使用 Full 构建。")
        if shutil.disk_usage(home.parent).free < 2 * 1024**3:
            raise RuntimeError("下载 SigLIP2 前至少保留 2 GiB 可用空间。")
        os.environ.setdefault("HF_HOME", str(home / "cache" / "huggingface"))
        from huggingface_hub import snapshot_download

        model_id = "google/siglip2-base-patch16-224"
        snapshot_download(repo_id=model_id)
        return model_id
    if kind == "ocr":
        if importlib.util.find_spec("paddleocr") is None:
            raise RuntimeError("此安装包未包含 PaddleOCR 运行依赖，请使用 Full 构建。")
        if shutil.disk_usage(home.parent).free < 1024**3:
            raise RuntimeError("下载 PaddleOCR 前至少保留 1 GiB 可用空间。")
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(home / "cache" / "paddlex"))
        from paddleocr import PaddleOCR

        PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            cpu_threads=1,
        )
        return "PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec"
    raise ValueError(f"Unsupported optional model: {kind}")
