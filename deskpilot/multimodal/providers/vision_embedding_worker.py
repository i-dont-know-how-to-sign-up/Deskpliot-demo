from __future__ import annotations

import json
import sys
from pathlib import Path

from ...core.config import load_config
from .vision_embedding import SigLIP2EmbeddingProvider


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: vision_embedding_worker INPUT_JSON OUTPUT_JSON")
    load_config()
    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    mode = str(payload.get("mode", ""))
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("视觉编码输入不能为空")

    provider = SigLIP2EmbeddingProvider()
    if mode == "images":
        vectors = provider._embed_images_in_process([Path(value) for value in values])
    elif mode == "texts":
        vectors = provider._embed_texts_in_process([str(value) for value in values])
    else:
        raise ValueError(f"不支持的视觉编码模式：{mode}")
    output_path.write_text(
        json.dumps({"vectors": vectors}, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
