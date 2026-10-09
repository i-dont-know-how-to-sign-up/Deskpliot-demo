from __future__ import annotations

import json
import sys
from pathlib import Path

from ...core.config import load_config
from .ocr import PaddleOCRProvider


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: ocr_worker IMAGE_PATH OUTPUT_JSON")
    load_config()
    image_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    provider = PaddleOCRProvider()
    if image_path.suffix.casefold() == ".json":
        payload = json.loads(image_path.read_text(encoding="utf-8"))
        paths = [Path(value) for value in payload.get("image_paths", [])]
        if not paths:
            raise ValueError("OCR 批处理输入不能为空")
        groups = [provider._recognize_in_process(path) for path in paths]
        result = {"results": [[
            {"text": row.text, "confidence": row.confidence, "bbox": row.bbox}
            for row in rows
        ] for rows in groups]}
    else:
        rows = provider._recognize_in_process(image_path)
        result = [
            {"text": row.text, "confidence": row.confidence, "bbox": row.bbox}
            for row in rows
        ]
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
