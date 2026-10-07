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
    rows = provider._recognize_in_process(image_path)
    output_path.write_text(
        json.dumps([
            {"text": row.text, "confidence": row.confidence, "bbox": row.bbox}
            for row in rows
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
