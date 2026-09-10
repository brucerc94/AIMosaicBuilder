"""Real multimodal smoke test.

Usage:
    python test_vision.py MODEL.gguf MMPROJ.gguf IMAGE.jpg
"""
from __future__ import annotations

import json
import sys

from engine.vision_llm import get_vision_engine


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("Usage: python test_vision.py MODEL.gguf MMPROJ.gguf IMAGE")
        return 2

    engine = get_vision_engine()
    print(json.dumps(engine.capabilities, indent=2, ensure_ascii=False))
    engine.load_model(
        model_path=argv[1],
        mmproj_path=argv[2],
        n_ctx=4096,
        n_gpu_layers=0,
        n_threads=4,
    )
    raw, analysis = engine.analyze_image_raw(argv[3])
    print("\nRaw response:\n" + raw)
    print("\nParsed analysis:\n" + json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
    return 1 if analysis.reject and analysis.reject_reason == "llm_error" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
