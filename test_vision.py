"""Real multimodal smoke test for one image.

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

    model_path, mmproj_path, image_path = argv[1:4]
    engine = get_vision_engine()

    capabilities = engine.capabilities_for(model_path)
    print(json.dumps(capabilities, indent=2, ensure_ascii=False))

    engine.load_model(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=4096,
        n_gpu_layers=0,
        n_threads=4,
    )
    raw, analysis = engine.analyze_image_raw(image_path)
    print("\nRaw response:\n" + raw)
    print(
        "\nParsed analysis:\n"
        + json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)
    )
    return 1 if analysis.reject and analysis.reject_reason == "llm_error" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
