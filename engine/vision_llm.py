"""Local multimodal inference wrapper based on llama-cpp-python."""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from engine import llama_features
from engine.models import ImageAnalysis

logger = logging.getLogger("vision_llm")

try:
    from llama_cpp import Llama  # type: ignore
    _llama_available = True
except ImportError:
    Llama = None  # type: ignore
    _llama_available = False

_SYSTEM_PROMPT = """You are a photo-quality evaluator for a portrait mosaic.
Analyze the supplied photograph itself. Respond with ONLY one valid JSON object.
Do not use markdown fences or explanations."""

_USER_PROMPT = """Analyze this photograph for a portrait mosaic.
Return JSON with these exact fields:
{
  "has_person": boolean,
  "person_count": integer,
  "main_subject_is_person": boolean,
  "person_visibility": number 0..1,
  "face_visible": boolean,
  "body_visible": boolean,
  "occluded": boolean,
  "blur": number 0..1 where 0 is sharp,
  "composition": number 0..1,
  "image_quality": number 0..1,
  "subject_quality": number 0..1,
  "mosaic_value": number 0..1,
  "reject": boolean,
  "reject_reason": string,
  "notes": string
}
Use only: no_person, blurry, low_quality, occluded, person_too_small, or empty for reject_reason."""


def _detect_gpu_info() -> Optional[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [p.strip() for p in result.stdout.splitlines()[0].split(",")]
        if len(parts) < 4:
            return None
        return {"name": parts[0], "mem_total": int(float(parts[1])), "mem_free": int(float(parts[2])), "compute_cap": parts[3]}
    except Exception:
        return None


def _needs_mmq_fallback(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in ("GTX 16", "GTX 10", "GTX 9", "GTX 7"))


def _find_handler_class(name: str):
    for module_name in ("llama_cpp.llama_multimodal", "llama_cpp.llama_chat_format"):
        try:
            module = __import__(module_name, fromlist=[name])
            handler = getattr(module, name, None)
            if handler is not None:
                return handler
        except Exception:
            continue
    return None


def _build_chat_handler(handler_name: str, mmproj_path: str):
    handler_cls = _find_handler_class(handler_name)
    if handler_cls is None:
        raise RuntimeError(f"Vision handler {handler_name!r} is not available")
    import inspect
    try:
        params = inspect.signature(handler_cls).parameters
    except Exception:
        params = {}
    if "mmproj_path" in params:
        return handler_cls(mmproj_path=mmproj_path)
    if "clip_model_path" in params:
        return handler_cls(clip_model_path=mmproj_path)
    try:
        return handler_cls(mmproj_path)
    except TypeError:
        return handler_cls(clip_model_path=mmproj_path)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    clean = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean):
        try:
            value, _ = decoder.raw_decode(clean[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _strict_analysis(data: dict) -> dict:
    required_bools = ("has_person", "main_subject_is_person", "face_visible", "body_visible", "occluded", "reject")
    required_numbers = ("person_visibility", "blur", "composition", "image_quality", "subject_quality", "mosaic_value")
    for field in required_bools:
        if not isinstance(data.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    try:
        person_count = int(data.get("person_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("person_count must be integer") from exc
    if not 0 <= person_count <= 99:
        raise ValueError("person_count out of range")
    for field in required_numbers:
        value = float(data.get(field))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in 0..1")
        data[field] = value
    reason = str(data.get("reject_reason", "") or "")
    if reason not in {"", "no_person", "blurry", "low_quality", "occluded", "person_too_small"}:
        reason = ""
    data["person_count"] = person_count
    data["reject_reason"] = reason
    data["notes"] = str(data.get("notes", "") or "")[:1000]
    return data


def _parse_analysis(text: str, model_name: str) -> ImageAnalysis:
    data = _extract_json(text)
    if data is None:
        return ImageAnalysis.error_result("Model response was not valid JSON.", model_name)
    try:
        data = _strict_analysis(data)
        analysis = ImageAnalysis.from_dict(data)
    except Exception as exc:
        return ImageAnalysis.error_result(f"Invalid analysis JSON: {exc}", model_name)
    analysis.raw_response = text
    analysis.model_used = model_name
    return analysis


def _encode_image_b64(path: str, max_dimension: int = 1600) -> str:
    from PIL import Image
    try:
        with Image.open(path) as original:
            image = original.convert("RGB")
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90, optimize=True)
        payload = buffer.getvalue()
    except Exception:
        with open(path, "rb") as handle:
            payload = handle.read()
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _build_messages(img_b64: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_b64}},
            {"type": "text", "text": _USER_PROMPT},
        ]},
    ]


class VisionLLMEngine:
    def __init__(self) -> None:
        self._model = None
        self._model_path = ""
        self._mmproj_path = ""
        self._model_name = ""
        self._vision_ready = False
        self._capabilities: Optional[dict] = None
        self._lock = threading.RLock()

    @property
    def is_available(self) -> bool:
        return _llama_available

    @property
    def capabilities(self) -> dict:
        if self._capabilities is None:
            self._capabilities = llama_features.detect_vision_capabilities()
        return self._capabilities

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def vision_ready(self) -> bool:
        return self._vision_ready and self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def log_capabilities(self) -> None:
        for key, value in self.capabilities.items():
            logger.info("[vision] %-22s %s", key, value)

    def load_model(
        self,
        model_path: str,
        mmproj_path: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        n_threads_batch: int = 0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not _llama_available:
            raise RuntimeError("llama-cpp-python is not installed.")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not Path(mmproj_path).is_file():
            raise FileNotFoundError("A valid mmproj file is required for image analysis.")

        with self._lock:
            if self._model is not None and self._model_path == model_path and self._mmproj_path == mmproj_path:
                return
            self._unload()
            self.log_capabilities()
            caps = self.capabilities
            extra: dict = {}
            if caps["flash_attn_param"]:
                extra["flash_attn"] = True
            if n_threads_batch > 0 and caps["n_threads_batch_param"]:
                extra["n_threads_batch"] = n_threads_batch

            if n_gpu_layers != 0:
                gpu = _detect_gpu_info()
                if gpu:
                    logger.info("[vision] GPU=%s | VRAM=%s/%s MiB | CC=%s", gpu["name"], gpu["mem_free"], gpu["mem_total"], gpu["compute_cap"])
                    if _needs_mmq_fallback(gpu["name"]):
                        os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
                        if caps["flash_attn_param"]:
                            extra["flash_attn"] = False

            mechanism = caps["vision_mechanism"]
            if mechanism == "mmproj_path":
                extra["mmproj_path"] = mmproj_path
            elif mechanism == "chat_handler":
                extra["chat_handler"] = _build_chat_handler(caps["handler_class"], mmproj_path)
            elif mechanism == "clip_model_path":
                extra["clip_model_path"] = mmproj_path
            else:
                raise RuntimeError("The installed llama-cpp-python build has no usable multimodal vision mechanism.")

            if progress_callback:
                progress_callback(f"Loading {Path(model_path).name}…")
            kwargs = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "verbose": False,
                **extra,
            }
            try:
                self._model = Llama(**kwargs)
            except TypeError as exc:
                raise RuntimeError(f"llama-cpp-python rejected the vision configuration: {exc}") from exc

            self._model_path = model_path
            self._mmproj_path = mmproj_path
            self._model_name = Path(model_path).name
            self._vision_ready = True
            if progress_callback:
                progress_callback("Vision model ready.")

    def _unload(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._model_path = ""
        self._mmproj_path = ""
        self._model_name = ""
        self._vision_ready = False

    def unload(self) -> None:
        with self._lock:
            self._unload()

    def analyze_image(self, image_path: str, max_tokens: int = 600, temperature: float = 0.1) -> ImageAnalysis:
        if not self.vision_ready:
            raise RuntimeError("Vision model is not loaded with a valid mmproj.")
        img_b64 = _encode_image_b64(image_path)
        kwargs = {
            "messages": _build_messages(img_b64),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "stream": False,
        }
        if llama_features.supports_chat_completion_param("response_format"):
            kwargs["response_format"] = {"type": "json_object"}
        with self._lock:
            try:
                response = self._model.create_chat_completion(**kwargs)
                text = response["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                raise RuntimeError(f"Vision inference failed: {exc}") from exc
        return _parse_analysis(text, self._model_name)

    def analyze_image_raw(self, image_path: str, max_tokens: int = 600, temperature: float = 0.1):
        analysis = self.analyze_image(image_path, max_tokens=max_tokens, temperature=temperature)
        return analysis.raw_response, analysis


_engine: Optional[VisionLLMEngine] = None


def get_vision_engine() -> VisionLLMEngine:
    global _engine
    if _engine is None:
        _engine = VisionLLMEngine()
    return _engine
