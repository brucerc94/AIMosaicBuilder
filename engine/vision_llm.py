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
import time
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

_SYSTEM_PROMPT = """You are a photo evaluator for a portrait mosaic.
Analyze the supplied photograph and return ONLY one valid JSON object.
Do not explain anything. Use unknown for uncertain visual attributes."""

_USER_PROMPT = """Analyze this photo for a portrait mosaic. Return ONLY JSON.
Required fields:
{
  "has_person": boolean,
  "person_count": integer,
  "main_subject_is_person": boolean,
  "person_visibility": number 0..1,
  "face_visible": boolean,
  "body_visible": boolean,
  "occluded": boolean,
  "blur": number 0..1,
  "composition": number 0..1,
  "image_quality": number 0..1,
  "subject_quality": number 0..1,
  "mosaic_value": number 0..1,
  "visual_tags": {
    "framing": "face_only|head_shoulders|upper_body|half_body|full_body|unknown",
    "orientation": "front|three_quarter_front|side|three_quarter_back|back|unknown",
    "gender_presentation": "male|female|unknown",
    "content_rating": "safe|suggestive|explicit|unknown",
    "pose": "standing|seated|lying|walking|other|unknown",
    "looking_at_camera": boolean
  },
  "reject": boolean,
  "reject_reason": "no_person|blurry|low_quality|occluded|person_too_small|empty"
}
Be conservative. gender_presentation is visual presentation only.
"""


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
        raise RuntimeError(
            f"The installed llama-cpp-python build does not provide {handler_name}. "
            "Gemma-4 requires Gemma4ChatHandler; upgrade llama-cpp-python to a build that supports Gemma-4."
        )
    import inspect
    try:
        params = inspect.signature(handler_cls).parameters
    except Exception:
        params = {}
    if "clip_model_path" in params:
        return handler_cls(clip_model_path=mmproj_path)
    if "mmproj_path" in params:
        return handler_cls(mmproj_path=mmproj_path)
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
    tags = data.get("visual_tags")
    if not isinstance(tags, dict):
        raise ValueError("visual_tags must be an object")
    enum_fields = {
        "framing": {"face_only", "head_shoulders", "upper_body", "half_body", "full_body", "unknown"},
        "orientation": {"front", "three_quarter_front", "side", "three_quarter_back", "back", "unknown"},
        "gender_presentation": {"male", "female", "unknown"},
        "content_rating": {"safe", "suggestive", "explicit", "unknown"},
        "pose": {"standing", "seated", "lying", "walking", "other", "unknown"},
    }
    normalized_tags: dict[str, object] = {}
    for field_name, allowed_values in enum_fields.items():
        value = str(tags.get(field_name, "unknown"))
        normalized_tags[field_name] = value if value in allowed_values else "unknown"
    looking_at_camera = tags.get("looking_at_camera", False)
    if not isinstance(looking_at_camera, bool):
        raise ValueError("visual_tags.looking_at_camera must be boolean")
    normalized_tags["looking_at_camera"] = looking_at_camera
    data["visual_tags"] = normalized_tags
    reason = str(data.get("reject_reason", "") or "")
    allowed = {"", "no_person", "blurry", "low_quality", "occluded", "person_too_small"}
    if reason not in allowed:
        reason = ""
    data["person_count"] = person_count
    data["reject_reason"] = reason
    data["notes"] = str(data.get("notes", "") or "")[:500]
    return data


def _parse_analysis(text: str, model_name: str) -> ImageAnalysis:
    data = _extract_json(text)
    if data is None:
        return ImageAnalysis.error_result("Model response was not valid JSON.", model_name)
    try:
        analysis = ImageAnalysis.from_dict(_strict_analysis(data))
    except Exception as exc:
        return ImageAnalysis.error_result(f"Invalid analysis JSON: {exc}", model_name)
    analysis.raw_response = text
    analysis.model_used = model_name
    return analysis


def _encode_image_b64(path: str, max_dimension: int = 1280) -> str:
    from PIL import Image
    try:
        with Image.open(path) as original:
            image = original.convert("RGB")
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
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
        self._model_ctx = 0
        self._vision_ready = False
        self._capabilities: Optional[dict] = None
        self._lock = threading.RLock()

    @property
    def is_available(self) -> bool:
        return _llama_available

    @property
    def capabilities(self) -> dict:
        if self._capabilities is None:
            self._capabilities = llama_features.detect_vision_capabilities(self._model_path)
        return self._capabilities

    def capabilities_for(self, model_path: str) -> dict:
        return llama_features.detect_vision_capabilities(model_path)

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

    def load_model(self, model_path: str, mmproj_path: str = "", n_ctx: int = 2048, n_gpu_layers: int = 0,
                   n_threads: int = 4, n_threads_batch: int = 0,
                   n_batch: int = 512, n_ubatch: int = 512,
                   progress_callback: Optional[Callable[[str], None]] = None) -> None:
        if not _llama_available:
            raise RuntimeError("llama-cpp-python is not installed.")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not Path(mmproj_path).is_file():
            raise FileNotFoundError("A valid mmproj file is required for image analysis.")
        configured_ctx = int(n_ctx)
        if configured_ctx <= 0:
            raise ValueError("Context must be greater than zero.")
        with self._lock:
            if self._model is not None and self._model_path == model_path and self._mmproj_path == mmproj_path and self._model_ctx == configured_ctx:
                logger.info("[vision] MODEL REUSE | model=%s | mmproj=%s | n_ctx=%d | instance_id=%s", Path(model_path).name, Path(mmproj_path).name, configured_ctx, hex(id(self._model)))
                return
            load_started = time.perf_counter()
            logger.info("[vision] MODEL LOAD START | model=%s | mmproj=%s | n_ctx=%d | n_gpu_layers=%d | n_threads=%d | n_batch=%d | n_ubatch=%d",
                        Path(model_path).name, Path(mmproj_path).name, configured_ctx, n_gpu_layers, n_threads, n_batch, n_ubatch)
            self._unload()
            self._capabilities = llama_features.detect_vision_capabilities(model_path)
            self.log_capabilities()
            caps = self._capabilities
            model_family = caps.get("model_family", "unknown")
            handler_name = caps.get("handler_class", "")
            if model_family == "gemma4" and handler_name != "Gemma4ChatHandler":
                raise RuntimeError("Gemma-4 was detected but Gemma4ChatHandler is unavailable in the installed llama-cpp-python.")
            extra: dict = {}

            # --- CUDA / GPU optimizations (set env vars BEFORE constructing Llama) ---
            mmq_forced = False
            if n_gpu_layers != 0:
                gpu = _detect_gpu_info()
                if gpu:
                    logger.info("[vision] GPU=%s | VRAM=%s/%s MiB | CC=%s", gpu["name"], gpu["mem_free"], gpu["mem_total"], gpu["compute_cap"])
                    if _needs_mmq_fallback(gpu["name"]):
                        os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
                        mmq_forced = True
                        logger.info("[vision] GGML_CUDA_FORCE_MMQ=1 set (Turing/Pascal GPU: no tensor cores)")
                    elif os.environ.get("GGML_CUDA_FORCE_MMQ") == "1":
                        mmq_forced = True
                        logger.info("[vision] GGML_CUDA_FORCE_MMQ=1 already in environment")

            # flash_attn: enable on CUDA, disable when MMQ fallback is active
            if caps["flash_attn_param"]:
                if mmq_forced:
                    extra["flash_attn"] = False
                    logger.info("[vision] flash_attn=False (MMQ fallback active)")
                else:
                    extra["flash_attn"] = True
                    logger.info("[vision] flash_attn=True")

            # n_threads_batch: default to n_threads when not explicitly set
            effective_n_threads_batch = n_threads_batch if n_threads_batch > 0 else n_threads
            if caps["n_threads_batch_param"]:
                extra["n_threads_batch"] = effective_n_threads_batch
                logger.info("[vision] n_threads_batch=%d", effective_n_threads_batch)

            # n_batch / n_ubatch: CRITICAL — controls how many tokens are evaluated per GPU call.
            # Default (512) is fine for text, but multimodal image-token batches can stall if
            # llama.cpp falls back to a smaller value. Explicitly set to match AIStoryWriter.
            if caps["n_batch_param"]:
                extra["n_batch"] = n_batch
                logger.info("[vision] n_batch=%d", n_batch)
            if caps["n_ubatch_param"]:
                extra["n_ubatch"] = n_ubatch
                logger.info("[vision] n_ubatch=%d", n_ubatch)

            # Vision mechanism
            mechanism = caps["vision_mechanism"]
            if mechanism == "chat_handler":
                extra["chat_handler"] = _build_chat_handler(handler_name, mmproj_path)
            elif mechanism == "mmproj_path":
                extra["mmproj_path"] = mmproj_path
            elif mechanism == "clip_model_path":
                extra["clip_model_path"] = mmproj_path
            else:
                raise RuntimeError(f"No usable multimodal vision mechanism for model family {model_family!r}.")

            if progress_callback:
                progress_callback(f"Loading {Path(model_path).name}…")

            kwargs = {
                "model_path": model_path,
                "n_ctx": configured_ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "verbose": True,
                **extra,
            }
            # Log every effective kwarg so we can verify parity with AIStoryWriter
            logger.info(
                "[vision] EFFECTIVE LLAMA KWARGS: n_ctx=%d n_gpu_layers=%d n_threads=%d "
                "n_threads_batch=%s n_batch=%s n_ubatch=%s flash_attn=%s "
                "GGML_CUDA_FORCE_MMQ=%s mechanism=%s handler=%s verbose=%s",
                configured_ctx, n_gpu_layers, n_threads,
                extra.get("n_threads_batch", "not_set"),
                extra.get("n_batch", "not_set"),
                extra.get("n_ubatch", "not_set"),
                extra.get("flash_attn", "not_set"),
                os.environ.get("GGML_CUDA_FORCE_MMQ", "0"),
                mechanism, handler_name, True,
            )
            try:
                self._model = Llama(**kwargs)
            except TypeError as exc:
                raise RuntimeError(f"llama-cpp-python rejected the vision configuration: {exc}") from exc
            self._model_path = model_path
            self._mmproj_path = mmproj_path
            self._model_name = Path(model_path).name
            self._model_ctx = configured_ctx
            self._vision_ready = True
            elapsed = time.perf_counter() - load_started
            logger.info(
                "[vision] MODEL LOAD COMPLETE | model=%s | elapsed=%.2fs | n_ctx=%d | n_gpu_layers=%d | "
                "n_batch=%s | n_ubatch=%s | n_threads=%d | n_threads_batch=%s | flash_attn=%s | "
                "GGML_CUDA_FORCE_MMQ=%s | instance_id=%s | verbose=%s",
                self._model_name, elapsed, configured_ctx, n_gpu_layers,
                extra.get("n_batch", "default"), extra.get("n_ubatch", "default"),
                n_threads, extra.get("n_threads_batch", "default"),
                extra.get("flash_attn", "default"),
                os.environ.get("GGML_CUDA_FORCE_MMQ", "0"),
                hex(id(self._model)), True,
            )
            if progress_callback:
                progress_callback("Vision model ready.")

    def _unload(self) -> None:
        if self._model is not None:
            logger.info("[vision] MODEL UNLOAD INTERNAL | model=%s | instance_id=%s", self._model_name or "unknown", hex(id(self._model)))
        if self._model is not None:
            del self._model
        self._model = None
        self._model_path = ""
        self._mmproj_path = ""
        self._model_name = ""
        self._model_ctx = 0
        self._vision_ready = False

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("[vision] MODEL UNLOAD | model=%s | instance_id=%s", self._model_name or "unknown", hex(id(self._model)))
            self._unload()
            self._capabilities = None

    def analyze_image(self, image_path: str, max_tokens: int = 192, temperature: float = 0.0) -> ImageAnalysis:
        if not self.vision_ready:
            raise RuntimeError("Vision model is not loaded with a valid model and mmproj.")
        image_name = Path(image_path).name
        total_started = time.perf_counter()

        # --- Phase 1: preprocess (resize + base64 encode) ---
        preprocess_started = time.perf_counter()
        img_b64 = _encode_image_b64(image_path)
        preprocess_elapsed = time.perf_counter() - preprocess_started
        logger.info("[vision] PREPROCESS | image=%s | encode=%.3fs | b64_bytes=%d",
                    image_name, preprocess_elapsed, len(img_b64))

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

        logger.info(
            "[vision] INFERENCE START | image=%s | model=%s | n_ctx=%d | max_tokens=%d | "
            "temperature=%.2f | top_p=%.2f | top_k=%d | response_format=%s | instance_id=%s",
            image_name, self._model_name, self._model_ctx, max_tokens, temperature,
            kwargs["top_p"], kwargs["top_k"],
            kwargs.get("response_format", {}).get("type", "none"),
            hex(id(self._model)),
        )

        # --- Phase 2: model call (visual encode + prompt eval + generation) ---
        inference_started = time.perf_counter()
        with self._lock:
            try:
                response = self._model.create_chat_completion(**kwargs)
                text = response["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                inference_elapsed = time.perf_counter() - inference_started
                logger.error("[vision] INFERENCE ERROR | image=%s | inference=%.2fs | total=%.2fs | error=%s",
                             image_name, inference_elapsed, time.perf_counter() - total_started, exc)
                raise RuntimeError(f"Vision inference failed: {exc}") from exc
        inference_elapsed = time.perf_counter() - inference_started

        # --- Phase 3: parse JSON ---
        parse_started = time.perf_counter()
        analysis = _parse_analysis(text, self._model_name)
        parse_elapsed = time.perf_counter() - parse_started

        total_elapsed = time.perf_counter() - total_started
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Compute effective tokens/s for the generation phase.
        # Approximate: assume visual encode ~1.8s, rest is prompt_eval + generation.
        # We report tok/s over the full model_call for comparison with AIStoryWriter logs.
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        gen_tps = (completion_tokens / inference_elapsed) if (completion_tokens and inference_elapsed > 0) else 0.0
        total_tps = (total_tokens / inference_elapsed) if (total_tokens and inference_elapsed > 0) else 0.0

        logger.info(
            "[vision] INFERENCE DONE | image=%s | preprocess=%.3fs | model_call=%.2fs | parse=%.3fs | total=%.2fs | "
            "prompt_tokens=%s | completion_tokens=%s | gen_toks/s=%.1f | total_toks/s=%.1f | response_chars=%d",
            image_name, preprocess_elapsed, inference_elapsed, parse_elapsed, total_elapsed,
            prompt_tokens or "?", completion_tokens or "?",
            gen_tps, total_tps, len(text),
        )
        return analysis

    def analyze_image_raw(self, image_path: str, max_tokens: int = 192, temperature: float = 0.0):
        analysis = self.analyze_image(image_path, max_tokens=max_tokens, temperature=temperature)
        return analysis.raw_response, analysis


_engine: Optional[VisionLLMEngine] = None


def get_vision_engine() -> VisionLLMEngine:
    global _engine
    if _engine is None:
        _engine = VisionLLMEngine()
    return _engine
