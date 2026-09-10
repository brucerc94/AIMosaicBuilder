"""
Vision LLM Engine for AI Mosaic Builder.

Wraps llama-cpp-python for multimodal (vision) inference.
Follows the same patterns as AIStoryWriter's engine/chat.py:
  - Runtime introspection of llama-cpp-python (never assumes a version)
  - GPU detection via nvidia-smi
  - thread-safe, one model at a time
  - inference runs off the UI thread

Vision mechanism auto-selection:
  1. mmproj_path param available  → pass mmproj to Llama()
  2. chat_handler + llama_chat_format available → build handler and pass
  3. clip_model_path available (older builds) → try that
  4. None → only text analysis possible (fallback)

Image encoding:
  Images are sent as base64 data-URI in the user message content,
  following the OpenAI vision message format that llama-cpp-python's
  create_chat_completion() supports.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from engine import llama_features
from engine.models import ImageAnalysis

logger = logging.getLogger("vision_llm")

_llama_available = False
try:
    from llama_cpp import Llama  # type: ignore
    _llama_available = True
except ImportError:
    pass


# ─── GPU detection (mirrors AIStoryWriter engine/chat.py) ─────────────────────

_NO_TENSOR_CORE_MARKERS = ("GTX 16", "GTX 10", "GTX 9", "GTX 7")


def _detect_gpu_info() -> Optional[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        name, mem_total, mem_free, compute_cap = parts[:4]
        return {
            "name": name,
            "mem_total": int(float(mem_total)),
            "mem_free": int(float(mem_free)),
            "compute_cap": compute_cap,
        }
    except Exception:
        return None


def _needs_mmq_fallback(gpu_name: str) -> bool:
    upper = gpu_name.upper()
    return any(marker in upper for marker in _NO_TENSOR_CORE_MARKERS)


# ─── Analysis prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a photo quality evaluation assistant. Your job is to analyze photographs and determine their suitability for inclusion in a portrait mosaic.

You must respond with ONLY a valid JSON object — no markdown, no explanation, no code fences.
"""

_USER_PROMPT = """Analyze this photograph and respond with ONLY this JSON object (fill in the values, no other text):

{
  "has_person": <true/false>,
  "person_count": <integer 0-99>,
  "main_subject_is_person": <true/false>,
  "person_visibility": <float 0.0-1.0>,
  "face_visible": <true/false>,
  "body_visible": <true/false>,
  "occluded": <true/false — is the person significantly blocked by other objects?>,
  "blur": <float 0.0-1.0 — 0=sharp, 1=very blurry>,
  "composition": <float 0.0-1.0 — photographic composition quality>,
  "image_quality": <float 0.0-1.0 — overall technical quality>,
  "subject_quality": <float 0.0-1.0 — quality of the person as a subject>,
  "mosaic_value": <float 0.0-1.0 — how good this photo would be in a portrait mosaic>,
  "reject": <true/false — should this photo be excluded from the mosaic?>,
  "reject_reason": <"" or one of: "no_person", "blurry", "low_quality", "occluded", "person_too_small">,
  "notes": <"" or one short sentence with the most important observation>
}"""


# ─── JSON extraction ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from LLM output that may contain
    surrounding text, markdown fences, or reasoning.
    """
    if not text:
        return None

    # 1. Strip common markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip()

    # 2. Try direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 3. Extract first {...} block
    m = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 4. Try to find the last complete JSON block
    matches = list(re.finditer(r"\{.*?\}", clean, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    return None


def _parse_analysis(text: str, model_name: str = "") -> ImageAnalysis:
    """Convert raw LLM text into a validated ImageAnalysis dataclass."""
    data = _extract_json(text)
    if data is None:
        logger.warning(f"[vision_llm] Could not extract JSON from response: {text[:200]}")
        return ImageAnalysis.error_result(
            reason=f"JSON parse failed. Raw: {text[:300]}",
            model=model_name,
        )

    try:
        analysis = ImageAnalysis.from_dict(data)
        analysis.raw_response = text
        analysis.model_used = model_name
        return analysis
    except Exception as e:
        logger.warning(f"[vision_llm] Could not build ImageAnalysis from dict: {e}")
        return ImageAnalysis.error_result(
            reason=f"Dict→dataclass failed: {e}. Data: {data}",
            model=model_name,
        )


# ─── VisionLLMEngine ──────────────────────────────────────────────────────────

class VisionLLMEngine:
    """
    Singleton-style multimodal LLM engine.

    Usage:
        engine = VisionLLMEngine()
        engine.load_model(model_path="...", mmproj_path="...", ...)
        analysis = engine.analyze_image("/path/to/photo.jpg")
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._current_model_path: str = ""
        self._current_mmproj_path: str = ""
        self._lock = threading.Lock()
        self._capabilities: Optional[dict] = None
        self._model_basename: str = ""

    # ── Capability detection ───────────────────────────────────────────────────

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
    def model_name(self) -> str:
        return self._model_basename

    def log_capabilities(self) -> None:
        caps = self.capabilities
        logger.info(f"[vision_llm] llama-cpp-python available:  {caps['llama_cpp_available']}")
        logger.info(f"[vision_llm] Version:                     {caps['version']}")
        logger.info(f"[vision_llm] Llama class OK:              {caps['llama_class_ok']}")
        logger.info(f"[vision_llm] mmproj_path param:           {caps['mmproj_path_param']}")
        logger.info(f"[vision_llm] chat_handler param:          {caps['chat_handler_param']}")
        logger.info(f"[vision_llm] clip_model_path param:       {caps['clip_model_path_param']}")
        logger.info(f"[vision_llm] chat_format module:          {caps['chat_format_module']}")
        logger.info(f"[vision_llm] LLaVA handler class:         {caps['llava_handler_class']}")
        logger.info(f"[vision_llm] Vision mechanism:            {caps['vision_mechanism']}")
        logger.info(f"[vision_llm] n_gpu_layers param:          {caps['n_gpu_layers_param']}")
        logger.info(f"[vision_llm] flash_attn param:            {caps['flash_attn_param']}")

    # ── Model loading ──────────────────────────────────────────────────────────

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
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with self._lock:
            # Reload only if something changed
            if (self._current_model_path == model_path
                    and self._current_mmproj_path == mmproj_path
                    and self._model is not None):
                logger.info("[vision_llm] Model already loaded — skipping reload")
                return

            if progress_callback:
                progress_callback("Unloading previous model…")
            self._unload()

            if progress_callback:
                progress_callback(f"Loading {Path(model_path).name}…")

            self.log_capabilities()

            # GPU / flash-attn detection (same pattern as AIStoryWriter)
            extra_kwargs: dict = {}
            force_mmq = False
            flash_attn = True

            if n_gpu_layers != 0 and self.capabilities["n_gpu_layers_param"]:
                gpu_info = _detect_gpu_info()
                if gpu_info:
                    logger.info(
                        f"[vision_llm] GPU: {gpu_info['name']} "
                        f"| VRAM total: {gpu_info['mem_total']} MiB "
                        f"| free: {gpu_info['mem_free']} MiB"
                    )
                    fallback = _needs_mmq_fallback(gpu_info["name"])
                    force_mmq = fallback
                    flash_attn = not fallback
                    if force_mmq:
                        os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
                        logger.info("[vision_llm] GGML_CUDA_FORCE_MMQ=1 (no real Tensor Cores)")
                else:
                    logger.info("[vision_llm] nvidia-smi unavailable — assuming no GPU")
                    n_gpu_layers = 0

            if llama_features.supports("flash_attn"):
                extra_kwargs["flash_attn"] = flash_attn

            if n_threads_batch > 0 and llama_features.supports("n_threads_batch"):
                extra_kwargs["n_threads_batch"] = n_threads_batch

            # ── Vision mechanism ───────────────────────────────────────────────
            mechanism = self.capabilities["vision_mechanism"]
            logger.info(f"[vision_llm] Using vision mechanism: {mechanism}")

            if mechanism == "mmproj_path" and mmproj_path:
                if not Path(mmproj_path).is_file():
                    logger.warning(f"[vision_llm] mmproj file not found: {mmproj_path} — will try text-only")
                else:
                    extra_kwargs["mmproj_path"] = mmproj_path
                    logger.info(f"[vision_llm] mmproj: {mmproj_path}")

            elif mechanism == "chat_handler" and mmproj_path:
                try:
                    import llama_cpp.llama_chat_format as lcf  # type: ignore
                    handler_cls = getattr(lcf, self.capabilities["llava_handler_class"])
                    handler = handler_cls(clip_model_path=mmproj_path)
                    extra_kwargs["chat_handler"] = handler
                    logger.info(
                        f"[vision_llm] chat_handler: "
                        f"{self.capabilities['llava_handler_class']} "
                        f"clip_model_path={mmproj_path}"
                    )
                except Exception as e:
                    logger.warning(f"[vision_llm] Failed to build chat_handler: {e}")

            elif mechanism == "clip_model_path" and mmproj_path:
                extra_kwargs["clip_model_path"] = mmproj_path
                logger.info(f"[vision_llm] clip_model_path: {mmproj_path}")

            if not mmproj_path and mechanism != "none":
                logger.warning(
                    "[vision_llm] No mmproj/clip file provided — "
                    "model will run text-only (no real image analysis)"
                )

            # ── Construct Llama ────────────────────────────────────────────────
            try:
                self._model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    verbose=False,
                    **extra_kwargs,
                )
            except TypeError as e:
                # If an extra kwarg is rejected, retry without it
                logger.warning(f"[vision_llm] TypeError with extra_kwargs {e} — retrying without them")
                self._model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    verbose=False,
                )

            self._current_model_path = model_path
            self._current_mmproj_path = mmproj_path
            self._model_basename = Path(model_path).name

            logger.info(f"[vision_llm] Model loaded: {self._model_basename}")
            if progress_callback:
                progress_callback("Model ready.")

    def unload(self) -> None:
        with self._lock:
            self._unload()

    def _unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._current_model_path = ""
            self._current_mmproj_path = ""
            self._model_basename = ""

    # ── Image analysis ─────────────────────────────────────────────────────────

    def analyze_image(
        self,
        image_path: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
    ) -> ImageAnalysis:
        """
        Analyze a single image and return a structured ImageAnalysis.
        Must be called from a background thread — never from the UI thread.
        Raises RuntimeError if no model is loaded.
        """
        if not _llama_available:
            raise RuntimeError("llama-cpp-python not installed")
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Load and encode image
        try:
            img_b64 = _encode_image_b64(image_path)
        except Exception as e:
            logger.error(f"[vision_llm] Cannot encode image {image_path}: {e}")
            return ImageAnalysis.error_result(
                reason=f"Cannot read image: {e}",
                model=self._model_basename,
            )

        has_vision = bool(self._current_mmproj_path)
        messages = _build_messages(img_b64, has_vision=has_vision)

        with self._lock:
            try:
                response = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,
                    stream=False,
                )
                text: str = response["choices"][0]["message"]["content"] or ""
                logger.debug(f"[vision_llm] Raw response ({len(text)} chars): {text[:300]}")
                return _parse_analysis(text, model_name=self._model_basename)
            except Exception as e:
                logger.error(f"[vision_llm] Inference error for {image_path}: {e}")
                return ImageAnalysis.error_result(
                    reason=f"Inference error: {e}",
                    model=self._model_basename,
                )

    def analyze_image_raw(
        self,
        image_path: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
    ) -> tuple[str, ImageAnalysis]:
        """
        Like analyze_image() but also returns the raw LLM text.
        Useful for the Phase 2 minimal test.
        """
        if self._model is None:
            raise RuntimeError("No model loaded.")

        try:
            img_b64 = _encode_image_b64(image_path)
        except Exception as e:
            err = ImageAnalysis.error_result(f"Cannot read image: {e}", self._model_basename)
            return str(e), err

        has_vision = bool(self._current_mmproj_path)
        messages = _build_messages(img_b64, has_vision=has_vision)

        with self._lock:
            try:
                response = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,
                    stream=False,
                )
                text: str = response["choices"][0]["message"]["content"] or ""
                analysis = _parse_analysis(text, model_name=self._model_basename)
                return text, analysis
            except Exception as e:
                err = ImageAnalysis.error_result(f"Inference error: {e}", self._model_basename)
                return str(e), err


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _encode_image_b64(path: str) -> str:
    """Return a base64 data-URI for the image at path."""
    ext = Path(path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _build_messages(img_b64: str, has_vision: bool = True) -> list[dict]:
    """
    Build the message list for create_chat_completion().

    If has_vision=True (mmproj loaded), use the OpenAI vision format:
      content = [{"type": "image_url", ...}, {"type": "text", ...}]

    If has_vision=False (text-only fallback), omit the image.
    """
    if has_vision:
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": img_b64},
            },
            {
                "type": "text",
                "text": _USER_PROMPT,
            },
        ]
    else:
        # Text-only fallback — model has no image access; returns generic values
        user_content = (
            _USER_PROMPT
            + "\n\n(Note: No image provided. Use placeholder values and set reject=true.)"
        )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ─── Module-level singleton ───────────────────────────────────────────────────

_engine: Optional[VisionLLMEngine] = None


def get_vision_engine() -> VisionLLMEngine:
    global _engine
    if _engine is None:
        _engine = VisionLLMEngine()
    return _engine
