"""
Feature detection for the installed llama-cpp-python build.

Mirrors AIStoryWriter's engine/llama_features.py and extends it with
vision/multimodal capability detection:

  - Llama.__init__ parameter introspection (n_ctx, n_gpu_layers, etc.)
  - chat_handler / mmproj_path support for vision
  - llama_chat_format module presence
  - create_chat_completion multimodal parameters

Never hard-codes assumptions about "the current version".
Always introspects the actually-installed build at runtime.
"""

from __future__ import annotations

import inspect
import logging
from typing import Optional

logger = logging.getLogger("llama_features")

_supported_init_params: Optional[set] = None
_supported_chat_params: Optional[set] = None


# ─── Llama.__init__ parameters ────────────────────────────────────────────────

def _llama_init_params() -> set:
    global _supported_init_params
    if _supported_init_params is not None:
        return _supported_init_params
    try:
        from llama_cpp import Llama
        _supported_init_params = set(inspect.signature(Llama.__init__).parameters.keys())
    except Exception as e:
        logger.warning(f"[llama_features] Could not introspect Llama.__init__ signature: {e}")
        _supported_init_params = set()
    return _supported_init_params


def supports(param_name: str) -> bool:
    """True if the installed llama-cpp-python's Llama() accepts this kwarg."""
    return param_name in _llama_init_params()


# ─── create_chat_completion parameters ────────────────────────────────────────

def _chat_completion_params() -> set:
    global _supported_chat_params
    if _supported_chat_params is not None:
        return _supported_chat_params
    try:
        from llama_cpp import Llama
        _supported_chat_params = set(
            inspect.signature(Llama.create_chat_completion).parameters.keys()
        )
    except Exception as e:
        logger.warning(f"[llama_features] Could not introspect create_chat_completion: {e}")
        _supported_chat_params = set()
    return _supported_chat_params


def supports_chat_completion_param(param_name: str) -> bool:
    """True if create_chat_completion() accepts this kwarg (or accepts **kwargs)."""
    try:
        from llama_cpp import Llama
        params = inspect.signature(Llama.create_chat_completion).parameters.values()
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD or p.name == param_name
            for p in params
        )
    except Exception:
        return False


# ─── Vision / multimodal capability detection ─────────────────────────────────

def detect_vision_capabilities() -> dict:
    """
    Introspect the installed llama-cpp-python for all vision/multimodal
    mechanisms. Returns a dict describing what is actually available.

    Keys:
        llama_cpp_available     : bool - package importable at all
        version                 : str  - llama_cpp.__version__ or "unknown"
        llama_class_ok          : bool - Llama class importable
        chat_handler_param      : bool - Llama.__init__ accepts chat_handler
        mmproj_path_param       : bool - Llama.__init__ accepts mmproj_path
        clip_model_path_param   : bool - Llama.__init__ accepts clip_model_path (older)
        chat_format_module      : bool - llama_cpp.llama_chat_format importable
        llava_handler_class     : str  - class name found in chat_format, or ""
        vision_mechanism        : str  - "mmproj_path" | "chat_handler" | "none"
        n_batch_param           : bool
        n_ubatch_param          : bool
        flash_attn_param        : bool
        n_threads_batch_param   : bool
        n_gpu_layers_param      : bool
    """
    caps: dict = {
        "llama_cpp_available": False,
        "version": "unknown",
        "llama_class_ok": False,
        "chat_handler_param": False,
        "mmproj_path_param": False,
        "clip_model_path_param": False,
        "chat_format_module": False,
        "llava_handler_class": "",
        "vision_mechanism": "none",
        "n_batch_param": False,
        "n_ubatch_param": False,
        "flash_attn_param": False,
        "n_threads_batch_param": False,
        "n_gpu_layers_param": False,
    }

    try:
        import llama_cpp
        caps["llama_cpp_available"] = True
        caps["version"] = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        return caps

    try:
        from llama_cpp import Llama
        caps["llama_class_ok"] = True
        init_params = set(inspect.signature(Llama.__init__).parameters.keys())

        caps["chat_handler_param"] = "chat_handler" in init_params
        caps["mmproj_path_param"] = "mmproj_path" in init_params
        caps["clip_model_path_param"] = "clip_model_path" in init_params
        caps["n_batch_param"] = "n_batch" in init_params
        caps["n_ubatch_param"] = "n_ubatch" in init_params
        caps["flash_attn_param"] = "flash_attn" in init_params
        caps["n_threads_batch_param"] = "n_threads_batch" in init_params
        caps["n_gpu_layers_param"] = "n_gpu_layers" in init_params
    except Exception as e:
        logger.warning(f"[llama_features] Llama introspection failed: {e}")
        return caps

    # Try importing the chat_format module for handler classes
    try:
        import llama_cpp.llama_chat_format as lcf  # type: ignore
        caps["chat_format_module"] = True

        # Look for LLaVA-style handlers
        llava_candidates = [
            "Llava15ChatHandler",
            "LlavaR34ChatHandler",
            "ObsidianChatHandler",
            "MoondreamChatHandler",
            "NanoLlavaChatHandler",
            "Llava16ChatHandler",
        ]
        for candidate in llava_candidates:
            if hasattr(lcf, candidate):
                caps["llava_handler_class"] = candidate
                break

        # If no specific handler found, check for generic clip handler
        if not caps["llava_handler_class"]:
            for name in dir(lcf):
                if "llava" in name.lower() or "vision" in name.lower() or "clip" in name.lower():
                    caps["llava_handler_class"] = name
                    break
    except ImportError:
        pass

    # Determine preferred vision mechanism
    if caps["mmproj_path_param"]:
        caps["vision_mechanism"] = "mmproj_path"
    elif caps["chat_handler_param"] and caps["chat_format_module"] and caps["llava_handler_class"]:
        caps["vision_mechanism"] = "chat_handler"
    elif caps["clip_model_path_param"]:
        caps["vision_mechanism"] = "clip_model_path"

    return caps


def reset_cache() -> None:
    """For tests, or if llama_cpp is (re)installed during a running process."""
    global _supported_init_params, _supported_chat_params
    _supported_init_params = None
    _supported_chat_params = None


def supported_advanced_params() -> dict:
    """Snapshot of optional kwargs for diagnostics/logging."""
    candidates = [
        "flash_attn", "n_batch", "n_ubatch", "n_gpu_layers",
        "n_threads_batch", "offload_kqv", "split_mode",
        "main_gpu", "tensor_split", "chat_handler", "mmproj_path",
        "clip_model_path",
    ]
    return {name: supports(name) for name in candidates}
