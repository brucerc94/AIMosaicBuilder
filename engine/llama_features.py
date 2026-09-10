"""Runtime capability detection for the installed llama-cpp-python build."""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("llama_features")

_init_params: Optional[set[str]] = None
_chat_params: Optional[set[str]] = None


def _get_init_params() -> set[str]:
    global _init_params
    if _init_params is not None:
        return _init_params
    try:
        from llama_cpp import Llama
        _init_params = set(inspect.signature(Llama.__init__).parameters)
    except Exception as exc:
        logger.warning("Could not inspect Llama.__init__: %s", exc)
        _init_params = set()
    return _init_params


def _get_chat_params() -> set[str]:
    global _chat_params
    if _chat_params is not None:
        return _chat_params
    try:
        from llama_cpp import Llama
        _chat_params = set(inspect.signature(Llama.create_chat_completion).parameters)
    except Exception as exc:
        logger.warning("Could not inspect create_chat_completion: %s", exc)
        _chat_params = set()
    return _chat_params


def supports(param_name: str) -> bool:
    params = _get_init_params()
    return param_name in params or "kwargs" in params


def supports_chat_completion_param(param_name: str) -> bool:
    try:
        from llama_cpp import Llama
        params = inspect.signature(Llama.create_chat_completion).parameters.values()
        return any(p.kind is inspect.Parameter.VAR_KEYWORD or p.name == param_name for p in params)
    except Exception:
        return False


def infer_model_family(model_path: str = "") -> str:
    """Infer a model family from its GGUF filename."""
    name = Path(model_path).name.lower().replace("_", "-")
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "gemma-3" in name or "gemma3" in name:
        return "gemma3"
    if "qwen2.5-vl" in name or "qwen-2.5-vl" in name:
        return "qwen25vl"
    if "qwen2-vl" in name or "qwen-2-vl" in name:
        return "qwen2vl"
    if "qwen3-vl" in name or "qwen-3-vl" in name:
        return "qwen3vl"
    return "unknown"


def _handler_candidates_for_family(family: str) -> list[str]:
    return {
        "gemma4": ["Gemma4ChatHandler"],
        "gemma3": ["Gemma3ChatHandler"],
        "qwen25vl": ["Qwen25VLChatHandler"],
        "qwen2vl": ["Qwen2VLChatHandler"],
        "qwen3vl": ["Qwen3VLChatHandler"],
        "unknown": ["GenericMTMDChatHandler", "MTMDChatHandler"],
    }.get(family, ["GenericMTMDChatHandler", "MTMDChatHandler"])


def _find_model_handler(model_path: str) -> tuple[str, str]:
    family = infer_model_family(model_path)
    candidates = _handler_candidates_for_family(family)
    for module_name in ("llama_cpp.llama_multimodal", "llama_cpp.llama_chat_format"):
        try:
            module = __import__(module_name, fromlist=candidates)
        except Exception:
            continue
        for name in candidates:
            if hasattr(module, name):
                return family, name
    return family, ""


def detect_vision_capabilities(model_path: str = "") -> dict:
    caps = {
        "llama_cpp_available": False,
        "version": "unknown",
        "llama_class_ok": False,
        "model_family": infer_model_family(model_path),
        "mmproj_path_param": False,
        "chat_handler_param": False,
        "clip_model_path_param": False,
        "chat_format_module": False,
        "multimodal_module": False,
        "handler_class": "",
        "vision_mechanism": "none",
        "n_batch_param": False,
        "n_ubatch_param": False,
        "flash_attn_param": False,
        "n_threads_batch_param": False,
        "n_gpu_layers_param": False,
        "response_format_param": False,
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
        init_params = _get_init_params()
        chat_params = _get_chat_params()
        caps["mmproj_path_param"] = "mmproj_path" in init_params
        caps["chat_handler_param"] = "chat_handler" in init_params
        caps["clip_model_path_param"] = "clip_model_path" in init_params
        caps["n_batch_param"] = "n_batch" in init_params
        caps["n_ubatch_param"] = "n_ubatch" in init_params
        caps["flash_attn_param"] = "flash_attn" in init_params
        caps["n_threads_batch_param"] = "n_threads_batch" in init_params
        caps["n_gpu_layers_param"] = "n_gpu_layers" in init_params
        caps["response_format_param"] = (
            "response_format" in chat_params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(Llama.create_chat_completion).parameters.values()
            )
        )
    except Exception as exc:
        logger.warning("Llama introspection failed: %s", exc)

    family, handler = _find_model_handler(model_path)
    caps["model_family"] = family
    caps["handler_class"] = handler

    try:
        import llama_cpp.llama_multimodal  # type: ignore  # noqa: F401
        caps["multimodal_module"] = True
    except Exception:
        pass
    try:
        import llama_cpp.llama_chat_format  # type: ignore  # noqa: F401
        caps["chat_format_module"] = True
    except Exception:
        pass

    if caps["chat_handler_param"] and handler:
        caps["vision_mechanism"] = "chat_handler"
    elif caps["mmproj_path_param"]:
        caps["vision_mechanism"] = "mmproj_path"
    elif caps["clip_model_path_param"] and handler:
        caps["vision_mechanism"] = "clip_model_path"

    if family != "unknown" and not handler:
        logger.warning("Model family %s requires its specific vision handler, but none was found.", family)

    return caps


def reset_cache() -> None:
    global _init_params, _chat_params
    _init_params = None
    _chat_params = None


def supported_advanced_params() -> dict[str, bool]:
    names = [
        "flash_attn", "n_batch", "n_ubatch", "n_gpu_layers",
        "n_threads_batch", "offload_kqv", "split_mode", "main_gpu",
        "tensor_split", "chat_handler", "mmproj_path", "clip_model_path",
    ]
    return {name: supports(name) for name in names}
