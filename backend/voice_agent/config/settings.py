"""Configuration and service factory for voice agent."""
import os
import platform
import threading
from typing import Dict, Any

from ..services.llm import BaseLLMModel, MedGemmaLlamaCppService
from ..services.stt import BaseSTTModel, MedASRService, MedASRMLXService
from ..services.tts import BaseTTSModel, TTSService


# Singleton service instances
_services: Dict[str, Any] = None
_SUPPORTED_LLM_VALUES = ("medgemma_llamacpp", "llamacpp")
_SUPPORTED_INFERENCE_PROFILES = ("local", "space")
_SUPPORTED_STT_BACKENDS = ("auto", "mlx", "torch")
_SUPPORTED_TTS_DEVICES = ("auto", "cuda", "mps", "cpu")


def _build_llm(service_name: str) -> BaseLLMModel:
    normalized = service_name.lower()
    if normalized in _SUPPORTED_LLM_VALUES:
        print("[Services] Using MedGemma (llama.cpp)")
        return MedGemmaLlamaCppService()
    supported = ", ".join(_SUPPORTED_LLM_VALUES)
    raise ValueError(
        f"Unsupported LLM service '{service_name}'. Supported values: {supported}."
    )


def _normalize_choice(env_name: str, supported: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(env_name, default).strip().lower()
    if raw in supported:
        return raw
    supported_values = ", ".join(supported)
    raise ValueError(
        f"Unsupported {env_name} value '{raw}'. Supported values: {supported_values}."
    )


def _resolve_inference_profile() -> str:
    default_profile = "local" if platform.system() == "Darwin" else "space"
    return _normalize_choice(
        "INFERENCE_PROFILE",
        _SUPPORTED_INFERENCE_PROFILES,
        default_profile,
    )


def _resolve_stt_backend(profile: str) -> str:
    backend = _normalize_choice("STT_BACKEND", _SUPPORTED_STT_BACKENDS, "auto")
    if backend != "auto":
        return backend
    if profile == "local" and platform.system() == "Darwin":
        return "mlx"
    return "torch"


def _resolve_tts_device(profile: str) -> str:
    device = _normalize_choice("TTS_DEVICE", _SUPPORTED_TTS_DEVICES, "auto")
    if device != "auto":
        return device
    if profile == "space":
        return "cuda"
    if platform.system() == "Darwin":
        return "mps"
    return "cpu"


def _build_stt(backend_name: str) -> BaseSTTModel:
    if backend_name == "mlx":
        if MedASRMLXService is None:
            raise RuntimeError(
                "STT_BACKEND=mlx is unavailable in this environment (MLX import failed)."
            )
        print("[Services] Using MedASR MLX backend")
        return MedASRMLXService()
    if backend_name == "torch":
        print("[Services] Using MedASR Torch backend")
        return MedASRService()
    raise ValueError(f"Unsupported STT backend: {backend_name}")


def get_services() -> Dict[str, Any]:
    """
    Factory function to get or initialize service instances.
    
    Returns a dictionary with:
        - 'stt': Speech-to-Text service
        - 'tts': Text-to-Speech service
        - 'llm': Language Model service
    """
    global _services
    if _services is None:
        profile = _resolve_inference_profile()
        stt_backend = _resolve_stt_backend(profile)
        tts_device = _resolve_tts_device(profile)

        print(f"[Services] INFERENCE_PROFILE={profile}")

        # Backward-compatible alias: MODEL_PATH can seed MEDGEMMA_GGUF_PATH.
        if not os.environ.get("MEDGEMMA_GGUF_PATH", "").strip():
            model_path = os.environ.get("MODEL_PATH", "").strip()
            if model_path:
                os.environ["MEDGEMMA_GGUF_PATH"] = model_path

        default_llm = os.environ.get("LLM_SERVICE", "medgemma_llamacpp")
        interaction_llm_name = os.environ.get(
            "LLM_INTERACTION_SERVICE", default_llm
        )
        extraction_llm_name = os.environ.get(
            "LLM_EXTRACTION_SERVICE", default_llm
        )

        llm_interaction = _build_llm(interaction_llm_name)
        if extraction_llm_name.lower() == interaction_llm_name.lower():
            llm_extraction = llm_interaction
        else:
            llm_extraction = _build_llm(extraction_llm_name)

        llm_interaction_lock = threading.RLock()
        if llm_extraction is llm_interaction:
            llm_extraction_lock = llm_interaction_lock
        else:
            llm_extraction_lock = threading.RLock()

        try:
            stt_service = _build_stt(stt_backend)
        except Exception:
            explicit_stt = os.environ.get("STT_BACKEND", "auto").strip().lower()
            if stt_backend == "mlx" and explicit_stt == "auto":
                print("[Services] MLX backend unavailable, falling back to torch.")
                stt_service = _build_stt("torch")
            else:
                raise

        _services = {
            "stt": stt_service,
            "llm": llm_interaction,  # Backwards compatibility
            "llm_interaction": llm_interaction,
            "llm_extraction": llm_extraction,
            "llm_interaction_lock": llm_interaction_lock,
            "llm_extraction_lock": llm_extraction_lock,
            "tts": TTSService(device=tts_device),
        }
    return _services


def get_stt_service() -> BaseSTTModel:
    """Get the STT service instance."""
    return get_services()["stt"]


def get_tts_service() -> BaseTTSModel:
    """Get the TTS service instance."""
    return get_services()["tts"]


def get_llm_service() -> BaseLLMModel:
    """Get the LLM service instance."""
    return get_services()["llm"]
