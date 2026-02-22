"""Voice agent services (STT, TTS, LLM)."""
from .stt import BaseSTTModel, MedASRService, stt_stream, MedASRMLXService
from .tts import BaseTTSModel, TTSService, tts_stream
from .llm import (
    BaseLLMModel,
    GenerationOptions,
    LLMGenerationError,
    LLMContextBudgetExceededError,
    LLMDecodeError,
    MedGemmaLlamaCppService,
)

__all__ = [
    # STT
    "BaseSTTModel",
    "MedASRService",
    "MedASRMLXService",
    "stt_stream",
    # TTS
    "BaseTTSModel",
    "TTSService",
    "tts_stream",
    # LLM
    "BaseLLMModel",
    "GenerationOptions",
    "LLMGenerationError",
    "LLMContextBudgetExceededError",
    "LLMDecodeError",
    "MedGemmaLlamaCppService",
]
