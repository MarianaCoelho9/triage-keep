"""
Voice Agent Package for Medical Triage.

This package provides a complete voice-based triage system with:
- Speech-to-Text (STT) using MedASR
- Language Model processing using MedGemma/Gemini
- Text-to-Speech (TTS) using Kokoro
- Parallel interaction and extraction workflows

Main entry point:
    voice_agent_pipeline: Complete audio processing pipeline

Core components:
    - core: Event types and base abstractions
    - services: STT, TTS, and LLM service implementations
    - agents: Triage agent with parallel workflows
    - pipelines: Pipeline orchestration
    - config: Service configuration and factory
"""

# Main pipeline (primary public API)
from .pipelines import voice_agent_pipeline

# Core event types (for type hints and event handling)
from .core import (
    VoiceAgentEvent,
    STTChunkEvent,
    STTOutputEvent,
    AgentChunkEvent,
    TTSChunkEvent,
    ExtractionEvent,
    ExtractionStatusEvent,
)

# Configuration (for service initialization)
from .config import get_services, get_stt_service, get_tts_service, get_llm_service

__all__ = [
    # Main pipeline
    "voice_agent_pipeline",
    # Events
    "VoiceAgentEvent",
    "STTChunkEvent",
    "STTOutputEvent",
    "AgentChunkEvent",
    "TTSChunkEvent",
    "ExtractionEvent",
    "ExtractionStatusEvent",
    # Config
    "get_services",
    "get_stt_service",
    "get_tts_service",
    "get_llm_service",
]

__version__ = "2.0.0"
