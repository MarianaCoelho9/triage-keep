"""Triage voice agent pipeline orchestration."""
from typing import AsyncIterator
from ..core import VoiceAgentEvent, AudioStream
from ..services.stt import stt_stream
from ..services.tts import tts_stream
from ..agents.triage import agent_stream
from ..config import get_services


async def voice_agent_pipeline(
    audio_stream: AudioStream
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Full Voice Agent Pipeline for Triage:
    Audio Stream -> STT -> Agent (Parallel Interaction/Extraction) -> TTS -> Output Events
    
    This pipeline:
    1. Transcribes audio to text (STT)
    2. Processes transcripts through triage agent with parallel workflows:
       - Interaction: Generates conversational responses
       - Extraction: Extracts structured clinical data
    3. Synthesizes agent responses to speech (TTS)
    4. Yields all events for consumption
    
    :param audio_stream: Iterator yielding raw PCM audio bytes or control strings
    :yields: VoiceAgentEvent instances (STT, Agent, Extraction, TTS events)
    """
    services = get_services()
    
    # 1. STT Stream: Audio Bytes -> STT Events (STTChunkEvent and STTOutputEvent)
    stream_1 = stt_stream(
        audio_stream,
        services["stt"],
        emit_partial_transcripts=False,
    )
    
    # 2. Agent Stream: STT Events -> Agent Events (AgentChunkEvent and ExtractionEvent)
    stream_2 = agent_stream(stream_1)
    
    # 3. TTS Stream: Agent Events -> TTS Events (TTSChunkEvent)
    # The final stream yields all events flowing through it
    stream_3 = tts_stream(stream_2, services["tts"])
    
    async for event in stream_3:
        yield event
