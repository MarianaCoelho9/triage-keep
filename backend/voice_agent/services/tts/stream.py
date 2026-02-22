"""TTS streaming logic for voice agent pipeline."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator
from ...core import VoiceAgentEvent, AgentChunkEvent, TTSChunkEvent
from ...core.logging_utils import log_event, log_latency_event, set_turn_id

_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")


async def _run_tts_synthesize(tts_service, text: str) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_TTS_EXECUTOR, tts_service.synthesize, text)


async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
    tts_service,
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transforms AgentChunkEvents into TTSChunkEvents using the TTSService.
    
    :param event_stream: Incoming event stream
    :param tts_service: TTS service instance with synthesize() method
    """

    async for event in event_stream:
        yield event
        if isinstance(event, AgentChunkEvent):
            chunk_text = event.text
            if not chunk_text:
                continue
                
            # Run synthesis in a separate thread to avoid blocking the event loop
            # TTSService.synthesize is synchronous but does heavy I/O and processing
            tts_started_at = time.perf_counter()
            try:
                set_turn_id(event.turn_id)
                log_event(
                    component="tts_stream",
                    event="tts_started",
                    turn_id=event.turn_id,
                    details={"text_chars": len(chunk_text)},
                )
                audio_bytes = await _run_tts_synthesize(tts_service, chunk_text)
                
                if audio_bytes:
                    log_event(
                        component="tts_stream",
                        event="tts_completed",
                        turn_id=event.turn_id,
                        details={"audio_bytes": len(audio_bytes)},
                    )
                    log_latency_event(
                        component="tts_stream",
                        event="tts_latency",
                        stage="tts",
                        duration_s=time.perf_counter() - tts_started_at,
                        status="completed",
                        turn_id=event.turn_id,
                    )
                    yield TTSChunkEvent(audio=audio_bytes)
            except Exception as err:
                log_event(
                    component="tts_stream",
                    event="tts_failed",
                    level="ERROR",
                    turn_id=event.turn_id,
                    details={"error": str(err)},
                )
                log_latency_event(
                    component="tts_stream",
                    event="tts_latency",
                    stage="tts",
                    duration_s=time.perf_counter() - tts_started_at,
                    status="failed",
                    turn_id=event.turn_id,
                    level="ERROR",
                )
