"""STT streaming logic for voice agent pipeline."""
import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import AsyncIterator
from ...core import VoiceAgentEvent, STTOutputEvent, STTChunkEvent, AudioStream
from ...core.logging_utils import log_event, log_latency_event, set_turn_id

_STT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-infer")


async def _run_stt_transcribe(stt_service, audio_data: np.ndarray) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_STT_EXECUTOR, stt_service.transcribe, audio_data)


async def stt_stream(
    audio_stream: AudioStream,
    stt_service,
    sample_rate: int = 16000,
    chunk_buffer_seconds: float = 2.0,
    emit_partial_transcripts: bool = True,
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Stream audio chunks, buffer them, and run transcription periodically.
    
    This is a naive streaming implementation for a non-streaming local model:
    It accumulates audio and runs the full transcription on the growing buffer 
    (or chunks) periodically.
    
    :param audio_stream: Iterator yielding raw PCM bytes (assuming 16kHz, mono, 16-bit unless specified)
    :param stt_service: STT service instance with transcribe() method
    :param sample_rate: Sample rate of input audio
    :param chunk_buffer_seconds: How often (in seconds of audio) to attempt a partial transcription
    """
    
    # Validation / Converter constants
    # Assuming input is 16-bit PCM (2 bytes per sample)
    bytes_per_sample = 2 
    bytes_per_second = sample_rate * bytes_per_sample
    buffer_threshold_bytes = int(chunk_buffer_seconds * bytes_per_second)
    
    audio_buffer = bytearray()
    
    log_event(
        component="stt_stream",
        event="stt_stream_started",
        details={
            "sample_rate": sample_rate,
            "chunk_buffer_seconds": chunk_buffer_seconds,
            "emit_partial_transcripts": emit_partial_transcripts,
        },
    )
    
    chunk_count = 0
    finalized_turn_count = 0
    
    async for chunk in audio_stream:
        if isinstance(chunk, str) and chunk == "COMMIT":
            # User signaled end of turn
            if len(audio_buffer) > 0:
                data_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                data_float32 = data_int16.astype(np.float32) / 32768.0
                stt_started_at = time.perf_counter()
                try:
                    text = await _run_stt_transcribe(stt_service, data_float32)
                    stt_status = "completed"
                    stt_level = "INFO"
                except Exception:
                    text = ""
                    stt_status = "failed"
                    stt_level = "ERROR"
                    raise
                finally:
                    log_latency_event(
                        component="stt_stream",
                        event="stt_latency",
                        stage="stt",
                        duration_s=time.perf_counter() - stt_started_at,
                        status=stt_status,
                        turn_id=finalized_turn_count + 1,
                        level=stt_level,
                        details={"source": "commit"},
                    )
                cleaned_text = text.strip()
                if not cleaned_text:
                    log_event(
                        component="stt_stream",
                        event="stt_empty_transcript_ignored",
                        details={"source": "commit"},
                    )
                    audio_buffer.clear()
                    continue
                finalized_turn_count += 1
                transcript_hash = (
                    hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()[:12]
                )
                set_turn_id(finalized_turn_count)
                log_event(
                    component="stt_stream",
                    event="stt_turn_finalized",
                    turn_id=finalized_turn_count,
                    details={
                        "transcript_chars": len(cleaned_text),
                        "transcript_sha256_12": transcript_hash,
                        "source": "commit",
                    },
                )
                yield STTOutputEvent(text=cleaned_text, turn_id=finalized_turn_count)
                audio_buffer.clear()
            else:
                # User committed but buffer empty? Maybe just short silence.
                # Could optionally yield empty event or just ignore.
                pass
            continue

        # Valid audio bytes
        audio_buffer.extend(chunk)
        chunk_count += 1
        
        # Naive strategy: Transcribe every X bytes (simulating intermediate results)
        if emit_partial_transcripts and len(audio_buffer) >= buffer_threshold_bytes:
            # Convert buffer to numpy float32
            # 1. From bytes to int16
            # We use bytes() to create a copy, avoiding BufferError when resizing audio_buffer later
            data_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            # 2. To float32 normalized
            data_float32 = data_int16.astype(np.float32) / 32768.0
            
            # Run inference (blocking call, maybe push to thread if blocking event loop)
            # using to_thread to keep async loop responsive
            text = await _run_stt_transcribe(stt_service, data_float32)
            
            if text:
                # Emit partial result
                yield STTChunkEvent(text=text, is_final=False)
            
            # Decide if we clear buffer? 
            # If we clear, we lose context. If we keep, it gets slower.
            # For now, let's keep it to allow context to improve, 
            # BUT this will get very slow for long calls.
            # Real streaming implementations usually use a rolling window or specific model support.
            # Let's simple "consume" the buffer if we found a pause? Hard to detect VAD here easily.
            # For this MVP: Keep buffer growing, but maybe limit max size?
            
            # ALTERNATIVE: Just process the *new* audio? No, ASR needs context.
            pass

    # End of stream (final transcription)
    # With COMMIT logic, this might only be reached on disconnect
    if len(audio_buffer) > 0:
        data_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
        data_float32 = data_int16.astype(np.float32) / 32768.0
        stt_started_at = time.perf_counter()
        try:
            text = await _run_stt_transcribe(stt_service, data_float32)
            stt_status = "completed"
            stt_level = "INFO"
        except Exception:
            text = ""
            stt_status = "failed"
            stt_level = "ERROR"
            raise
        finally:
            log_latency_event(
                component="stt_stream",
                event="stt_latency",
                stage="stt",
                duration_s=time.perf_counter() - stt_started_at,
                status=stt_status,
                turn_id=finalized_turn_count + 1,
                level=stt_level,
                details={"source": "stream_end"},
            )
        cleaned_text = text.strip()
        if not cleaned_text:
            log_event(
                component="stt_stream",
                event="stt_empty_transcript_ignored",
                details={"source": "stream_end"},
            )
            log_event(
                component="stt_stream",
                event="stt_stream_ended",
                details={
                    "chunks_received": chunk_count,
                    "finalized_turns": finalized_turn_count,
                },
            )
            return
        finalized_turn_count += 1
        transcript_hash = (
            hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()[:12]
        )
        set_turn_id(finalized_turn_count)
        log_event(
            component="stt_stream",
            event="stt_turn_finalized",
            turn_id=finalized_turn_count,
            details={
                "transcript_chars": len(cleaned_text),
                "transcript_sha256_12": transcript_hash,
                "source": "stream_end",
            },
        )
        yield STTOutputEvent(text=cleaned_text, turn_id=finalized_turn_count)
    
    log_event(
        component="stt_stream",
        event="stt_stream_ended",
        details={
            "chunks_received": chunk_count,
            "finalized_turns": finalized_turn_count,
        },
    )
