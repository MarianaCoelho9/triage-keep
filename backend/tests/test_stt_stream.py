import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch
from voice_agent.services.stt import stt_stream
from voice_agent.core import STTChunkEvent, STTOutputEvent

# Mock generator for audio stream
async def mock_audio_generator(chunks=3, result_bytes=b"0"*32000):
    for _ in range(chunks):
        yield result_bytes
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_stt_stream_calls_transcribe():
    """
    Verify that stt_stream accumulates bytes and calls transcribe.
    We mock MedASRLocalSTT to avoid loading real model.
    """
    chunk_size = 32000 # ~1 second at 16kHz 16-bit mono
    mock_chunk = b'\x00' * chunk_size
    
    # Mock STT service
    mock_stt_service = MagicMock()
    mock_stt_service.transcribe.return_value = "hello world"
    
    # Create a stream with 3 chunks
    audio_gen = mock_audio_generator(chunks=3, result_bytes=mock_chunk)
    
    # Run stream
    events = []
    async for event in stt_stream(audio_gen, mock_stt_service, chunk_buffer_seconds=0.5):
        events.append(event)
            
    # Assertions
    assert len(events) > 0
    # Should have called transcribe at least once
    assert mock_stt_service.transcribe.called
    
    # Check event types
    assert isinstance(events[-1], STTOutputEvent)
    assert events[-1].text == "hello world"

@pytest.mark.asyncio
async def test_stt_stream_chunk_events():
    """Verify chunk events are emitted."""
    chunk_size = 32000 
    mock_chunk = b'\x00' * chunk_size
    
    # Mock STT service
    mock_stt_service = MagicMock()
    mock_stt_service.transcribe.return_value = "partial text"
    
    audio_gen = mock_audio_generator(chunks=5, result_bytes=mock_chunk)
    
    events = []
    async for event in stt_stream(audio_gen, mock_stt_service, chunk_buffer_seconds=0.5):
        events.append(event)
        
    # Check that we got ChunkEvents
    chunk_events = [e for e in events if isinstance(e, STTChunkEvent)]
    assert len(chunk_events) > 0
    assert chunk_events[0].text == "partial text"
