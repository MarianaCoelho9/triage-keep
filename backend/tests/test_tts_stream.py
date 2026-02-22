import pytest
import asyncio
from unittest.mock import MagicMock, patch
from voice_agent.services.tts import tts_stream
from voice_agent.core import AgentChunkEvent, TTSChunkEvent

# Mock generator for agent events
async def mock_event_generator(events):
    for event in events:
        yield event
        await asyncio.sleep(0.01)

@pytest.mark.asyncio
async def test_tts_stream_synthesizes_audio():
    """
    Verify that tts_stream consumes AgentChunkEvents and produces TTSChunkEvents.
    We mock the TTSService to avoid loading the heavy Kokoro model.
    """
    input_events = [
        AgentChunkEvent(text="Hello"),
        AgentChunkEvent(text="World")
    ]
    
    # Mock the TTS service - tts_stream now takes service as parameter
    mock_tts_service = MagicMock()
    mock_tts_service.synthesize.return_value = b"fake_audio_bytes"
    
    # Run stream
    events_out = []
    async for event in tts_stream(mock_event_generator(input_events), mock_tts_service):
        events_out.append(event)
        
    # Assertions
    tts_events = [e for e in events_out if isinstance(e, TTSChunkEvent)]
    
    # Should have 2 TTS chunks (one for each text chunk)
    assert len(tts_events) == 2
    assert tts_events[0].audio == b"fake_audio_bytes"
    assert tts_events[1].audio == b"fake_audio_bytes"
    
    # Verify synthesize was called correctly
    assert mock_tts_service.synthesize.call_count == 2
    mock_tts_service.synthesize.assert_any_call("Hello")
    mock_tts_service.synthesize.assert_any_call("World")
