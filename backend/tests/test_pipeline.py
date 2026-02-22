import pytest
from unittest.mock import MagicMock, patch
from voice_agent import voice_agent_pipeline
from voice_agent.core import STTOutputEvent, AgentChunkEvent, TTSChunkEvent

# Mock audio generator
async def mock_audio_stream():
    yield b"some_audio"

@pytest.mark.asyncio
async def test_pipeline_flow():
    """
    Verify that voice_agent_pipeline chains STT -> Agent -> TTS.
    We will mock the internal streams to verify flow control.
    """
    
    # We patch the stream functions themselves to avoid complicated inner logic dependencies
    with patch("voice_agent.pipelines.triage_pipeline.get_services") as mock_get_services, \
         patch("voice_agent.pipelines.triage_pipeline.stt_stream") as mock_stt, \
         patch("voice_agent.pipelines.triage_pipeline.agent_stream") as mock_agent, \
         patch("voice_agent.pipelines.triage_pipeline.tts_stream") as mock_tts:
        mock_get_services.return_value = {
            "stt": MagicMock(),
            "tts": MagicMock(),
        }
        
        # Setup Generators
        # STT yields one output event
        async def stt_gen(audio, stt_service, **_kwargs):
            yield STTOutputEvent(text="Hello")
        
        # Agent consumes STT and yields Agent event
        async def agent_gen(events):
            async for e in events:
                yield e
                if isinstance(e, STTOutputEvent):
                    yield AgentChunkEvent(text="Hi there")
        
        # TTS consumes Agent and yields TTS event
        async def tts_gen(events, tts_service):
            async for e in events:
                yield e
                if isinstance(e, AgentChunkEvent):
                    yield TTSChunkEvent(audio=b"audio_bytes")
                    
        mock_stt.side_effect = stt_gen
        mock_agent.side_effect = agent_gen
        mock_tts.side_effect = tts_gen
        
        # Run pipeline
        events = []
        async for event in voice_agent_pipeline(mock_audio_stream()):
            events.append(event)
            
        # Verify correct chaining
        mock_stt.assert_called_once()
        mock_agent.assert_called_once()
        mock_tts.assert_called_once()
        
        # Verify we got all events cascading
        # 1. STTOutput (from STT)
        # 2. AgentChunk (from Agent)
        # 3. TTSChunk (from TTS)
        
        assert any(isinstance(e, STTOutputEvent) for e in events)
        assert any(isinstance(e, AgentChunkEvent) for e in events)
        assert any(isinstance(e, TTSChunkEvent) for e in events)
