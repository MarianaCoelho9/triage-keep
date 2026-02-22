import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from starlette.websockets import WebSocketDisconnect
from voice_agent.core import (
    TTSChunkEvent,
    STTOutputEvent,
    AgentChunkEvent,
    ExtractionEvent,
    ExtractionStatusEvent,
)

@pytest.mark.asyncio
async def test_websocket_flow():
    """
    Verify WebSocket endpoint accepts bytes, passes them to pipeline, 
    and returns bytes from TTSChunkEvents.
    """
    
    # Mock the pipeline to act as a simple echo:
    # Input audio -> [TTSChunkEvent(audio=input_audio)]
    # Mock the pipeline to act as a simple echo
    async def mock_pipeline(audio_stream):
        async for audio_chunk in audio_stream:
            yield TTSChunkEvent(audio=audio_chunk)

    # Mock get_services to return dummy objects and avoid loading models
    with patch("routes.http.get_services") as mock_get_services, \
         patch("routes.ws.voice_agent_pipeline", side_effect=mock_pipeline):
        
        mock_get_services.return_value = {
            "stt": MagicMock(),
            "llm": MagicMock(),
            "tts": MagicMock()
        }
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as websocket:
                # Send bytes
                test_data = b"hello_audio"
                websocket.send_bytes(test_data)
                
                # Receive bytes
                response = websocket.receive_bytes()
                assert response == test_data


@pytest.mark.asyncio
async def test_websocket_commit_event_sequence():
    """
    Verify deterministic event sequence for v2 endpoint after COMMIT.
    """

    async def mock_pipeline(audio_stream):
        async for message in audio_stream:
            if isinstance(message, str) and message == "COMMIT":
                yield STTOutputEvent(text="patient transcript")
                yield AgentChunkEvent(text="next question?")
                yield ExtractionStatusEvent(status="running", revision=1)
                yield ExtractionEvent(data={"severity_risk": "medium"})
                yield TTSChunkEvent(audio=b"audio_reply")
                return

    with patch("routes.http.get_services") as mock_get_services, patch(
        "routes.ws.voice_agent_pipeline", side_effect=mock_pipeline
    ):
        mock_get_services.return_value = {
            "stt": MagicMock(),
            "llm": MagicMock(),
            "tts": MagicMock(),
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as websocket:
                websocket.send_bytes(b"audio_chunk")
                websocket.send_text("PING")
                websocket.send_text("COMMIT")

                user_msg = websocket.receive_json()
                agent_msg = websocket.receive_json()
                extraction_status_msg = websocket.receive_json()
                extraction_msg = websocket.receive_json()
                audio_msg = websocket.receive_bytes()

                assert user_msg == {"type": "user", "text": "patient transcript"}
                assert agent_msg == {"type": "agent", "text": "next question?"}
                assert extraction_status_msg == {
                    "type": "extraction_status",
                    "status": "running",
                    "revision": 1,
                }
                assert extraction_msg == {
                    "type": "extraction",
                    "data": {"severity_risk": "medium"},
                }
                assert audio_msg == b"audio_reply"


def test_websocket_v2_unavailable():
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/audio/v2"):
                pass
