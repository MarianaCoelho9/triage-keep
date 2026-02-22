from fastapi import APIRouter, Depends, WebSocket

from routes.http import get_ai_services
from voice_agent import voice_agent_pipeline
from voice_agent.core import (
    AgentChunkEvent,
    ExtractionEvent,
    ExtractionStatusEvent,
    ReportEvent,
    ReportStatusEvent,
    STTOutputEvent,
    TTSChunkEvent,
    VoiceAgentEvent,
)

from .ws_shared import run_websocket_session

router = APIRouter()


async def _send_v2_event(websocket: WebSocket, event: VoiceAgentEvent) -> None:
    if isinstance(event, TTSChunkEvent):
        await websocket.send_bytes(event.audio)
    elif isinstance(event, STTOutputEvent):
        payload = {"type": "user", "text": event.text}
        await websocket.send_json(payload)
    elif isinstance(event, AgentChunkEvent):
        payload = {"type": "agent", "text": event.text}
        await websocket.send_json(payload)
    elif isinstance(event, ExtractionEvent):
        payload = {"type": "extraction", "data": event.data}
        await websocket.send_json(payload)
    elif isinstance(event, ExtractionStatusEvent):
        payload = {
            "type": "extraction_status",
            "status": event.status,
            "revision": event.revision,
        }
        if event.is_final:
            payload["is_final"] = True
        await websocket.send_json(payload)
    elif isinstance(event, ReportStatusEvent):
        payload = {
            "type": "report_status",
            "status": event.status,
        }
        await websocket.send_json(payload)
    elif isinstance(event, ReportEvent):
        payload = {
            "type": "report",
            "success": event.success,
            "data": event.data,
            "error": event.error,
        }
        await websocket.send_json(payload)


@router.websocket("/ws/audio")
async def websocket_endpoint(
    websocket: WebSocket,
    services: dict = Depends(get_ai_services),
) -> None:
    _ = services
    await run_websocket_session(
        websocket=websocket,
        component="websocket_v2",
        session_prefix="ws-v2",
        pipeline_factory=voice_agent_pipeline,
        send_event=_send_v2_event,
    )
