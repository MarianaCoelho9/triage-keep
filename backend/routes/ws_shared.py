import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from voice_agent.core import VoiceAgentEvent
from voice_agent.core.logging_utils import (
    clear_log_context,
    log_event,
    pop_session_metrics_summary,
    set_session_id,
    set_turn_id,
)


def _is_websocket_closed_error(err: BaseException) -> bool:
    message = str(err)
    known_markers = (
        "Unexpected ASGI message 'websocket.send'",
        "disconnect message has been received",
        "WebSocket is not connected",
    )
    return any(marker in message for marker in known_markers)


async def websocket_audio_stream(
    websocket: WebSocket,
    component: str,
) -> AsyncIterator[bytes | str]:
    """Yield audio bytes and COMMIT control messages from websocket."""
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                yield message["bytes"]
            elif "text" in message:
                text_data = message["text"]
                if text_data == "COMMIT":
                    log_event(component=component, event="ws_commit_received")
                    yield "COMMIT"
                elif text_data in {"PING", "PONG"}:
                    continue
    except WebSocketDisconnect:
        log_event(
            component=component,
            event="ws_disconnected",
            details={"source": "receive", "reason": "websocket_disconnect"},
        )
        return
    except RuntimeError as err:
        if "disconnect message has been received" in str(err):
            log_event(
                component=component,
                event="ws_disconnected",
                details={"source": "receive", "reason": "disconnect_message_received"},
            )
            return
        log_event(
            component=component,
            event="ws_receive_failed",
            level="ERROR",
            details={"error": str(err)},
        )
        return
    except Exception as err:  # pragma: no cover - defensive parity with prior behavior
        log_event(
            component=component,
            event="ws_receive_failed",
            level="ERROR",
            details={"error": str(err)},
        )
        return


async def run_websocket_session(
    *,
    websocket: WebSocket,
    component: str,
    session_prefix: str,
    pipeline_factory: Callable[[AsyncIterator[bytes | str]], AsyncIterator[VoiceAgentEvent]],
    send_event: Callable[[WebSocket, VoiceAgentEvent], Awaitable[None]],
) -> None:
    """Run one websocket session with shared lifecycle and cleanup."""
    await websocket.accept()
    session_id = f"{session_prefix}-{uuid.uuid4().hex[:12]}"
    set_session_id(session_id)
    set_turn_id(None)
    log_event(component=component, event="ws_connected")

    try:
        input_stream = websocket_audio_stream(websocket, component)
        output_stream = pipeline_factory(input_stream)
        async for event in output_stream:
            try:
                await send_event(websocket, event)
            except WebSocketDisconnect:
                log_event(
                    component=component,
                    event="ws_disconnected",
                    details={"source": "send", "reason": "websocket_disconnect"},
                )
                break
            except RuntimeError as err:
                if _is_websocket_closed_error(err):
                    log_event(
                        component=component,
                        event="ws_disconnected",
                        details={
                            "source": "send",
                            "reason": "send_on_closed_socket",
                            "error": str(err),
                        },
                    )
                    break
                raise
    except Exception as err:
        log_event(
            component=component,
            event="ws_pipeline_failed",
            level="ERROR",
            details={"error": str(err)},
        )
    finally:
        metrics_summary: dict[str, Any] = pop_session_metrics_summary(session_id)
        if metrics_summary["stages"]:
            log_event(
                component=component,
                event="session_metrics_summary",
                details=metrics_summary,
            )
        try:
            await websocket.close()
        except Exception:
            pass
        clear_log_context()
