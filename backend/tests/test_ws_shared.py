import pytest

from routes.ws_shared import _is_websocket_closed_error, run_websocket_session
from voice_agent.core import AgentChunkEvent


class DummyWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            RuntimeError("Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."),
            True,
        ),
        (RuntimeError("disconnect message has been received"), True),
        (RuntimeError("WebSocket is not connected"), True),
        (RuntimeError("some unrelated runtime error"), False),
    ],
)
def test_is_websocket_closed_error(error: RuntimeError, expected: bool) -> None:
    assert _is_websocket_closed_error(error) is expected


@pytest.mark.asyncio
async def test_run_websocket_session_graceful_on_send_after_close() -> None:
    websocket = DummyWebSocket()

    async def pipeline_factory(_audio_stream):
        yield AgentChunkEvent(text="hello")

    async def send_event(_websocket, _event):
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed."
        )

    await run_websocket_session(
        websocket=websocket,
        component="websocket_v2",
        session_prefix="ws-v2",
        pipeline_factory=pipeline_factory,
        send_event=send_event,
    )

    assert websocket.accepted is True
    assert websocket.closed is True
