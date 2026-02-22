from contextlib import contextmanager
from io import StringIO
import json
import logging
from unittest.mock import MagicMock

import pytest

from voice_agent.core import AgentChunkEvent
from voice_agent.core.logging_utils import clear_log_context, log_event, set_session_id, set_turn_id
from voice_agent.services.stt import stt_stream
from voice_agent.services.tts import tts_stream


def _parse_log_lines(raw_output: str) -> list[dict]:
    lines = [line for line in raw_output.splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


@contextmanager
def _capture_structured_logs():
    logger = logging.getLogger("voice_agent.structured")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield buffer
    finally:
        handler.flush()
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_log_event_schema_includes_required_fields():
    set_session_id("session-schema")
    set_turn_id(None)

    with _capture_structured_logs() as buffer:
        log_event(component="test_component", event="test_event")
    parsed = _parse_log_lines(buffer.getvalue())
    assert parsed
    record = parsed[-1]

    assert "ts" in record
    assert record["level"] == "INFO"
    assert record["component"] == "test_component"
    assert record["event"] == "test_event"
    assert record["session_id"] == "session-schema"
    assert "turn_id" in record
    assert "details" in record
    assert isinstance(record["details"], dict)

    clear_log_context()


@pytest.mark.asyncio
async def test_stt_turn_finalized_log_contains_turn_id():
    async def mock_audio_stream():
        yield b"\x00" * 32000
        yield "COMMIT"

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "hello world"

    set_session_id("session-stt")
    set_turn_id(None)

    events = []
    with _capture_structured_logs() as buffer:
        async for event in stt_stream(mock_audio_stream(), mock_stt, chunk_buffer_seconds=1.0):
            events.append(event)

    assert events
    assert events[-1].turn_id == 1

    parsed = _parse_log_lines(buffer.getvalue())
    finalized_logs = [log for log in parsed if log.get("event") == "stt_turn_finalized"]
    assert len(finalized_logs) == 1
    record = finalized_logs[0]

    assert record["component"] == "stt_stream"
    assert record["session_id"] == "session-stt"
    assert record["turn_id"] == 1
    assert record["details"]["transcript_chars"] == 11
    assert len(record["details"]["transcript_sha256_12"]) == 12

    clear_log_context()


@pytest.mark.asyncio
async def test_tts_turn_logs_include_turn_id():
    async def mock_event_stream():
        yield AgentChunkEvent(text="test response", turn_id=2)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = b"audio"

    set_session_id("session-tts")
    set_turn_id(None)

    events = []
    with _capture_structured_logs() as buffer:
        async for event in tts_stream(mock_event_stream(), mock_tts):
            events.append(event)

    assert events

    parsed = _parse_log_lines(buffer.getvalue())
    started_logs = [log for log in parsed if log.get("event") == "tts_started"]
    completed_logs = [log for log in parsed if log.get("event") == "tts_completed"]

    assert len(started_logs) == 1
    assert len(completed_logs) == 1
    assert started_logs[0]["component"] == "tts_stream"
    assert started_logs[0]["session_id"] == "session-tts"
    assert started_logs[0]["turn_id"] == 2
    assert completed_logs[0]["turn_id"] == 2

    clear_log_context()
