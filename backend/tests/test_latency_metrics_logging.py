from contextlib import contextmanager
from io import StringIO
import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from voice_agent.agents.triage import agent_stream
from voice_agent.core import AgentChunkEvent, STTOutputEvent
from voice_agent.core.logging_utils import clear_log_context, set_session_id, set_turn_id
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


def _assert_latency_record(record: dict, expected_component: str, expected_turn_id: int) -> None:
    assert record["component"] == expected_component
    assert record["turn_id"] == expected_turn_id
    assert isinstance(record["details"]["duration_ms"], (int, float))
    assert record["details"]["duration_ms"] >= 0
    assert "status" in record["details"]
    assert "stage" in record["details"]


@pytest.mark.asyncio
async def test_stt_latency_event_schema():
    async def mock_audio_stream():
        yield b"\x00" * 32000
        yield "COMMIT"

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "hello world"

    set_session_id("latency-stt-session")
    set_turn_id(None)

    with _capture_structured_logs() as buffer:
        async for _event in stt_stream(mock_audio_stream(), mock_stt, chunk_buffer_seconds=1.0):
            pass

    parsed = _parse_log_lines(buffer.getvalue())
    latency_logs = [log for log in parsed if log.get("event") == "stt_latency"]
    assert len(latency_logs) == 1
    record = latency_logs[0]
    assert record["session_id"] == "latency-stt-session"
    _assert_latency_record(record, expected_component="stt_stream", expected_turn_id=1)
    assert record["details"]["status"] == "completed"
    assert record["details"]["stage"] == "stt"

    clear_log_context()


@pytest.mark.asyncio
async def test_tts_latency_event_schema():
    async def mock_event_stream():
        yield AgentChunkEvent(text="assistant reply", turn_id=2)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = b"audio"

    set_session_id("latency-tts-session")
    set_turn_id(None)

    with _capture_structured_logs() as buffer:
        async for _event in tts_stream(mock_event_stream(), mock_tts):
            pass

    parsed = _parse_log_lines(buffer.getvalue())
    latency_logs = [log for log in parsed if log.get("event") == "tts_latency"]
    assert len(latency_logs) == 1
    record = latency_logs[0]
    assert record["session_id"] == "latency-tts-session"
    _assert_latency_record(record, expected_component="tts_stream", expected_turn_id=2)
    assert record["details"]["status"] == "completed"
    assert record["details"]["stage"] == "tts"

    clear_log_context()


@pytest.mark.asyncio
async def test_agent_interaction_and_extraction_latency_schema():
    async def mock_event_stream():
        yield STTOutputEvent(text="patient report", turn_id=7)

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Please describe the symptom onset.",
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        return_value={
            "main_complaint": "pain",
            "additional_symptoms": [],
            "medical_history": "",
            "severity_risk": "low",
        },
    ), patch.dict(os.environ, {"MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01"}):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()
        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        set_session_id("latency-agent-session")
        set_turn_id(None)

        with _capture_structured_logs() as buffer:
            async for _event in agent_stream(mock_event_stream()):
                pass

    parsed = _parse_log_lines(buffer.getvalue())
    interaction_logs = [log for log in parsed if log.get("event") == "interaction_latency"]
    extraction_logs = [log for log in parsed if log.get("event") == "extraction_latency"]

    assert len(interaction_logs) == 1
    assert len(extraction_logs) == 1

    interaction_record = interaction_logs[0]
    extraction_record = extraction_logs[0]

    assert interaction_record["session_id"] == "latency-agent-session"
    assert extraction_record["session_id"] == "latency-agent-session"
    _assert_latency_record(interaction_record, expected_component="agent_stream", expected_turn_id=7)
    _assert_latency_record(extraction_record, expected_component="agent_stream", expected_turn_id=7)
    assert interaction_record["details"]["status"] == "completed"
    assert interaction_record["details"]["stage"] == "interaction"
    assert extraction_record["details"]["status"] == "completed"
    assert extraction_record["details"]["stage"] == "extraction"

    clear_log_context()


@pytest.mark.asyncio
async def test_agent_interaction_latency_emergency_bypass_status():
    async def mock_event_stream():
        yield STTOutputEvent(text="I have chest pain and shortness of breath", turn_id=9)

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="This should not be used.",
    ) as mock_interaction, patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        return_value={
            "main_complaint": "chest pain",
            "additional_symptoms": ["shortness of breath"],
            "medical_history": "",
            "severity_risk": "high",
        },
    ), patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01",
            "MEDGEMMA_ENABLE_EMERGENCY_RULE_GATE": "true",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()
        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        set_session_id("latency-bypass-session")
        set_turn_id(None)

        with _capture_structured_logs() as buffer:
            async for _event in agent_stream(mock_event_stream()):
                pass

    parsed = _parse_log_lines(buffer.getvalue())
    interaction_logs = [log for log in parsed if log.get("event") == "interaction_latency"]
    assert len(interaction_logs) == 1
    record = interaction_logs[0]
    assert record["session_id"] == "latency-bypass-session"
    _assert_latency_record(record, expected_component="agent_stream", expected_turn_id=9)
    assert record["details"]["status"] == "bypassed_emergency_rule"
    assert record["details"]["duration_ms"] == 0.0
    mock_interaction.assert_not_called()

    clear_log_context()
