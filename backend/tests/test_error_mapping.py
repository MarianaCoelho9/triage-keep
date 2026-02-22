import asyncio
import os
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from routes.http import get_ai_services
from voice_agent.core import ReportEvent, STTOutputEvent
from voice_agent.core.error_mapping import (
    REPORT_ERROR_CODE_CONTEXT_BUDGET,
    REPORT_ERROR_CODE_DECODE,
    REPORT_ERROR_CODE_GENERIC,
    REPORT_ERROR_CODE_INVALID_JSON,
    build_report_error_payload,
    classify_report_error_code,
)
from voice_agent.agents.triage.agent import agent_stream


@pytest.mark.parametrize(
    ("error_message", "details", "expected_code"),
    [
        ("LLM generation failed", "Requested tokens exceed context window", REPORT_ERROR_CODE_CONTEXT_BUDGET),
        ("LLM generation failed", "context budget exceeded", REPORT_ERROR_CODE_CONTEXT_BUDGET),
        ("LLM generation failed", "llama_decode returned -1", REPORT_ERROR_CODE_DECODE),
        ("LLM generation failed", "decoder failed", REPORT_ERROR_CODE_DECODE),
        ("Failed to parse report JSON", None, REPORT_ERROR_CODE_INVALID_JSON),
        ("Report JSON is not an object", None, REPORT_ERROR_CODE_INVALID_JSON),
        ("Unknown runtime failure", None, REPORT_ERROR_CODE_GENERIC),
    ],
)
def test_classify_report_error_code_matrix(
    error_message: str, details: str | None, expected_code: str
):
    assert classify_report_error_code(error_message, details) == expected_code


def test_build_report_error_payload_omits_empty_optional_fields():
    report = {
        "error": "Unknown runtime failure",
        "details": "",
        "raw_output": "",
    }
    payload = build_report_error_payload(report)
    assert payload == {
        "code": REPORT_ERROR_CODE_GENERIC,
        "message": "Unknown runtime failure",
    }


def test_build_report_error_payload_includes_optional_fields():
    report = {
        "error": "LLM generation failed",
        "details": "Requested tokens exceed context window",
        "raw_output": "raw text",
    }
    payload = build_report_error_payload(report)
    assert payload == {
        "code": REPORT_ERROR_CODE_CONTEXT_BUDGET,
        "message": "LLM generation failed",
        "details": "Requested tokens exceed context window",
        "raw_output": "raw text",
    }


async def _single_stt_event_stream() -> AsyncIterator[STTOutputEvent]:
    yield STTOutputEvent(text="Please end session")


@pytest.mark.asyncio
async def test_http_and_agent_callers_emit_same_error_code():
    synthetic_report = {
        "error": "LLM generation failed",
        "details": "Requested tokens exceed context window",
        "raw_output": "",
    }
    expected_code = build_report_error_payload(synthetic_report)["code"]

    mock_llm = MagicMock()
    mock_services = {
        "llm": mock_llm,
        "llm_interaction": mock_llm,
        "llm_extraction": mock_llm,
        "stt": MagicMock(),
        "tts": MagicMock(),
    }

    app.dependency_overrides[get_ai_services] = lambda: mock_services
    try:
        with patch("routes.http.run_triage_report", return_value=synthetic_report):
            with TestClient(app) as client:
                response = client.post("/report", json={"chat_history": []})
        assert response.status_code == 200
        http_code = response.json()["error"]["code"]
    finally:
        app.dependency_overrides = {}

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="All set. END_SESSION",
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        return_value={
            "main_complaint": "session wrap-up",
            "additional_symptoms": [],
            "medical_history": "",
            "severity_risk": "low",
        },
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_report",
        return_value=synthetic_report,
    ), patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01",
            "MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS": "true",
        },
    ):
        mock_get_services.return_value = {
            "llm": mock_llm,
            "llm_interaction": mock_llm,
            "llm_extraction": mock_llm,
        }
        events_out = []
        async for event in agent_stream(_single_stt_event_stream()):
            events_out.append(event)

    report_events = [event for event in events_out if isinstance(event, ReportEvent)]
    assert len(report_events) == 1
    agent_code = report_events[0].error["code"]

    assert http_code == expected_code
    assert agent_code == expected_code
