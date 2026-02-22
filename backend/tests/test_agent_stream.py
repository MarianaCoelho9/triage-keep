import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from voice_agent.agents.triage import agent_stream
from voice_agent.core import (
    AgentChunkEvent,
    ExtractionEvent,
    ExtractionStatusEvent,
    ReportEvent,
    ReportStatusEvent,
    STTOutputEvent,
)


async def mock_event_generator(events):
    for event in events:
        yield event
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_agent_stream_generates_response():
    input_events = [STTOutputEvent(text="My patient has a fever")]

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Please check their temperature.",
    ) as mock_interact, patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        return_value={
            "main_complaint": "fever",
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

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        agent_events = [e for e in events_out if isinstance(e, AgentChunkEvent)]

        assert len(agent_events) == 1
        assert agent_events[0].text == "Please check their temperature."
        mock_interact.assert_called_once()


@pytest.mark.asyncio
async def test_agent_stream_non_blocking_extraction_order():
    input_events = [STTOutputEvent(text="Goodbye")]

    def slow_extract(*_args, **_kwargs):
        time.sleep(0.1)
        return {
            "main_complaint": "goodbye",
            "additional_symptoms": ["fatigue"],
            "medical_history": "",
            "severity_risk": "medium",
        }

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Take care.",
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        side_effect=slow_extract,
    ), patch.dict(os.environ, {"MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01"}):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        agent_events = [e for e in events_out if isinstance(e, AgentChunkEvent)]
        extraction_events = [e for e in events_out if isinstance(e, ExtractionEvent)]

        assert len(agent_events) == 1
        assert len(extraction_events) == 1
        assert agent_events[0].text == "Take care."
        assert extraction_events[0].data["severity_risk"] == "medium"

        agent_index = events_out.index(agent_events[0])
        extraction_index = events_out.index(extraction_events[0])
        assert agent_index < extraction_index


@pytest.mark.asyncio
async def test_agent_stream_extraction_timeout_emits_status():
    input_events = [STTOutputEvent(text="I have a headache")]

    def very_slow_extract(*_args, **_kwargs):
        time.sleep(0.2)
        return {
            "main_complaint": "headache",
            "additional_symptoms": [],
            "medical_history": "",
            "severity_risk": "low",
        }

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Noted.",
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        side_effect=very_slow_extract,
    ), patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_TIMEOUT_S": "0.05",
            "MEDGEMMA_EXTRACTION_FINAL_TIMEOUT_S": "0.05",
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01",
            "MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS": "true",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        extraction_events = [e for e in events_out if isinstance(e, ExtractionEvent)]
        assert extraction_events == []

        status_events = [e for e in events_out if isinstance(e, ExtractionStatusEvent)]
        assert any(e.status == "timed_out" for e in status_events)


@pytest.mark.asyncio
async def test_agent_stream_debounce_coalesces_rapid_turns():
    input_events = [
        STTOutputEvent(text="First symptom"),
        STTOutputEvent(text="Second symptom"),
    ]

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        side_effect=["Question 1?", "Question 2?"],
    ), patch(
        "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
        return_value={
            "main_complaint": "Second symptom",
            "additional_symptoms": [],
            "medical_history": "",
            "severity_risk": "low",
        },
    ) as mock_incremental, patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.08",
            "MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS": "true",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        assert mock_incremental.call_count == 1
        status_events = [e for e in events_out if isinstance(e, ExtractionStatusEvent)]
        assert any(e.status == "completed" for e in status_events)


@pytest.mark.asyncio
async def test_agent_stream_emits_final_extraction_status_before_report():
    input_events = [STTOutputEvent(text="Please end session")]

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
        return_value={"assessment": {}},
    ), patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01",
            "MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS": "true",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        status_events = [e for e in events_out if isinstance(e, ExtractionStatusEvent)]
        assert any(e.status == "completed" and e.is_final for e in status_events)

        final_extraction_event = next(
            e for e in status_events if e.status == "completed" and e.is_final
        )
        report_status_event = next(
            e for e in events_out if isinstance(e, ReportStatusEvent)
        )
        report_event = next(e for e in events_out if isinstance(e, ReportEvent))

        assert events_out.index(final_extraction_event) < events_out.index(report_status_event)
        assert events_out.index(final_extraction_event) < events_out.index(report_event)


@pytest.mark.asyncio
async def test_agent_stream_emits_report_running_heartbeats_for_long_report():
    input_events = [STTOutputEvent(text="Please end session now")]

    def slow_report(*_args, **_kwargs):
        time.sleep(0.05)
        return {"assessment": {"summary": "ok"}}

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Done. END_SESSION",
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
        side_effect=slow_report,
    ), patch.dict(
        os.environ,
        {
            "MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01",
            "MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS": "true",
            "MEDGEMMA_REPORT_AUTO_TIMEOUT_S": "0.2",
            "MEDGEMMA_REPORT_STATUS_HEARTBEAT_S": "0.01",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        running_status_events = [
            event
            for event in events_out
            if isinstance(event, ReportStatusEvent) and event.status == "running"
        ]
        assert len(running_status_events) >= 2
        assert any(
            isinstance(event, ReportStatusEvent) and event.status == "completed"
            for event in events_out
        )
        assert any(
            isinstance(event, ReportEvent) and event.success
            for event in events_out
        )


@pytest.mark.asyncio
async def test_agent_stream_emergency_gate_bypasses_interaction():
    input_events = [STTOutputEvent(text="I have chest pain and shortness of breath")]

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="This should not be used.",
    ) as mock_interact, patch(
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

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        agent_events = [e for e in events_out if isinstance(e, AgentChunkEvent)]
        assert len(agent_events) == 1
        assert "This may be an emergency." in agent_events[0].text
        mock_interact.assert_not_called()


@pytest.mark.asyncio
async def test_agent_stream_emergency_gate_can_be_disabled():
    input_events = [STTOutputEvent(text="I have chest pain and shortness of breath")]

    with patch("voice_agent.agents.triage.agent.get_services") as mock_get_services, patch(
        "voice_agent.agents.triage.agent.run_triage_interaction",
        return_value="Please tell me when the pain started.",
    ) as mock_interact, patch(
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
            "MEDGEMMA_ENABLE_EMERGENCY_RULE_GATE": "false",
        },
    ):
        mock_llm_interaction = MagicMock()
        mock_llm_extraction = MagicMock()

        mock_get_services.return_value = {
            "llm": mock_llm_interaction,
            "llm_interaction": mock_llm_interaction,
            "llm_extraction": mock_llm_extraction,
        }

        events_out = []
        async for event in agent_stream(mock_event_generator(input_events)):
            events_out.append(event)

        agent_events = [e for e in events_out if isinstance(e, AgentChunkEvent)]
        assert len(agent_events) == 1
        assert agent_events[0].text == "Please tell me when the pain started."
        mock_interact.assert_called_once()
