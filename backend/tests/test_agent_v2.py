import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend modules are found
import os
# sys.path is handled by pytest running from backend dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from voice_agent.core import STTOutputEvent, AgentChunkEvent, ExtractionEvent
from voice_agent.agents.triage import agent_stream

@pytest.mark.asyncio
async def test_agent_v2_parallel_execution():
    print("\n[TEST] Starting Agent V2 Mock Test")
    # Mock services
    mock_llm = MagicMock()
    
    with patch(
        "voice_agent.agents.triage.agent.get_services",
        return_value={
            "llm": mock_llm,
            "llm_interaction": mock_llm,
            "llm_extraction": mock_llm,
        },
    ), \
        patch(
            "voice_agent.agents.triage.agent.run_triage_interaction",
            return_value="What is your symptom?",
        ) as mock_interact, \
        patch(
            "voice_agent.agents.triage.agent.run_triage_extraction_incremental",
            return_value={
                "main_complaint": "test_symptom",
                "additional_symptoms": [],
                "medical_history": "",
                "severity_risk": "low",
            },
        ) as mock_extract, patch.dict(
            os.environ,
            {"MEDGEMMA_EXTRACTION_DEBOUNCE_S": "0.01"},
        ):
        
        # Input Stream Generator
        async def input_stream():
            yield STTOutputEvent(text="I have a headache")
        
        # Run Agent
        events = []
        async for event in agent_stream(input_stream()):
            events.append(event)
        
        # Verify Interaction Event
        chunk_events = [e for e in events if isinstance(e, AgentChunkEvent)]
        assert len(chunk_events) > 0
        assert chunk_events[0].text == "What is your symptom?"
        print("[TEST] Verified Interaction Event")

        # Verify Extraction Event
        extract_events = [e for e in events if isinstance(e, ExtractionEvent)]
        assert len(extract_events) > 0
        assert extract_events[0].data["main_complaint"] == "test_symptom"
        print("[TEST] Verified Extraction Event")
        
        # Verify calls were made
        mock_interact.assert_called_once()
        mock_extract.assert_called_once()
        print("[TEST] Verified Parallel Calls")
