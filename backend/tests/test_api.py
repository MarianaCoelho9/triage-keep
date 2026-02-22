import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from routes.http import get_ai_services
from voice_agent.services.llm.base import LLMContextBudgetExceededError, LLMDecodeError

class TestTriageAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Mock services to avoid loading heavy models
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "Mocked LLM Response"
        
        self.mock_stt = MagicMock()
        self.mock_stt.transcribe.return_value = "Mocked Transcript"
        
        self.mock_tts = MagicMock()
        self.mock_tts.synthesize.return_value = b"mock_audio_bytes"
        
        self.mock_services = {
            "llm": self.mock_llm,
            "llm_interaction": self.mock_llm,
            "llm_extraction": self.mock_llm,
            "stt": self.mock_stt,
            "tts": self.mock_tts
        }
        
        # Override dependency
        app.dependency_overrides[get_ai_services] = lambda: self.mock_services

    def tearDown(self):
        app.dependency_overrides = {}

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "online", "system": "TriageKeep Brain"})

    def test_analyze(self):
        payload = {"user_input": "I have a headache", "chat_history": []}
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("analysis", data)
        self.assertIn("response", data["analysis"])
        self.mock_llm.generate.assert_called_once()

    def test_transcribe(self):
        # Create a dummy file
        files = {'file': ('test.wav', b'dummy content', 'audio/wav')}
        response = self.client.post("/transcribe", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transcript": "Mocked Transcript"})
        self.mock_stt.transcribe.assert_called_once()

    def test_synthesize(self):
        response = self.client.post("/synthesize", params={"text": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"mock_audio_bytes")
        self.assertEqual(response.headers["content-type"], "audio/wav")

    def test_extract_envelope_success(self):
        self.mock_llm.generate.return_value = '<json>{"severity_risk":"low"}</json>'
        payload = {"user_input": "mild cough", "chat_history": []}
        response = self.client.post("/extract", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "data": {"severity_risk": "low"},
                "error": None,
            },
        )

    def test_report_envelope_success(self):
        self.mock_llm.generate.return_value = '<json>{"plan":{"care_advice":"Rest"}}</json>'
        payload = {"chat_history": [{"role": "user", "content": "I feel tired"}]}
        response = self.client.post("/report", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertIn("administrative", body["data"])
        self.assertEqual(body["data"]["plan"]["care_advice"], "Rest")

    def test_extract_context_budget_error_mapping(self):
        payload = {"user_input": "long transcript", "chat_history": []}
        with patch(
            "routes.http.run_triage_extraction",
            side_effect=LLMContextBudgetExceededError("Requested tokens exceed context window"),
        ):
            response = self.client.post("/extract", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_CONTEXT_BUDGET_EXCEEDED")

    def test_extract_decode_error_mapping(self):
        payload = {"user_input": "symptoms", "chat_history": []}
        with patch(
            "routes.http.run_triage_extraction",
            side_effect=LLMDecodeError("llama_decode returned -1"),
        ):
            response = self.client.post("/extract", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_DECODE_FAILED")

    def test_extract_invalid_json_error_mapping(self):
        self.mock_llm.generate.return_value = "not json"
        payload = {"user_input": "mild cough", "chat_history": []}
        response = self.client.post("/extract", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_INVALID_JSON")

    def test_report_context_budget_error_mapping(self):
        with patch(
            "routes.http.run_triage_report",
            return_value={
                "error": "LLM generation failed",
                "details": "Requested tokens exceed context window",
                "raw_output": "",
            },
        ):
            response = self.client.post("/report", json={"chat_history": []})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_CONTEXT_BUDGET_EXCEEDED")

    def test_report_returns_busy_when_lock_not_acquired(self):
        class BusyLock:
            def acquire(self, timeout=None):
                _ = timeout
                return False

            def release(self):
                pass

        self.mock_services["llm_extraction_lock"] = BusyLock()
        response = self.client.post("/report", json={"chat_history": []})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_BUSY")

    def test_report_fhir_success_envelope(self):
        payload = {
            "report": {
                "administrative": {
                    "process_id": "test-process-001",
                    "timestamp": "2026-02-20T10:30:00Z",
                },
                "patient_information": {
                    "sex": "male",
                    "age": "54",
                },
                "assessment": {
                    "chief_complaint": "Crushing chest pain radiating to left arm",
                    "hpi": {
                        "duration": "20 minutes",
                        "severity": "9/10",
                        "associated_symptoms": ["nausea", "sweating"],
                    },
                    "medical_history": ["hypertension"],
                },
                "disposition": {
                    "triage_level": "Emergency (Red)",
                    "reasoning": "High-risk ACS symptoms",
                },
                "plan": {"care_advice": "Call emergency services immediately"},
            },
            "include_validation": True,
        }
        response = self.client.post("/report/fhir", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertEqual(body["data"]["bundle"]["resourceType"], "Bundle")
        self.assertEqual(body["data"]["bundle"]["type"], "collection")

    def test_report_fhir_input_invalid(self):
        payload = {
            "report": {
                "patient_information": {"sex": "male", "age": "54"},
            },
            "include_validation": True,
        }
        response = self.client.post("/report/fhir", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "FHIR_INPUT_INVALID")

    def test_report_fhir_validation_failed(self):
        payload = {
            "report": {
                "administrative": {
                    "process_id": "test-process-001",
                    "timestamp": "2026-02-20T10:30:00Z",
                },
                "patient_information": {
                    "sex": "male",
                    "age": "54",
                },
                "assessment": {
                    "chief_complaint": "Crushing chest pain radiating to left arm",
                    "hpi": {
                        "duration": "20 minutes",
                        "severity": "9/10",
                        "associated_symptoms": ["nausea", "sweating"],
                    },
                    "medical_history": ["hypertension"],
                },
                "disposition": {
                    "triage_level": "Emergency (Red)",
                    "reasoning": "High-risk ACS symptoms",
                },
                "plan": {"care_advice": "Call emergency services immediately"},
            },
            "include_validation": True,
        }
        with patch(
            "routes.http.validate_fhir_bundle_structure",
            return_value=["fatal: unresolved internal reference"],
        ):
            response = self.client.post("/report/fhir", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "FHIR_VALIDATION_FAILED")

if __name__ == '__main__':
    unittest.main()
