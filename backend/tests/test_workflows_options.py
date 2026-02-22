import os

from voice_agent.agents.triage.workflows import (
    EXTRACTION_JSON_SCHEMA,
    REPORT_JSON_SCHEMA,
    merge_extraction_state,
    run_triage_extraction,
    run_triage_extraction_incremental,
    run_triage_report,
)


class DummyLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = ""
        self.last_options = None

    def generate(self, prompt, options=None):
        self.last_prompt = prompt
        self.last_options = options
        return self.response


def test_run_triage_extraction_passes_schema_options(monkeypatch):
    monkeypatch.setenv("MEDGEMMA_N_CTX", "4096")
    monkeypatch.setenv("MEDGEMMA_MAX_NEW_TOKENS", "320")
    llm = DummyLLM('{"main_complaint":"cough","additional_symptoms":[],"medical_history":"","severity_risk":"low"}')

    result = run_triage_extraction(
        llm,
        [{"role": "user", "content": "I have mild cough"}],
    )

    assert "JSON OUTPUT:" in llm.last_prompt
    assert result
    assert llm.last_options is not None
    assert llm.last_options.json_schema == EXTRACTION_JSON_SCHEMA
    assert llm.last_options.max_new_tokens == 224


def test_run_triage_report_passes_schema_options(monkeypatch):
    monkeypatch.setenv("MEDGEMMA_N_CTX", "4096")
    monkeypatch.setenv("MEDGEMMA_MAX_NEW_TOKENS", "320")
    llm = DummyLLM(
        '{"patient_information":{"sex":"unknown","age":"Not specified"},'
        '"assessment":{"chief_complaint":"Not specified","hpi":{},"red_flags_checked":[],"medical_history":[]},'
        '"disposition":{"triage_level":"Non-Urgent (Green)","reasoning":"No red flags"},'
        '"plan":{"care_advice":"Hydrate"}}'
    )

    report = run_triage_report(
        llm,
        [{"role": "user", "content": "I feel unwell"}],
    )

    assert "JSON REPORT OUTPUT:" in llm.last_prompt
    assert "administrative" in report
    assert llm.last_options is not None
    assert llm.last_options.json_schema == REPORT_JSON_SCHEMA
    report_cap = int(os.getenv("MEDGEMMA_REPORT_MAX_NEW_TOKENS", "256"))
    assert llm.last_options.max_new_tokens == min(320, report_cap)


def test_run_triage_extraction_incremental_uses_schema_and_merges(monkeypatch):
    monkeypatch.setenv("MEDGEMMA_N_CTX", "4096")
    monkeypatch.setenv("MEDGEMMA_MAX_NEW_TOKENS", "320")
    llm = DummyLLM(
        '{"main_complaint":"headache","additional_symptoms":["nausea"],'
        '"medical_history":"hypertension","severity_risk":"medium"}'
    )

    merged = run_triage_extraction_incremental(
        llm,
        current_state={
            "main_complaint": "headache",
            "additional_symptoms": ["dizziness"],
            "medical_history": "",
            "severity_risk": "low",
        },
        delta_turns=[{"role": "user", "content": "Now I also feel nausea"}],
    )

    assert "CURRENT_STATE" in llm.last_prompt
    assert "DELTA_TURNS" in llm.last_prompt
    assert llm.last_options is not None
    assert llm.last_options.json_schema == EXTRACTION_JSON_SCHEMA
    assert merged["severity_risk"] == "medium"
    assert merged["additional_symptoms"] == ["dizziness", "nausea"]


def test_merge_extraction_state_prefers_informative_values():
    merged = merge_extraction_state(
        current_state={
            "main_complaint": "chest pain",
            "additional_symptoms": ["nausea"],
            "medical_history": "diabetes",
            "severity_risk": "high",
        },
        update_state={
            "main_complaint": "unknown",
            "additional_symptoms": ["Nausea", "fatigue"],
            "medical_history": "Not specified",
            "severity_risk": "low",
        },
    )
    assert merged["main_complaint"] == "chest pain"
    assert merged["medical_history"] == "diabetes"
    assert merged["severity_risk"] == "high"
    assert merged["additional_symptoms"] == ["nausea", "fatigue"]
