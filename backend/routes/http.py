import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from voice_agent.agents.triage import (
    run_triage_extraction,
    run_triage_interaction,
    run_triage_report,
)
from voice_agent.agents.triage.utils import extract_json_from_text
from voice_agent.config import get_services
from voice_agent.core import (
    FHIRExportRequest,
    FHIRExportResponse,
    StatusResponse,
    TranscriptionResponse,
    TriageAnalysisResponse,
    TriageExtractionResponse,
    TriageReportRequest,
    TriageReportResponse,
    TriageRequest,
)
from voice_agent.core.error_mapping import build_report_error_payload
from voice_agent.fhir import map_report_to_fhir_bundle, validate_fhir_bundle_structure
from voice_agent.services.llm.base import (
    LLMContextBudgetExceededError,
    LLMDecodeError,
    LLMGenerationError,
)

router = APIRouter()


def get_ai_services():
    return get_services()


@router.get("/", response_model=StatusResponse)
def read_root():
    return {"status": "online", "system": "TriageKeep Brain"}


@router.post("/analyze", response_model=TriageAnalysisResponse)
def analyze_text(request: TriageRequest, services: dict = Depends(get_ai_services)):
    llm = services.get("llm_interaction", services["llm"])
    llm_lock = services.get("llm_interaction_lock")
    with _lock_context(llm_lock):
        response_text = run_triage_interaction(llm, request.user_input, request.chat_history)

    return {
        "analysis": {
            "response": response_text,
        }
    }


def _build_error_payload(
    code: str,
    message: str,
    details: str | None = None,
    raw_output: str | None = None,
) -> dict:
    payload: dict[str, str] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    if raw_output:
        payload["raw_output"] = raw_output
    return payload


def _parse_json_object(raw_output: str) -> tuple[dict | None, dict | None]:
    try:
        parsed = json.loads(extract_json_from_text(raw_output))
    except json.JSONDecodeError as err:
        return None, _build_error_payload(
            "LLM_INVALID_JSON",
            "Failed to parse model output as JSON object.",
            details=str(err),
            raw_output=raw_output,
        )

    if not isinstance(parsed, dict):
        return None, _build_error_payload(
            "INVALID_JSON_OBJECT",
            "Model output is valid JSON but not an object.",
            raw_output=raw_output,
        )

    return parsed, None


def _has_fatal_validation_issue(issues: list[str]) -> bool:
    fatal_markers = ("fatal", "critical", "missing required")
    return any(marker in issue.lower() for issue in issues for marker in fatal_markers)


def _lock_context(lock: Any):
    if lock is None:
        return nullcontext()
    return lock


def _get_report_http_lock_timeout_seconds() -> float:
    raw = os.environ.get("MEDGEMMA_REPORT_HTTP_LOCK_TIMEOUT_S", "3.0")
    try:
        value = float(raw)
    except ValueError as err:
        raise ValueError("MEDGEMMA_REPORT_HTTP_LOCK_TIMEOUT_S must be a number") from err
    if value <= 0:
        raise ValueError("MEDGEMMA_REPORT_HTTP_LOCK_TIMEOUT_S must be positive")
    return value


@router.post("/extract", response_model=TriageExtractionResponse)
def extract_data(request: TriageRequest, services: dict = Depends(get_ai_services)):
    llm = services.get("llm_extraction", services["llm"])
    llm_lock = services.get("llm_extraction_lock")

    # Optionally append current input to history if relevant for extraction
    history_to_analyze = request.chat_history.copy()
    if request.user_input:
        history_to_analyze.append({"role": "user", "content": request.user_input})

    try:
        with _lock_context(llm_lock):
            response_json_str = run_triage_extraction(llm, history_to_analyze)
    except LLMContextBudgetExceededError as err:
        return {
            "success": False,
            "data": {},
            "error": _build_error_payload(
                "LLM_CONTEXT_BUDGET_EXCEEDED",
                "Prompt and output budget exceed model context window.",
                details=str(err),
            ),
        }
    except LLMDecodeError as err:
        return {
            "success": False,
            "data": {},
            "error": _build_error_payload(
                "LLM_DECODE_FAILED",
                "Model decoding failed.",
                details=str(err),
            ),
        }
    except LLMGenerationError as err:
        return {
            "success": False,
            "data": {},
            "error": _build_error_payload(
                "LLM_DECODE_FAILED",
                "Model generation failed.",
                details=str(err),
            ),
        }

    data, error = _parse_json_object(response_json_str)
    if error:
        return {"success": False, "data": {}, "error": error}
    return {"success": True, "data": data, "error": None}


@router.post("/report", response_model=TriageReportResponse)
def generate_report(
    request: TriageReportRequest, services: dict = Depends(get_ai_services)
):
    llm = services.get("llm_extraction", services["llm"])
    llm_lock = services.get("llm_extraction_lock")
    lock_timeout_s = _get_report_http_lock_timeout_seconds()

    # Optionally append current input to history
    history_to_analyze = request.chat_history.copy()
    if request.user_input and request.user_input.strip():
        history_to_analyze.append({"role": "user", "content": request.user_input})

    if llm_lock is not None and hasattr(llm_lock, "acquire") and hasattr(llm_lock, "release"):
        acquired = llm_lock.acquire(timeout=lock_timeout_s)
        if not acquired:
            return {
                "success": False,
                "data": {},
                "error": _build_error_payload(
                    "LLM_BUSY",
                    "Model is busy with another generation. Retry in a few seconds.",
                ),
            }
        try:
            report = run_triage_report(llm, history_to_analyze)
        finally:
            llm_lock.release()
    else:
        with _lock_context(llm_lock):
            report = run_triage_report(llm, history_to_analyze)
    if "error" in report:
        return {
            "success": False,
            "data": {},
            "error": build_report_error_payload(report),
        }
    return {"success": True, "data": report, "error": None}


@router.post("/report/fhir", response_model=FHIRExportResponse)
def export_report_fhir(request: FHIRExportRequest):
    try:
        bundle, warnings = map_report_to_fhir_bundle(request.report)
    except ValueError as err:
        return {
            "success": False,
            "data": {},
            "error": _build_error_payload(
                "FHIR_INPUT_INVALID",
                "Invalid report payload for FHIR export.",
                details=str(err),
            ),
        }
    except Exception as err:
        return {
            "success": False,
            "data": {},
            "error": _build_error_payload(
                "FHIR_EXPORT_FAILED",
                "Unexpected failure during FHIR export.",
                details=str(err),
            ),
        }

    validation_issues: list[str] = []
    if request.include_validation:
        validation_issues = validate_fhir_bundle_structure(bundle)
        if _has_fatal_validation_issue(validation_issues):
            return {
                "success": False,
                "data": {
                    "bundle": bundle,
                    "warnings": validation_issues,
                },
                "error": _build_error_payload(
                    "FHIR_VALIDATION_FAILED",
                    "Generated bundle failed FHIR structural validation.",
                ),
            }

    return {
        "success": True,
        "data": {
            "bundle": bundle,
            "warnings": [*warnings, *validation_issues],
        },
        "error": None,
    }


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...), services: dict = Depends(get_ai_services)
):
    stt = services["stt"]

    # Save upload to temp file to pass to pipeline
    # HF pipeline usually handles filenames best for various formats
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = stt.transcribe(tmp_path)
    finally:
        os.remove(tmp_path)

    return {"transcript": transcript}


@router.post("/synthesize")
def synthesize_speech(text: str, services: dict = Depends(get_ai_services)):
    tts = services["tts"]
    audio_bytes = tts.synthesize(text)
    # Return as audio
    return Response(content=audio_bytes, media_type="audio/wav")
