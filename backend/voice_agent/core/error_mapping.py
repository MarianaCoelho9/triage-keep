"""Shared error code mapping for report and LLM failures."""
from typing import Any, Mapping

REPORT_ERROR_CODE_CONTEXT_BUDGET = "LLM_CONTEXT_BUDGET_EXCEEDED"
REPORT_ERROR_CODE_DECODE = "LLM_DECODE_FAILED"
REPORT_ERROR_CODE_INVALID_JSON = "LLM_INVALID_JSON"
REPORT_ERROR_CODE_GENERIC = "REPORT_GENERATION_FAILED"


def classify_report_error_code(error_message: str, details: str | None) -> str:
    """Classify report failure metadata into a stable error code."""
    lowered_details = (details or "").lower()
    lowered_error = error_message.lower()
    if "context window" in lowered_details or "context budget" in lowered_details:
        return REPORT_ERROR_CODE_CONTEXT_BUDGET
    if "llama_decode returned -1" in lowered_details or "decode" in lowered_details:
        return REPORT_ERROR_CODE_DECODE
    if "parse report json" in lowered_error or "not an object" in lowered_error:
        return REPORT_ERROR_CODE_INVALID_JSON
    return REPORT_ERROR_CODE_GENERIC


def build_report_error_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build standardized error payload from report generation output."""
    details = str(report.get("details", "")) or None
    error_message = str(report.get("error", ""))
    payload: dict[str, Any] = {
        "code": classify_report_error_code(error_message, details),
        "message": error_message,
    }
    if details:
        payload["details"] = details
    raw_output = str(report.get("raw_output", "")) or None
    if raw_output:
        payload["raw_output"] = raw_output
    return payload
