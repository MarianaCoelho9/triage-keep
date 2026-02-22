"""Deterministic mapping from triage report JSON to FHIR R4 bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4


REQUIRED_REPORT_KEYS = {
    "administrative",
    "patient_information",
    "assessment",
    "disposition",
    "plan",
}


@dataclass(frozen=True)
class _ResourceRef:
    full_url: str
    resource: dict[str, Any]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list_of_text(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    text = str(values).strip()
    return [text] if text else []


def _to_urn(resource_id: str) -> str:
    return f"urn:uuid:{resource_id}"


def _require_report_shape(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - set(report.keys())
    if missing:
        sorted_missing = ", ".join(sorted(missing))
        raise ValueError(f"report is missing required keys: {sorted_missing}")


def _map_gender(value: str, warnings: list[str]) -> str | None:
    normalized = value.lower()
    if not normalized or normalized in {"unknown", "not specified"}:
        warnings.append("Patient sex unavailable; omitting Patient.gender.")
        return None
    if normalized in {"male", "female", "other", "unknown"}:
        return normalized
    warnings.append(f"Unrecognized patient sex '{value}'; omitting Patient.gender.")
    return None


def _build_patient(report: dict[str, Any], warnings: list[str]) -> _ResourceRef:
    patient_id = str(uuid4())
    patient_info = report["patient_information"]
    sex = _normalize_text(patient_info.get("sex"))
    age = _normalize_text(patient_info.get("age"))

    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "text": {
            "status": "generated",
            "div": (
                "<div>"
                f"Patient demographics: sex={sex or 'unknown'}, age={age or 'unknown'}."
                "</div>"
            ),
        },
    }
    gender = _map_gender(sex, warnings)
    if gender is not None:
        patient["gender"] = gender
    if age and age.lower() not in {"unknown", "not specified"}:
        patient["extension"] = [
            {
                "url": "http://example.org/fhir/StructureDefinition/patient-age-text",
                "valueString": age,
            }
        ]
    else:
        warnings.append("Patient age unavailable; exported as narrative text only.")

    return _ResourceRef(
        full_url=_to_urn(patient_id),
        resource=patient,
    )


def _build_encounter(
    report: dict[str, Any],
    patient_ref: _ResourceRef,
) -> _ResourceRef:
    encounter_id = str(uuid4())
    administrative = report["administrative"]
    assessment = report["assessment"]
    disposition = report["disposition"]

    timestamp = _normalize_text(administrative.get("timestamp"))
    chief_complaint = _normalize_text(assessment.get("chief_complaint")) or "Not specified"
    triage_level = _normalize_text(disposition.get("triage_level")) or "unknown"
    reasoning = _normalize_text(disposition.get("reasoning")) or "Not specified"

    encounter = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory",
        },
        "subject": {"reference": patient_ref.full_url},
        "period": {"start": timestamp} if timestamp else {},
        "reasonCode": [{"text": chief_complaint}],
        "priority": {"text": triage_level},
        "text": {
            "status": "generated",
            "div": (
                "<div>"
                f"Triage encounter. Complaint: {chief_complaint}. "
                f"Triage level: {triage_level}. Reasoning: {reasoning}."
                "</div>"
            ),
        },
    }

    return _ResourceRef(
        full_url=_to_urn(encounter_id),
        resource=encounter,
    )


def _build_condition(
    report: dict[str, Any],
    patient_ref: _ResourceRef,
    encounter_ref: _ResourceRef,
) -> _ResourceRef:
    condition_id = str(uuid4())
    assessment = report["assessment"]
    disposition = report["disposition"]
    hpi = assessment.get("hpi", {})

    chief_complaint = _normalize_text(assessment.get("chief_complaint")) or "Not specified"
    duration = _normalize_text(hpi.get("duration")) or "Not specified"
    severity = _normalize_text(hpi.get("severity")) or "Not specified"
    symptoms = _as_list_of_text(hpi.get("associated_symptoms"))
    reasoning = _normalize_text(disposition.get("reasoning")) or "Not specified"

    condition = {
        "resourceType": "Condition",
        "id": condition_id,
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }
            ]
        },
        "code": {"text": chief_complaint},
        "subject": {"reference": patient_ref.full_url},
        "encounter": {"reference": encounter_ref.full_url},
        "note": [
            {"text": f"Duration: {duration}"},
            {"text": f"Severity: {severity}"},
            {"text": f"Associated symptoms: {', '.join(symptoms) if symptoms else 'none'}"},
            {"text": f"Triage reasoning: {reasoning}"},
        ],
    }

    return _ResourceRef(
        full_url=_to_urn(condition_id),
        resource=condition,
    )


def _build_observation(
    report: dict[str, Any],
    patient_ref: _ResourceRef,
    encounter_ref: _ResourceRef,
) -> _ResourceRef:
    observation_id = str(uuid4())
    hpi = report["assessment"].get("hpi", {})
    symptoms = _as_list_of_text(hpi.get("associated_symptoms"))
    severity = _normalize_text(hpi.get("severity")) or "unknown"
    duration = _normalize_text(hpi.get("duration")) or "unknown"

    observation = {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "final",
        "code": {"text": "Triage symptom severity"},
        "subject": {"reference": patient_ref.full_url},
        "encounter": {"reference": encounter_ref.full_url},
        "valueString": severity,
        "note": [
            {"text": f"Duration: {duration}"},
            {"text": f"Associated symptoms: {', '.join(symptoms) if symptoms else 'none'}"},
        ],
    }

    return _ResourceRef(
        full_url=_to_urn(observation_id),
        resource=observation,
    )


def _build_care_plan(
    report: dict[str, Any],
    patient_ref: _ResourceRef,
    encounter_ref: _ResourceRef,
) -> _ResourceRef:
    care_plan_id = str(uuid4())
    plan = report["plan"]
    disposition = report["disposition"]

    advice = _normalize_text(plan.get("care_advice")) or "Not specified"
    triage_level = _normalize_text(disposition.get("triage_level")) or "unknown"
    reasoning = _normalize_text(disposition.get("reasoning")) or "Not specified"

    care_plan = {
        "resourceType": "CarePlan",
        "id": care_plan_id,
        "status": "active",
        "intent": "plan",
        "subject": {"reference": patient_ref.full_url},
        "encounter": {"reference": encounter_ref.full_url},
        "description": advice,
        "note": [
            {"text": f"Triage level: {triage_level}"},
            {"text": f"Triage reasoning: {reasoning}"},
        ],
    }

    return _ResourceRef(
        full_url=_to_urn(care_plan_id),
        resource=care_plan,
    )


def map_report_to_fhir_bundle(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map triage report JSON to a deterministic FHIR Bundle."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    _require_report_shape(report)

    warnings: list[str] = []
    patient = _build_patient(report, warnings)
    encounter = _build_encounter(report, patient)
    condition = _build_condition(report, patient, encounter)
    observation = _build_observation(report, patient, encounter)
    care_plan = _build_care_plan(report, patient, encounter)

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"fullUrl": patient.full_url, "resource": patient.resource},
            {"fullUrl": encounter.full_url, "resource": encounter.resource},
            {"fullUrl": condition.full_url, "resource": condition.resource},
            {"fullUrl": observation.full_url, "resource": observation.resource},
            {"fullUrl": care_plan.full_url, "resource": care_plan.resource},
        ],
    }
    return bundle, warnings
