from __future__ import annotations

from copy import deepcopy
import os
import sys
from typing import Any

# Ensure backend modules are importable when running tests directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from voice_agent.fhir.mapping import map_report_to_fhir_bundle
from voice_agent.fhir.validation import validate_fhir_bundle_structure


def _build_chest_pain_report() -> dict[str, Any]:
    return {
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
            "reasoning": (
                "High-risk acute coronary syndrome pattern with crushing chest pain, "
                "left arm radiation, diaphoresis, and nausea."
            ),
        },
        "plan": {
            "care_advice": "Call emergency services immediately",
        },
    }


def _collect_references(node: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str) and value.startswith("urn:uuid:"):
                refs.add(value)
            else:
                refs.update(_collect_references(value))
    elif isinstance(node, list):
        for item in node:
            refs.update(_collect_references(item))
    return refs


def _collect_text_values(node: Any) -> list[str]:
    values: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            values.extend(_collect_text_values(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(_collect_text_values(item))
    elif isinstance(node, str):
        values.append(node)
    return values


def test_map_report_to_fhir_bundle_returns_bundle():
    report = _build_chest_pain_report()
    bundle, warnings = map_report_to_fhir_bundle(report)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert isinstance(warnings, list)


def test_map_report_to_fhir_bundle_contains_core_resources():
    report = _build_chest_pain_report()
    bundle, _warnings = map_report_to_fhir_bundle(report)

    entries = bundle.get("entry", [])
    resource_types = {
        entry.get("resource", {}).get("resourceType")
        for entry in entries
        if isinstance(entry, dict)
    }
    assert {"Patient", "Encounter", "Condition", "Observation", "CarePlan"}.issubset(resource_types)


def test_map_report_to_fhir_bundle_has_resolvable_internal_references():
    report = _build_chest_pain_report()
    bundle, _warnings = map_report_to_fhir_bundle(report)

    entries = bundle.get("entry", [])
    full_urls = {
        entry.get("fullUrl")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("fullUrl"), str)
    }
    references = _collect_references(bundle)

    unresolved = references - full_urls
    assert not unresolved, f"Unresolved internal references: {sorted(unresolved)}"


def test_map_report_to_fhir_bundle_preserves_clinical_signal_from_report():
    report = _build_chest_pain_report()
    bundle, _warnings = map_report_to_fhir_bundle(report)

    bundle_text = " ".join(_collect_text_values(bundle)).lower()
    assert "chest pain" in bundle_text
    assert "9/10" in bundle_text
    assert "emergency" in bundle_text


def test_map_report_to_fhir_bundle_handles_unknown_optional_fields():
    report = deepcopy(_build_chest_pain_report())
    report["patient_information"]["sex"] = "unknown"
    report["patient_information"]["age"] = None
    report["assessment"]["medical_history"] = None

    bundle, warnings = map_report_to_fhir_bundle(report)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert isinstance(warnings, list)


def test_validate_fhir_bundle_structure_returns_no_fatal_errors_for_fixture():
    report = _build_chest_pain_report()
    bundle, _warnings = map_report_to_fhir_bundle(report)
    issues = validate_fhir_bundle_structure(bundle)

    assert isinstance(issues, list)
    assert not any(
        marker in issue.lower()
        for issue in issues
        for marker in ("fatal", "critical", "missing required")
    )
