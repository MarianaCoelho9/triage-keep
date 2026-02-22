"""Lightweight structural validation for generated FHIR bundles."""

from __future__ import annotations

from typing import Any


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


def validate_fhir_bundle_structure(bundle: dict[str, Any]) -> list[str]:
    """Validate minimal bundle structure and internal references."""
    issues: list[str] = []

    if not isinstance(bundle, dict):
        return ["fatal: bundle must be a JSON object."]

    if bundle.get("resourceType") != "Bundle":
        issues.append("fatal: resourceType must be 'Bundle'.")
    if bundle.get("type") != "collection":
        issues.append("fatal: bundle type must be 'collection'.")

    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        issues.append("fatal: bundle.entry must be a non-empty array.")
        return issues

    full_urls: set[str] = set()
    resource_types: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"fatal: bundle.entry[{index}] must be an object.")
            continue
        full_url = entry.get("fullUrl")
        resource = entry.get("resource")
        if not isinstance(full_url, str) or not full_url.startswith("urn:uuid:"):
            issues.append(f"fatal: bundle.entry[{index}] missing valid urn:uuid fullUrl.")
        else:
            full_urls.add(full_url)

        if not isinstance(resource, dict):
            issues.append(f"fatal: bundle.entry[{index}].resource must be an object.")
            continue
        resource_type = resource.get("resourceType")
        if not isinstance(resource_type, str) or not resource_type:
            issues.append(f"fatal: bundle.entry[{index}] missing resourceType.")
        else:
            resource_types.add(resource_type)

    for required in ("Patient", "Encounter"):
        if required not in resource_types:
            issues.append(f"fatal: missing required resource type '{required}'.")

    unresolved = _collect_references(bundle) - full_urls
    if unresolved:
        for ref in sorted(unresolved):
            issues.append(f"fatal: unresolved internal reference '{ref}'.")

    return issues
