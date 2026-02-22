"""Triage agent workflow functions."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from ...core.types import ChatHistory
from .prompts import (
    SIMPLE_TRIAGE_PROMPT_NO_THOUGHT,
    TRIAGE_EXTRACTION_PROMPT,
    TRIAGE_EXTRACTION_INCREMENTAL_PROMPT,
    TRIAGE_REPORT_PROMPT,
)
from .utils import parse_agent_response, extract_json_from_text
from ...services import GenerationOptions

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "main_complaint",
        "additional_symptoms",
        "medical_history",
        "severity_risk",
    ],
    "properties": {
        "main_complaint": {"type": "string"},
        "additional_symptoms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "medical_history": {"type": "string"},
        "severity_risk": {
            "type": "string",
            "enum": ["low", "medium", "high", "unknown"],
        },
    },
}

REPORT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["patient_information", "assessment", "disposition", "plan"],
    "properties": {
        "patient_information": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sex", "age"],
            "properties": {
                "sex": {"type": ["string", "null"]},
                "age": {"type": ["string", "null"]},
            },
        },
        "assessment": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "chief_complaint",
                "hpi",
                "medical_history",
            ],
            "properties": {
                "chief_complaint": {"type": ["string", "null"]},
                "hpi": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "duration",
                        "severity",
                        "associated_symptoms",
                    ],
                    "properties": {
                        "duration": {"type": ["string", "null"]},
                        "severity": {"type": ["string", "null"]},
                        "associated_symptoms": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                    },
                },
                "medical_history": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
            },
        },
        "disposition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["triage_level", "reasoning"],
            "properties": {
                "triage_level": {"type": ["string", "null"]},
                "reasoning": {"type": ["string", "null"]},
            },
        },
        "plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["care_advice"],
            "properties": {
                "care_advice": {"type": ["string", "null"]},
            },
        },
    },
}


def _estimate_tokens(text: str) -> int:
    # Practical preflight approximation for prompt budgeting.
    return max(1, len(text) // 4)


def _sanitize_text(text: str) -> str:
    return text.replace("</s>", "").replace("<s>", "").strip()


def _build_history_lines(chat_history: ChatHistory) -> list[str]:
    lines: list[str] = []
    for msg in chat_history:
        role_label = "User" if msg.get("role") == "user" else "Triage Assistant"
        content = _sanitize_text(msg.get("content", ""))
        lines.append(f"{role_label}: {content}\n")
    return lines


def _fit_history_lines(
    prefix: str,
    lines: list[str],
    suffix: str,
    output_tokens: int,
) -> list[str]:
    n_ctx = int(os.getenv("MEDGEMMA_N_CTX", "4096"))
    margin = int(os.getenv("MEDGEMMA_CONTEXT_MARGIN", "64"))
    kept = list(lines)
    while kept:
        prompt = prefix + "".join(kept) + suffix
        total = _estimate_tokens(prompt) + output_tokens + margin
        if total <= n_ctx:
            break
        kept.pop(0)
    return kept


def _task_output_cap(task: str, fallback: int) -> int:
    max_default = int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "320"))
    specific = os.getenv(task)
    if specific is None:
        specific_value = fallback
    else:
        specific_value = int(specific)
    return min(max_default, specific_value)


def _truncate_chat_history(chat_history: ChatHistory, max_chars: int) -> ChatHistory:
    """Return the most recent chat history that fits within max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    truncated: ChatHistory = []
    total = 0

    for msg in reversed(chat_history):
        role_label = "User" if msg.get("role") == "user" else "Triage Assistant"
        content = msg.get("content", "")
        line = f"{role_label}: {content}\n"
        line_len = len(line)

        if total + line_len > max_chars:
            # If this single line would exceed budget, keep a trimmed tail
            remaining = max_chars - total
            if remaining <= 0:
                break
            trimmed_content = content[-max(0, remaining - len(role_label) - 3) :]
            trimmed_line = f"{role_label}: {trimmed_content}\n"
            truncated.append({"role": msg.get("role", ""), "content": trimmed_content})
            total += len(trimmed_line)
            break

        truncated.append(msg)
        total += line_len

    return list(reversed(truncated))


def run_triage_interaction(llm, user_input: str, chat_history: ChatHistory) -> str:
    """
    Constructs the prompt with system instructions and chat history,
    then generates a response from the LLM.
    Returns the parsed question for the user.

    :param llm: LLM service instance with generate() method
    :param user_input: Current user input text
    :param chat_history: List of previous chat messages
    :return: Question to ask the user
    """
    output_cap = _task_output_cap("MEDGEMMA_INTERACTION_MAX_NEW_TOKENS", 224)
    safe_user_input = _sanitize_text(user_input)
    prefix = f"{SIMPLE_TRIAGE_PROMPT_NO_THOUGHT}\n\n"
    suffix = f"User: {safe_user_input}\nTriage Assistant:"
    history_lines = _fit_history_lines(
        prefix=prefix,
        lines=_build_history_lines(chat_history),
        suffix=suffix,
        output_tokens=output_cap,
    )
    prompt = prefix + "".join(history_lines) + suffix
    print(f"[Run Triage] Prompt: {prompt}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SIMPLE_TRIAGE_PROMPT_NO_THOUGHT.strip()}
    ]
    for msg in chat_history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        messages.append(
            {
                "role": role,
                "content": _sanitize_text(msg.get("content", "")),
            }
        )
    messages.append({"role": "user", "content": safe_user_input})

    # 2. Generate
    response_text = llm.generate(
        prompt,
        options=GenerationOptions(
            stop=[
                "</question>",
                "\nUser:",
                "\nTriage Assistant:",
                "</s>",
                "<s>",
            ],
            max_new_tokens=output_cap,
            messages=messages,
        ),
    )
    print(f"[Run Triage] Response: {response_text}")

    # 3. Parse and return only the question (though we keep the thought for logging if needed)
    thought, question = parse_agent_response(response_text)

    if thought:
        if len(thought) > 40:
            print(f"[Run Triage] Thought: {thought[:20]}...{thought[-20:]}")
        else:
            print(f"[Run Triage] Thought: {thought}")

    return question


def run_triage_extraction(llm, chat_history: ChatHistory) -> str:
    """
    Uses the TRIAGE_EXTRACTION_PROMPT to extract structured clinical data
    from a chat history.
    Returns the raw LLM response (JSON string).

    :param llm: LLM service instance with generate() method
    :param chat_history: List of chat messages to extract from
    :return: JSON string with extracted data
    """
    output_cap = _task_output_cap("MEDGEMMA_EXTRACTION_MAX_NEW_TOKENS", 224)
    prefix = f"{TRIAGE_EXTRACTION_PROMPT}\n\nCHAT HISTORY:\n"
    suffix = "\nJSON OUTPUT:"
    history_lines = _fit_history_lines(
        prefix=prefix,
        lines=_build_history_lines(chat_history),
        suffix=suffix,
        output_tokens=output_cap,
    )
    prompt = prefix + "".join(history_lines) + suffix

    # Generate
    response_text = llm.generate(
        prompt,
        options=GenerationOptions(
            max_new_tokens=output_cap,
            json_schema=EXTRACTION_JSON_SCHEMA,
        ),
    )

    result = extract_json_from_text(response_text)
    if len(result) > 40:
        print(f"[Run Triage] Extraction Result: {result[:20]}...{result[-20:]}")
    else:
        print(f"[Run Triage] Extraction Result: {result}")

    return result


def _is_informative_string(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "unknown", "not specified"}


def normalize_extraction_state(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize extraction state to a schema-shaped payload."""
    main_complaint = data.get("main_complaint", "")
    if not isinstance(main_complaint, str):
        main_complaint = str(main_complaint)

    medical_history = data.get("medical_history", "")
    if not isinstance(medical_history, str):
        medical_history = str(medical_history)

    symptoms_raw = data.get("additional_symptoms", [])
    additional_symptoms: list[str]
    if isinstance(symptoms_raw, list):
        additional_symptoms = [
            str(item).strip() for item in symptoms_raw if str(item).strip()
        ]
    else:
        as_text = str(symptoms_raw).strip()
        additional_symptoms = [as_text] if as_text else []

    severity_raw = str(data.get("severity_risk", "unknown")).strip().lower()
    if severity_raw not in {"low", "medium", "high", "unknown"}:
        severity_raw = "unknown"

    return {
        "main_complaint": main_complaint.strip(),
        "additional_symptoms": additional_symptoms,
        "medical_history": medical_history.strip(),
        "severity_risk": severity_raw,
    }


def merge_extraction_state(
    current_state: dict[str, Any],
    update_state: dict[str, Any],
) -> dict[str, Any]:
    """Merge incremental extraction update into current state deterministically."""
    current = normalize_extraction_state(current_state)
    update = normalize_extraction_state(update_state)

    merged = dict(current)

    if _is_informative_string(update["main_complaint"]):
        merged["main_complaint"] = update["main_complaint"]

    if _is_informative_string(update["medical_history"]):
        merged["medical_history"] = update["medical_history"]

    seen: set[str] = set()
    combined_symptoms: list[str] = []
    for symptom in [*current["additional_symptoms"], *update["additional_symptoms"]]:
        if not symptom:
            continue
        key = symptom.lower()
        if key in seen:
            continue
        seen.add(key)
        combined_symptoms.append(symptom)
    merged["additional_symptoms"] = combined_symptoms

    severity_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    current_severity = current["severity_risk"]
    update_severity = update["severity_risk"]
    if severity_rank[update_severity] >= severity_rank[current_severity]:
        merged["severity_risk"] = update_severity
    else:
        merged["severity_risk"] = current_severity

    return normalize_extraction_state(merged)


def run_triage_extraction_incremental(
    llm,
    current_state: dict[str, Any],
    delta_turns: ChatHistory,
) -> dict[str, Any]:
    """
    Incrementally update extraction state from recent chat turns only.

    :param llm: LLM service instance with generate() method
    :param current_state: Current structured extraction state
    :param delta_turns: New chat turns since the last extraction
    :return: Merged extraction state
    """
    output_cap = _task_output_cap("MEDGEMMA_EXTRACTION_MAX_NEW_TOKENS", 224)
    prompt = (
        f"{TRIAGE_EXTRACTION_INCREMENTAL_PROMPT}\n\n"
        f"CURRENT_STATE:\n{json.dumps(normalize_extraction_state(current_state), ensure_ascii=True)}\n\n"
        f"DELTA_TURNS:\n"
    )
    for msg in delta_turns:
        role_label = "User" if msg.get("role") == "user" else "Triage Assistant"
        prompt += f"{role_label}: {_sanitize_text(msg.get('content', ''))}\n"
    prompt += "\nUPDATED_JSON_OUTPUT:"

    response_text = llm.generate(
        prompt,
        options=GenerationOptions(
            max_new_tokens=output_cap,
            json_schema=EXTRACTION_JSON_SCHEMA,
        ),
    )
    parsed = json.loads(extract_json_from_text(response_text))
    if not isinstance(parsed, dict):
        raise ValueError("Incremental extraction output is not a JSON object")
    return merge_extraction_state(current_state, parsed)


def run_triage_report(llm, chat_history: ChatHistory) -> Dict[str, Any]:
    """
    Uses the TRIAGE_REPORT_PROMPT to generate a full JSON report
    from a chat history.
    Returns the raw LLM response (JSON string).

    :param llm: LLM service instance with generate() method
    :param chat_history: List of chat messages to create report from
    :return: JSON string with complete triage report
    """
    administrative = {
        "process_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    max_chars = int(os.getenv("MEDGEMMA_REPORT_MAX_CHARS", "10000"))
    safe_history = _truncate_chat_history(chat_history, max_chars)

    output_cap = _task_output_cap("MEDGEMMA_REPORT_MAX_NEW_TOKENS", 256)
    prefix = f"{TRIAGE_REPORT_PROMPT}\n\nCHAT HISTORY:\n"
    suffix = "\nJSON REPORT OUTPUT:"
    history_lines = _fit_history_lines(
        prefix=prefix,
        lines=_build_history_lines(safe_history),
        suffix=suffix,
        output_tokens=output_cap,
    )
    prompt = prefix + "".join(history_lines) + suffix

    # Generate
    try:
        response_text = llm.generate(
            prompt,
            options=GenerationOptions(
                max_new_tokens=output_cap,
                json_schema=REPORT_JSON_SCHEMA,
            ),
        )
    except RuntimeError as err:
        return {
            "administrative": administrative,
            "raw_output": "",
            "error": "LLM generation failed",
            "details": str(err),
        }

    try:
        clean_str = extract_json_from_text(response_text)
        data = json.loads(clean_str)
        if isinstance(data, dict):
            data["administrative"] = administrative
            return data
        return {
            "administrative": administrative,
            "raw_output": response_text,
            "error": "Report JSON is not an object",
        }
    except json.JSONDecodeError:
        return {
            "administrative": administrative,
            "raw_output": response_text,
            "error": "Failed to parse report JSON",
        }
