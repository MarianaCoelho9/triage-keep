"""Targeted endpoint stress harness for /analyze, /extract, /report across runtime profiles."""

from __future__ import annotations

import json
import os
import time
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

import main as app_main
from main import app, get_ai_services
from voice_agent.services.llm.base import GenerationOptions


@dataclass(frozen=True)
class Profile:
    n_ctx: int
    max_new_tokens: int
    n_batch: int


class ProfileAwareMockLLM:
    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        generation_options = options or GenerationOptions()
        n_ctx = int(os.getenv("MEDGEMMA_N_CTX", "4096"))
        max_new_tokens = int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "320"))
        requested = generation_options.max_new_tokens or max_new_tokens
        estimated_prompt_tokens = max(1, len(prompt) // 4)

        if estimated_prompt_tokens + requested + 64 > n_ctx:
            raise RuntimeError(
                f"Requested tokens ({estimated_prompt_tokens + requested}) exceed context window of {n_ctx}"
            )

        if "JSON REPORT OUTPUT:" in prompt:
            return json.dumps(
                {
                    "patient_information": {"sex": "unknown", "age": "Not specified"},
                    "assessment": {
                        "chief_complaint": "headache",
                        "hpi": {
                            "onset": "today",
                            "duration": "2 hours",
                            "severity": "mild",
                            "associated_symptoms": ["nausea"],
                        },
                        "red_flags_checked": [],
                        "medical_history": [],
                    },
                    "disposition": {
                        "triage_level": "Non-Urgent (Green)",
                        "reasoning": "No red flags",
                    },
                    "plan": {"care_advice": "Rest and hydration"},
                }
            )

        if "JSON OUTPUT:" in prompt:
            return json.dumps(
                {
                    "main_complaint": "headache",
                    "additional_symptoms": ["nausea"],
                    "medical_history": "",
                    "severity_risk": "low",
                }
            )

        return "<question>Could you tell me when your symptoms started?</question>"


class DummySTT:
    def transcribe(self, _: str) -> str:
        return "transcript"


class DummyTTS:
    def synthesize(self, _: str) -> bytes:
        return b"audio"


def _build_history(turns: int, content: str) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"{content} #{i}"})
    return history


def run_profile(profile: Profile) -> dict[str, Any]:
    os.environ["MEDGEMMA_N_CTX"] = str(profile.n_ctx)
    os.environ["MEDGEMMA_MAX_NEW_TOKENS"] = str(profile.max_new_tokens)
    os.environ["MEDGEMMA_N_BATCH"] = str(profile.n_batch)

    services = {
        "llm": ProfileAwareMockLLM(),
        "llm_interaction": ProfileAwareMockLLM(),
        "llm_extraction": ProfileAwareMockLLM(),
        "stt": DummySTT(),
        "tts": DummyTTS(),
    }
    app.dependency_overrides[get_ai_services] = lambda: services
    app_main.get_services = lambda: services  # Bypass heavy startup in lifespan.

    calls = [
        ("short_analyze", "/analyze", {"user_input": "I feel dizzy", "chat_history": []}),
        ("short_extract", "/extract", {"user_input": "I have mild headache", "chat_history": []}),
        ("short_report", "/report", {"chat_history": _build_history(4, "short symptom")}),
        ("long_extract", "/extract", {"user_input": "summary", "chat_history": _build_history(90, "long symptom narrative")}),
        ("long_report", "/report", {"chat_history": _build_history(120, "long symptom narrative with details")}),
    ]

    results: list[dict[str, Any]] = []
    json_valid = 0
    with redirect_stdout(io.StringIO()):
        with TestClient(app) as client:
            for name, endpoint, payload in calls:
                started = time.perf_counter()
                response = client.post(endpoint, json=payload)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                body = response.json()

                parse_ok = False
                if endpoint in ("/extract", "/report"):
                    parse_ok = bool(body.get("success")) and isinstance(body.get("data"), dict)
                    if parse_ok:
                        json_valid += 1

                error_code = None
                if isinstance(body, dict) and isinstance(body.get("error"), dict):
                    error_code = body["error"].get("code")

                results.append(
                    {
                        "call": name,
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "latency_ms": elapsed_ms,
                        "parse_ok": parse_ok,
                        "error_code": error_code,
                    }
                )

    app.dependency_overrides = {}
    error_distribution: dict[str, int] = {}
    for row in results:
        code = row["error_code"]
        if code is None:
            continue
        error_distribution[code] = error_distribution.get(code, 0) + 1

    return {
        "profile": {
            "MEDGEMMA_N_CTX": profile.n_ctx,
            "MEDGEMMA_MAX_NEW_TOKENS": profile.max_new_tokens,
            "MEDGEMMA_N_BATCH": profile.n_batch,
        },
        "results": results,
        "json_parse_validity_rate": f"{json_valid}/4",
        "error_code_distribution": error_distribution,
    }


def main() -> None:
    profiles = [
        Profile(n_ctx=2048, max_new_tokens=1024, n_batch=128),
        Profile(n_ctx=4096, max_new_tokens=1024, n_batch=128),
        Profile(n_ctx=4096, max_new_tokens=320, n_batch=128),
    ]

    report = [run_profile(profile) for profile in profiles]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
