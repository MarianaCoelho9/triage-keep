"""Manual A/B harness: END_SESSION marker vs tool-calling for session finalization.

This is intentionally not a default pytest unit test. Run it manually against the
local GGUF model to compare reliability and latency for the two approaches.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_cpp import Llama


DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_SAVE_PATH = Path("/tmp/end_session_vs_toolcall_report.json")


END_SESSION_SYSTEM_PROMPT = (
    "You are a triage assistant. "
    "If triage is complete, provide concise orientation and append END_SESSION "
    "as the final line. Otherwise ask exactly one short question."
)

TOOL_SYSTEM_PROMPT = (
    "You are a triage assistant. "
    "If triage is complete, call finalize_report. "
    "Otherwise ask exactly one short question. "
    "When calling the tool, pass {\"finalize\": true}. "
    "Example: if user confirms they will follow orientation and no more triage "
    "questions are needed, call finalize_report with {\"finalize\": true}."
)


SHARED_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Hello, I have had frontal headaches for the last two days.",
    },
    {
        "role": "assistant",
        "content": "Can you describe the headache and how severe it is from 1 to 10?",
    },
    {
        "role": "user",
        "content": (
            "It is constant in the frontal area, around 6/10, and I have blurry vision "
            "when trying to read far."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Given this pattern, you should be evaluated by a healthcare professional today. "
            "Seek urgent care if severe worsening, confusion, weakness, chest pain, "
            "or breathing difficulty appears."
        ),
    },
    {
        "role": "user",
        "content": "Okay, I will schedule an appointment. Thank you.",
    },
]


FINALIZE_REPORT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finalize_report",
        "description": "Finalize conversation and generate final triage report.",
        "parameters": {
            "type": "object",
            "properties": {
                "finalize": {
                    "type": "boolean",
                    "description": "Set true to trigger final report generation.",
                },
            },
            "required": ["finalize"],
        },
    },
}


@dataclass(frozen=True)
class RunConfig:
    model_path: Path
    n_ctx: int
    n_batch: int
    n_threads: int
    n_gpu_layers: int
    max_tokens: int
    temperature: float
    seeds: tuple[int, ...]
    include_forced_tool: bool


@dataclass(frozen=True)
class AttemptResult:
    seed: int
    success: bool
    latency_ms: float
    finish_reason: str | None
    output_excerpt: str
    details: dict[str, Any]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[rank], 2)


def _build_config_from_env(args: argparse.Namespace) -> RunConfig:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    model_path_raw = os.getenv("MEDGEMMA_GGUF_PATH", "").strip()
    if not model_path_raw:
        raise ValueError("MEDGEMMA_GGUF_PATH is required in .env or environment.")
    model_path = Path(model_path_raw)
    if not model_path.exists():
        raise FileNotFoundError(f"MEDGEMMA_GGUF_PATH does not exist: {model_path}")

    def env_int(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            return int(raw)
        except ValueError as err:
            raise ValueError(f"{name} must be an integer, got: {raw}") from err

    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    return RunConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch if args.n_batch is not None else env_int("MEDGEMMA_N_BATCH", 128),
        n_threads=args.n_threads if args.n_threads is not None else env_int("MEDGEMMA_N_THREADS", 0),
        n_gpu_layers=args.n_gpu_layers if args.n_gpu_layers is not None else env_int("MEDGEMMA_GPU_LAYERS", -1),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seeds=seeds,
        include_forced_tool=not args.skip_forced_tool,
    )


def _new_llama(config: RunConfig) -> Llama:
    n_threads = config.n_threads if config.n_threads > 0 else (os.cpu_count() or 1)
    return Llama(
        model_path=str(config.model_path),
        n_ctx=config.n_ctx,
        n_batch=config.n_batch,
        n_threads=n_threads,
        n_gpu_layers=config.n_gpu_layers,
        verbose=False,
    )


def _end_session_success(content: str) -> tuple[bool, dict[str, Any]]:
    marker_count = content.count("END_SESSION")
    before_marker = content.split("END_SESSION", maxsplit=1)[0].strip() if marker_count else ""
    success = marker_count == 1 and bool(before_marker)
    return success, {
        "marker_count": marker_count,
        "orientation_non_empty": bool(before_marker),
    }


def _tool_success(message: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False, {"tool_calls": []}

    matched = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if isinstance(fn, dict) and fn.get("name") == "finalize_report":
            matched.append(call)

    return bool(matched), {"tool_calls": matched}


def _run_end_session_scenario(llm: Llama, config: RunConfig) -> dict[str, Any]:
    attempts: list[AttemptResult] = []

    for seed in config.seeds:
        started = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": END_SESSION_SYSTEM_PROMPT},
                *SHARED_MESSAGES,
            ],
            temperature=config.temperature,
            top_p=1.0,
            max_tokens=config.max_tokens,
            seed=seed,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})
        content = str(message.get("content") or "")

        success, details = _end_session_success(content)
        attempts.append(
            AttemptResult(
                seed=seed,
                success=success,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                output_excerpt=content[:240],
                details=details,
            )
        )

    return _summarize_attempts("end_session", attempts)


def _run_tool_scenario(
    llm: Llama,
    config: RunConfig,
    *,
    forced: bool,
) -> dict[str, Any]:
    attempts: list[AttemptResult] = []

    if forced:
        tool_choice: str | dict[str, Any] = {
            "type": "function",
            "function": {"name": "finalize_report"},
        }
        scenario_name = "tool_call_forced"
    else:
        tool_choice = "auto"
        scenario_name = "tool_call_auto"

    for seed in config.seeds:
        started = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                *SHARED_MESSAGES,
            ],
            tools=[FINALIZE_REPORT_TOOL],
            tool_choice=tool_choice,
            temperature=config.temperature,
            top_p=1.0,
            max_tokens=config.max_tokens,
            seed=seed,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})
        content = str(message.get("content") or "")

        success, details = _tool_success(message)
        attempts.append(
            AttemptResult(
                seed=seed,
                success=success,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                output_excerpt=content[:240],
                details=details,
            )
        )

    return _summarize_attempts(scenario_name, attempts)


def _summarize_attempts(name: str, attempts: list[AttemptResult]) -> dict[str, Any]:
    successes = [attempt for attempt in attempts if attempt.success]
    failures = [attempt for attempt in attempts if not attempt.success]
    latencies = [attempt.latency_ms for attempt in attempts]

    finish_reasons: dict[str, int] = {}
    for attempt in attempts:
        key = attempt.finish_reason or "unknown"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1

    return {
        "scenario": name,
        "runs": len(attempts),
        "successes": len(successes),
        "success_rate": round((len(successes) / len(attempts)) if attempts else 0.0, 4),
        "latency_ms": {
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": _percentile(latencies, 95.0),
        },
        "finish_reason_distribution": finish_reasons,
        "sample_failures": [
            {
                "seed": failure.seed,
                "finish_reason": failure.finish_reason,
                "output_excerpt": failure.output_excerpt,
                "details": failure.details,
            }
            for failure in failures[:2]
        ],
    }


def _evaluate_decision(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = {entry["scenario"]: entry for entry in report["scenarios"]}

    end_rate = float(scenarios["end_session"]["success_rate"])
    auto_tool_rate = float(scenarios["tool_call_auto"]["success_rate"])
    forced_tool_rate = None
    if "tool_call_forced" in scenarios:
        forced_tool_rate = float(scenarios["tool_call_forced"]["success_rate"])

    recommendation = "inconclusive"
    rationale = "No decision rule matched exactly."

    if end_rate >= 0.85 and auto_tool_rate < 0.70:
        recommendation = "keep_end_session_primary"
        rationale = (
            "END_SESSION met threshold while tool auto-calling underperformed. "
            "Keep END_SESSION + timeout fallback + manual FE end."
        )
    elif auto_tool_rate >= end_rate:
        recommendation = "consider_tool_first"
        rationale = (
            "Tool auto-calling reached parity or better than END_SESSION. "
            "Consider tool-first finalization with fallback."
        )
    elif forced_tool_rate is not None and forced_tool_rate >= 0.85 and auto_tool_rate < 0.70:
        recommendation = "tool_plumbing_ok_model_selection_weak"
        rationale = (
            "Forced-tool succeeds while auto-tool fails. "
            "Backend plumbing is okay; model/template tool selection is weak."
        )

    return {
        "acceptance_rule_inputs": {
            "end_session_success_rate": end_rate,
            "tool_auto_success_rate": auto_tool_rate,
            "tool_forced_success_rate": forced_tool_rate,
        },
        "recommendation": recommendation,
        "rationale": rationale,
    }


def run_ab_test(config: RunConfig) -> dict[str, Any]:
    llm = _new_llama(config)

    scenarios = [
        _run_end_session_scenario(llm, config),
        _run_tool_scenario(llm, config, forced=False),
    ]

    if config.include_forced_tool:
        scenarios.append(_run_tool_scenario(llm, config, forced=True))

    report = {
        "config": {
            "model_path": str(config.model_path),
            "n_ctx": config.n_ctx,
            "n_batch": config.n_batch,
            "n_threads": config.n_threads,
            "n_gpu_layers": config.n_gpu_layers,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "seeds": list(config.seeds),
            "include_forced_tool": config.include_forced_tool,
        },
        "scenarios": scenarios,
    }
    report["decision"] = _evaluate_decision(report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ctx", type=int, default=1024, help="Runtime n_ctx override.")
    parser.add_argument(
        "--n-batch",
        type=int,
        default=None,
        help="Runtime n_batch override (defaults to MEDGEMMA_N_BATCH).",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Runtime n_threads override (defaults to MEDGEMMA_N_THREADS).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="Runtime n_gpu_layers override (defaults to MEDGEMMA_GPU_LAYERS).",
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="Max generation tokens per run.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated integer seeds, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--skip-forced-tool",
        action="store_true",
        help="Skip optional forced-tool scenario.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Also save report JSON to {DEFAULT_SAVE_PATH}.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Custom path to save report JSON.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _build_config_from_env(args)

    report = run_ab_test(config)
    output = json.dumps(report, indent=2, ensure_ascii=True)
    print(output)

    save_path = args.save_path
    if save_path is None and args.save:
        save_path = DEFAULT_SAVE_PATH

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(output + "\n", encoding="utf-8")
        print(f"Saved report to: {save_path}")


if __name__ == "__main__":
    main()
