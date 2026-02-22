"""Structured JSON logging utilities for voice pipeline tracing."""
from __future__ import annotations

import contextvars
import json
import logging
import math
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "voice_session_id",
    default=None,
)
_turn_id_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "voice_turn_id",
    default=None,
)
_metrics_session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "voice_metrics_session_id",
    default=None,
)

_metrics_lock = threading.Lock()
_session_stage_metrics: dict[str, dict[str, list[float]]] = {}
_session_stage_status_counts: dict[str, dict[str, dict[str, int]]] = {}

_level_map: dict[LogLevelName, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("voice_agent.structured")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def set_session_id(session_id: str | None) -> None:
    """Store the active websocket session id for the current context."""
    _session_id_ctx.set(session_id)
    _metrics_session_id_ctx.set(session_id)


def get_session_id() -> str | None:
    """Return the active websocket session id."""
    return _session_id_ctx.get()


def set_turn_id(turn_id: int | None) -> None:
    """Store the active turn id for the current context."""
    _turn_id_ctx.set(turn_id)


def get_turn_id() -> int | None:
    """Return the active turn id."""
    return _turn_id_ctx.get()


def clear_log_context() -> None:
    """Reset session and turn tracing metadata for the current context."""
    set_session_id(None)
    set_turn_id(None)


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log_event(
    *,
    component: str,
    event: str,
    level: LogLevelName = "INFO",
    session_id: str | None = None,
    turn_id: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured JSON log line to stdout."""
    resolved_session_id = session_id if session_id is not None else get_session_id()
    resolved_turn_id = turn_id if turn_id is not None else get_turn_id()
    payload: dict[str, Any] = {
        "ts": _iso_timestamp(),
        "level": level,
        "component": component,
        "event": event,
        "session_id": resolved_session_id,
        "turn_id": resolved_turn_id,
        "details": dict(details or {}),
    }
    _get_logger().log(_level_map[level], json.dumps(payload, ensure_ascii=True, separators=(",", ":")))


def _record_latency_metric(stage: str, duration_ms: float, status: str) -> None:
    session_id = _metrics_session_id_ctx.get()
    if not session_id:
        return

    with _metrics_lock:
        stage_metrics = _session_stage_metrics.setdefault(session_id, {})
        stage_status_counts = _session_stage_status_counts.setdefault(session_id, {})

        stage_metrics.setdefault(stage, []).append(duration_ms)
        status_counts = stage_status_counts.setdefault(stage, {})
        status_counts[status] = status_counts.get(status, 0) + 1


def _duration_to_ms(duration_s: float) -> float:
    if duration_s < 0:
        return 0.0
    return round(duration_s * 1000.0, 3)


def log_latency_event(
    *,
    component: str,
    event: str,
    stage: str,
    duration_s: float,
    status: str,
    turn_id: int | None = None,
    level: LogLevelName = "INFO",
    details: Mapping[str, Any] | None = None,
) -> None:
    """Emit latency metric log event and track per-session summary stats."""
    duration_ms = _duration_to_ms(duration_s)
    payload_details = dict(details or {})
    payload_details.update(
        {
            "stage": stage,
            "status": status,
            "duration_ms": duration_ms,
        }
    )
    _record_latency_metric(stage=stage, duration_ms=duration_ms, status=status)
    log_event(
        component=component,
        event=event,
        level=level,
        turn_id=turn_id,
        details=payload_details,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = percentile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def pop_session_metrics_summary(session_id: str) -> dict[str, Any]:
    """Pop collected latency metrics for one session and return summary stats."""
    with _metrics_lock:
        stage_metrics = _session_stage_metrics.pop(session_id, {})
        stage_status_counts = _session_stage_status_counts.pop(session_id, {})

    stages_summary: dict[str, dict[str, Any]] = {}
    for stage, durations in stage_metrics.items():
        if not durations:
            continue
        count = len(durations)
        avg_ms = round(sum(durations) / count, 3)
        p95_ms = round(_percentile(durations, 0.95), 3)
        max_ms = round(max(durations), 3)
        stages_summary[stage] = {
            "count": count,
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "status_counts": dict(stage_status_counts.get(stage, {})),
        }

    return {"stages": stages_summary}
