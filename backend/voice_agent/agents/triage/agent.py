"""Main triage agent with interaction-priority incremental extraction."""
from __future__ import annotations

import asyncio
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ...config import get_services
from ...core import (
    AgentChunkEvent,
    ChatHistory,
    ExtractionEvent,
    ExtractionStatusEvent,
    ReportEvent,
    ReportStatusEvent,
    STTOutputEvent,
    VoiceAgentEvent,
)
from ...core.error_mapping import build_report_error_payload
from ...core.logging_utils import log_event, log_latency_event, set_turn_id
from .safety_rules import evaluate_emergency_trigger
from .workflows import (
    normalize_extraction_state,
    run_triage_extraction_incremental,
    run_triage_interaction,
    run_triage_report,
)


def _get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as err:
        raise ValueError(f"{name} must be a number") from err
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer") from err
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_extraction_timeout_seconds() -> float:
    return _get_float_env("MEDGEMMA_EXTRACTION_TIMEOUT_S", 35.0)


def _get_final_extraction_timeout_seconds() -> float:
    return _get_float_env("MEDGEMMA_EXTRACTION_FINAL_TIMEOUT_S", 60.0)


def _get_extraction_debounce_seconds() -> float:
    return _get_float_env("MEDGEMMA_EXTRACTION_DEBOUNCE_S", 3.0)


def _get_extraction_max_delta_turns() -> int:
    return _get_int_env("MEDGEMMA_EXTRACTION_MAX_DELTA_TURNS", 8)


def _is_extraction_status_enabled() -> bool:
    raw = os.environ.get("MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_report_timeout_seconds() -> float:
    return _get_float_env("MEDGEMMA_REPORT_AUTO_TIMEOUT_S", 90.0)


def _get_report_status_heartbeat_seconds() -> float:
    return _get_float_env("MEDGEMMA_REPORT_STATUS_HEARTBEAT_S", 2.0)


def _is_emergency_rule_gate_enabled() -> bool:
    raw = os.environ.get("MEDGEMMA_ENABLE_EMERGENCY_RULE_GATE", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_emergency_escalation_message() -> str:
    default_message = (
        "This may be an emergency. Call emergency services now or go to the nearest "
        "ER immediately. I can keep collecting details while help is on the way."
    )
    raw = os.environ.get("MEDGEMMA_EMERGENCY_ESCALATION_MESSAGE", default_message)
    cleaned = raw.strip()
    return cleaned if cleaned else default_message


def _get_end_session_signal() -> str:
    return os.environ.get("MEDGEMMA_SESSION_END_SIGNAL", "END_SESSION").strip()


def _sanitize_text(text: str) -> str:
    return text.replace("</s>", "").replace("<s>", "").strip()


def _extract_end_signal(text: str, signal: str) -> tuple[str, bool]:
    if not signal:
        return text, False
    if signal not in text:
        return text, False
    cleaned = text.replace(signal, "").strip()
    return cleaned, True


def _normalize_assistant_text(text: str) -> str:
    normalized = text
    normalized = re.sub(
        r"(?im)^\s*orientation\s*:\s*",
        "",
        normalized,
    )
    normalized = re.sub(
        r"(?im)^\s*emergency\s+escalation\s+signs\s*:\s*",
        "",
        normalized,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


@dataclass(frozen=True)
class AgentRuntimeConfig:
    llm_interaction: Any
    llm_extraction: Any
    llm_interaction_lock: Any | None
    llm_extraction_lock: Any | None
    extraction_timeout: float
    final_extraction_timeout: float
    report_timeout: float
    report_status_heartbeat_s: float
    extraction_debounce_s: float
    max_delta_turns: int
    emit_status_events: bool
    emergency_rule_gate_enabled: bool
    emergency_escalation_message: str
    end_session_signal: str


@dataclass
class AgentRuntimeState:
    chat_history: ChatHistory = field(default_factory=list)
    extraction_state: dict[str, Any] = field(
        default_factory=lambda: normalize_extraction_state({})
    )
    last_extracted_turn_index: int = 0
    latest_revision: int = 0
    pending_revision: int | None = None
    end_session_requested: bool = False
    latest_turn_id: int | None = None


@dataclass
class AgentRuntimeTasks:
    queue: asyncio.Queue[VoiceAgentEvent] = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    debounce_task: asyncio.Task[None] | None = None
    extraction_task: asyncio.Task[None] | None = None
    producer_task: asyncio.Task[None] | None = None


async def _emit_extraction_status(
    config: AgentRuntimeConfig,
    tasks: AgentRuntimeTasks,
    status: str,
    revision: int,
    is_final: bool = False,
) -> None:
    if not config.emit_status_events:
        return
    await tasks.queue.put(
        ExtractionStatusEvent(status=status, revision=revision, is_final=is_final)
    )


def _compute_delta_turns(
    state: AgentRuntimeState,
    max_delta_turns: int,
) -> ChatHistory:
    if state.last_extracted_turn_index >= len(state.chat_history):
        return []
    delta = state.chat_history[state.last_extracted_turn_index :]
    if len(delta) > max_delta_turns:
        delta = delta[-max_delta_turns:]
    return [dict(turn) for turn in delta]


async def _run_with_optional_lock(
    lock: Any | None,
    func: Callable[..., Any],
    *args: Any,
) -> Any:
    def _call() -> Any:
        if lock is None:
            return func(*args)
        with lock:
            return func(*args)

    return await asyncio.to_thread(_call)


async def _run_incremental_extraction(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
    revision: int,
    state_snapshot: dict[str, Any],
    delta_snapshot: ChatHistory,
    timeout_s: float,
) -> None:
    await _emit_extraction_status(config, tasks, "running", revision)
    log_event(
        component="agent_stream",
        event="extraction_started",
        turn_id=state.latest_turn_id,
        details={
            "revision": revision,
            "delta_turns": len(delta_snapshot),
        },
    )
    extraction_started_at = time.perf_counter()

    try:
        merged_state = await asyncio.wait_for(
            _run_with_optional_lock(
                config.llm_extraction_lock,
                run_triage_extraction_incremental,
                config.llm_extraction,
                state_snapshot,
                delta_snapshot,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        elapsed_s = time.perf_counter() - extraction_started_at
        log_event(
            component="agent_stream",
            event="extraction_timed_out",
            level="WARNING",
            turn_id=state.latest_turn_id,
            details={
                "revision": revision,
                "timeout_s": timeout_s,
            },
        )
        log_latency_event(
            component="agent_stream",
            event="extraction_latency",
            stage="extraction",
            duration_s=elapsed_s,
            status="timed_out",
            turn_id=state.latest_turn_id,
            level="WARNING",
            details={"revision": revision},
        )
        await _emit_extraction_status(config, tasks, "timed_out", revision)
        return
    except Exception as err:
        elapsed_s = time.perf_counter() - extraction_started_at
        log_event(
            component="agent_stream",
            event="extraction_failed",
            level="ERROR",
            turn_id=state.latest_turn_id,
            details={"revision": revision, "error": str(err)},
        )
        log_latency_event(
            component="agent_stream",
            event="extraction_latency",
            stage="extraction",
            duration_s=elapsed_s,
            status="failed",
            turn_id=state.latest_turn_id,
            level="ERROR",
            details={"revision": revision},
        )
        return

    if revision != state.latest_revision:
        await _emit_extraction_status(config, tasks, "stale_discarded", revision)
        return

    state.extraction_state = normalize_extraction_state(merged_state)
    state.last_extracted_turn_index = len(state.chat_history)
    await tasks.queue.put(ExtractionEvent(data=state.extraction_state))
    log_event(
        component="agent_stream",
        event="extraction_completed",
        turn_id=state.latest_turn_id,
        details={
            "revision": revision,
            "extracted_fields": len(state.extraction_state),
        },
    )
    log_latency_event(
        component="agent_stream",
        event="extraction_latency",
        stage="extraction",
        duration_s=time.perf_counter() - extraction_started_at,
        status="completed",
        turn_id=state.latest_turn_id,
        details={"revision": revision},
    )
    await _emit_extraction_status(config, tasks, "completed", revision)


async def _start_extraction_if_needed(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
    timeout_s: float,
) -> None:
    if tasks.extraction_task and not tasks.extraction_task.done():
        state.pending_revision = state.latest_revision
        return

    delta_snapshot = _compute_delta_turns(state, config.max_delta_turns)
    if not delta_snapshot:
        return

    revision = state.latest_revision
    state_snapshot = dict(state.extraction_state)

    async def _job() -> None:
        try:
            await _run_incremental_extraction(
                config=config,
                state=state,
                tasks=tasks,
                revision=revision,
                state_snapshot=state_snapshot,
                delta_snapshot=delta_snapshot,
                timeout_s=timeout_s,
            )
        finally:
            tasks.extraction_task = None
            if (
                state.pending_revision is not None
                and state.pending_revision > revision
                and not tasks.done.is_set()
            ):
                state.pending_revision = None
                _schedule_debounced_extraction(config, state, tasks, delay_s=0.0)

    tasks.extraction_task = asyncio.create_task(_job())


def _schedule_debounced_extraction(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
    delay_s: float | None = None,
) -> None:
    if tasks.debounce_task and not tasks.debounce_task.done():
        tasks.debounce_task.cancel()

    async def _debounced_start() -> None:
        await _emit_extraction_status(config, tasks, "scheduled", state.latest_revision)
        if delay_s is None:
            await asyncio.sleep(config.extraction_debounce_s)
        else:
            await asyncio.sleep(delay_s)

        if tasks.done.is_set():
            return
        await _start_extraction_if_needed(
            config=config,
            state=state,
            tasks=tasks,
            timeout_s=config.extraction_timeout,
        )

    tasks.debounce_task = asyncio.create_task(_debounced_start())


async def _run_final_flush(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
) -> None:
    if tasks.debounce_task and not tasks.debounce_task.done():
        tasks.debounce_task.cancel()
        try:
            await tasks.debounce_task
        except asyncio.CancelledError:
            pass

    if tasks.extraction_task and not tasks.extraction_task.done():
        try:
            await tasks.extraction_task
        except asyncio.CancelledError:
            pass

    if state.last_extracted_turn_index >= len(state.chat_history):
        return

    state.pending_revision = None
    await _start_extraction_if_needed(
        config=config,
        state=state,
        tasks=tasks,
        timeout_s=config.final_extraction_timeout,
    )
    if tasks.extraction_task and not tasks.extraction_task.done():
        try:
            await tasks.extraction_task
        except asyncio.CancelledError:
            pass


async def _run_final_report(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
) -> None:
    await tasks.queue.put(ReportStatusEvent(status="running"))
    report_started_at = time.perf_counter()
    log_event(
        component="agent_stream",
        event="report_generation_started",
        details={
            "timeout_s": config.report_timeout,
            "heartbeat_s": config.report_status_heartbeat_s,
            "chat_turns": len(state.chat_history),
        },
    )

    report_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
        _run_with_optional_lock(
            config.llm_extraction_lock,
            run_triage_report,
            config.llm_extraction,
            list(state.chat_history),
        )
    )
    try:
        while not report_task.done():
            elapsed_s = time.perf_counter() - report_started_at
            remaining_s = config.report_timeout - elapsed_s
            if remaining_s <= 0:
                report_task.cancel()
                with suppress(asyncio.CancelledError):
                    await report_task
                raise asyncio.TimeoutError

            await asyncio.wait(
                {report_task},
                timeout=min(config.report_status_heartbeat_s, remaining_s),
            )
            if report_task.done():
                break

            await tasks.queue.put(ReportStatusEvent(status="running"))
            log_event(
                component="agent_stream",
                event="report_status_heartbeat",
                details={"elapsed_ms": round((time.perf_counter() - report_started_at) * 1000, 3)},
            )

        report = report_task.result()
        if "error" in report:
            log_event(
                component="agent_stream",
                event="report_generation_failed",
                level="WARNING",
                details={"reason": "error_payload_returned"},
            )
            await tasks.queue.put(ReportStatusEvent(status="failed"))
            await tasks.queue.put(
                ReportEvent(
                    success=False,
                    data={},
                    error=build_report_error_payload(report),
                )
            )
            return
        log_event(
            component="agent_stream",
            event="report_generation_completed",
            details={
                "duration_ms": round((time.perf_counter() - report_started_at) * 1000, 3),
            },
        )
        await tasks.queue.put(ReportStatusEvent(status="completed"))
        await tasks.queue.put(ReportEvent(success=True, data=report, error=None))
    except asyncio.TimeoutError:
        if not report_task.done():
            report_task.cancel()
            with suppress(asyncio.CancelledError):
                await report_task
        log_event(
            component="agent_stream",
            event="report_generation_timed_out",
            level="WARNING",
            details={"timeout_s": config.report_timeout},
        )
        await tasks.queue.put(ReportStatusEvent(status="failed"))
        await tasks.queue.put(
            ReportEvent(
                success=False,
                data={},
                error={
                    "code": "REPORT_GENERATION_TIMEOUT",
                    "message": (
                        f"Report generation timed out after {config.report_timeout:.2f}s"
                    ),
                },
            )
        )
    except Exception as err:
        if not report_task.done():
            report_task.cancel()
            with suppress(asyncio.CancelledError):
                await report_task
        log_event(
            component="agent_stream",
            event="report_generation_failed",
            level="ERROR",
            details={"error": str(err)},
        )
        await tasks.queue.put(ReportStatusEvent(status="failed"))
        await tasks.queue.put(
            ReportEvent(
                success=False,
                data={},
                error={
                    "code": "REPORT_GENERATION_FAILED",
                    "message": str(err),
                },
            )
        )


async def _process_stt_turn(
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
    event: STTOutputEvent,
) -> bool:
    user_text = _sanitize_text(event.text)
    if not user_text:
        return False

    turn_id = event.turn_id
    if turn_id is None:
        turn_id = 1 if state.latest_turn_id is None else state.latest_turn_id + 1
    state.latest_turn_id = turn_id
    set_turn_id(turn_id)
    log_event(
        component="agent_stream",
        event="agent_turn_started",
        turn_id=turn_id,
        details={"transcript_chars": len(user_text)},
    )

    trigger_result = evaluate_emergency_trigger(user_text)
    if config.emergency_rule_gate_enabled and trigger_result.triggered:
        response_text = config.emergency_escalation_message
        log_event(
            component="agent_stream",
            event="emergency_escalation_triggered",
            turn_id=turn_id,
            details={
                "rule_id": trigger_result.rule_id,
                "confidence": trigger_result.confidence,
                "matched_terms": list(trigger_result.matched_terms),
            },
        )
        log_latency_event(
            component="agent_stream",
            event="interaction_latency",
            stage="interaction",
            duration_s=0.0,
            status="bypassed_emergency_rule",
            turn_id=turn_id,
        )
    else:
        interaction_started_at = time.perf_counter()
        try:
            response_text = await _run_with_optional_lock(
                config.llm_interaction_lock,
                run_triage_interaction,
                config.llm_interaction,
                user_text,
                state.chat_history,
            )
        except Exception as err:
            elapsed_s = time.perf_counter() - interaction_started_at
            log_event(
                component="agent_stream",
                event="agent_turn_failed",
                level="ERROR",
                turn_id=turn_id,
                details={"error": str(err)},
            )
            log_latency_event(
                component="agent_stream",
                event="interaction_latency",
                stage="interaction",
                duration_s=elapsed_s,
                status="failed",
                turn_id=turn_id,
                level="ERROR",
            )
            response_text = ""
        else:
            log_latency_event(
                component="agent_stream",
                event="interaction_latency",
                stage="interaction",
                duration_s=time.perf_counter() - interaction_started_at,
                status="completed",
                turn_id=turn_id,
            )

    state.chat_history.append({"role": "user", "content": user_text})
    has_end_signal = False
    if response_text:
        cleaned_response, has_end_signal = _extract_end_signal(
            response_text,
            config.end_session_signal,
        )
        cleaned_response = _normalize_assistant_text(cleaned_response)
        if cleaned_response:
            state.chat_history.append({"role": "assistant", "content": cleaned_response})
            await tasks.queue.put(AgentChunkEvent(text=cleaned_response, turn_id=turn_id))
        if has_end_signal:
            state.end_session_requested = True

    log_event(
        component="agent_stream",
        event="agent_turn_completed",
        turn_id=turn_id,
        details={
            "response_chars": len(response_text),
            "end_session_requested": has_end_signal,
        },
    )

    state.latest_revision += 1
    state.pending_revision = state.latest_revision
    _schedule_debounced_extraction(config, state, tasks)

    if state.end_session_requested:
        log_event(
            component="agent_stream",
            event="end_session_detected",
            turn_id=turn_id,
        )
        return True

    return False


async def _produce_events(
    event_stream: AsyncIterator[VoiceAgentEvent],
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
    tasks: AgentRuntimeTasks,
) -> None:
    try:
        async for event in event_stream:
            await tasks.queue.put(event)

            if not isinstance(event, STTOutputEvent):
                continue

            should_end = await _process_stt_turn(config, state, tasks, event)
            if should_end:
                break
    finally:
        await _run_final_flush(config, state, tasks)
        if state.end_session_requested:
            await _emit_extraction_status(
                config,
                tasks,
                "completed",
                state.latest_revision,
                is_final=True,
            )
        if state.end_session_requested and state.chat_history:
            await _run_final_report(config, state, tasks)
        tasks.done.set()


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Consume STT events and emit interaction + incremental extraction events."""
    services = get_services()
    llm_interaction = services.get("llm_interaction", services["llm"])
    llm_extraction = services.get("llm_extraction", services["llm"])
    llm_interaction_lock = services.get("llm_interaction_lock")
    llm_extraction_lock = services.get("llm_extraction_lock")
    if llm_extraction is llm_interaction and llm_extraction_lock is None:
        llm_extraction_lock = llm_interaction_lock

    config = AgentRuntimeConfig(
        llm_interaction=llm_interaction,
        llm_extraction=llm_extraction,
        llm_interaction_lock=llm_interaction_lock,
        llm_extraction_lock=llm_extraction_lock,
        extraction_timeout=_get_extraction_timeout_seconds(),
        final_extraction_timeout=_get_final_extraction_timeout_seconds(),
        report_timeout=_get_report_timeout_seconds(),
        report_status_heartbeat_s=_get_report_status_heartbeat_seconds(),
        extraction_debounce_s=_get_extraction_debounce_seconds(),
        max_delta_turns=_get_extraction_max_delta_turns(),
        emit_status_events=_is_extraction_status_enabled(),
        emergency_rule_gate_enabled=_is_emergency_rule_gate_enabled(),
        emergency_escalation_message=_get_emergency_escalation_message(),
        end_session_signal=_get_end_session_signal(),
    )
    state = AgentRuntimeState()
    tasks = AgentRuntimeTasks()

    log_event(
        component="agent_stream",
        event="agent_stream_ready",
        details={
            "extraction_debounce_s": config.extraction_debounce_s,
            "emergency_rule_gate_enabled": config.emergency_rule_gate_enabled,
            "report_timeout_s": config.report_timeout,
            "report_status_heartbeat_s": config.report_status_heartbeat_s,
        },
    )

    tasks.producer_task = asyncio.create_task(
        _produce_events(event_stream, config, state, tasks)
    )

    try:
        while True:
            if tasks.done.is_set() and tasks.queue.empty():
                break

            try:
                event = await asyncio.wait_for(tasks.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield event
    finally:
        await _cancel_task(tasks.producer_task)
        await _cancel_task(tasks.debounce_task)
        await _cancel_task(tasks.extraction_task)
