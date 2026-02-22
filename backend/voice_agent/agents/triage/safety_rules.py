"""Deterministic emergency trigger rules for triage safety checks."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EmergencyRule:
    """Rule definition for deterministic emergency escalation."""

    rule_id: str
    description: str
    required_all: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    confidence: float = 0.9


@dataclass(frozen=True)
class EmergencyTriggerResult:
    """Result payload returned by rule evaluation."""

    triggered: bool
    rule_id: str | None
    confidence: float
    matched_terms: tuple[str, ...]
    reason: str | None = None


_NEGATION_TERMS: tuple[str, ...] = (
    "no",
    "not",
    "without",
    "denies",
    "deny",
    "never",
)

_RULES: tuple[EmergencyRule, ...] = (
    EmergencyRule(
        rule_id="chest_pain_breathing_distress",
        description="Chest pain plus breathing distress",
        required_all=("chest pain",),
        required_any=("shortness of breath", "difficulty breathing", "cannot breathe"),
        confidence=0.95,
    ),
    EmergencyRule(
        rule_id="stroke_signs",
        description="Potential stroke symptoms",
        required_any=(
            "face droop",
            "slurred speech",
            "one sided weakness",
            "one-sided weakness",
            "sudden confusion",
        ),
        confidence=0.95,
    ),
    EmergencyRule(
        rule_id="unresponsive_or_loc",
        description="Loss of consciousness or unresponsive state",
        required_any=(
            "loss of consciousness",
            "lost consciousness",
            "fainted and not responding",
            "unresponsive",
        ),
        confidence=0.98,
    ),
    EmergencyRule(
        rule_id="severe_bleeding",
        description="Severe active bleeding",
        required_any=("severe bleeding", "cannot stop bleeding", "bleeding heavily"),
        confidence=0.97,
    ),
    EmergencyRule(
        rule_id="possible_anaphylaxis",
        description="Airway swelling with breathing compromise",
        required_all=("swelling tongue",),
        required_any=("difficulty breathing", "shortness of breath", "wheezing"),
        confidence=0.97,
    ),
    EmergencyRule(
        rule_id="possible_anaphylaxis_lips",
        description="Lip swelling with breathing compromise",
        required_all=("swelling lips",),
        required_any=("difficulty breathing", "shortness of breath", "wheezing"),
        confidence=0.97,
    ),
)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    normalized_spaces = re.sub(r"\s+", " ", lowered)
    return normalized_spaces.strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    token_pattern = r"\s+".join(re.escape(token) for token in phrase.split())
    return re.compile(rf"\b{token_pattern}\b")


def _is_phrase_negated(normalized_text: str, phrase_match_start: int) -> bool:
    window_start = max(0, phrase_match_start - 40)
    context = normalized_text[window_start:phrase_match_start]
    negation_pattern = re.compile(rf"(?:^|\b)({'|'.join(_NEGATION_TERMS)})\b[^.!,;]{{0,24}}$")
    return bool(negation_pattern.search(context))


def _match_positive_phrase(normalized_text: str, phrase: str) -> bool:
    pattern = _phrase_pattern(phrase)
    for match in pattern.finditer(normalized_text):
        if not _is_phrase_negated(normalized_text, match.start()):
            return True
    return False


def evaluate_emergency_trigger(user_text: str) -> EmergencyTriggerResult:
    """
    Evaluate deterministic emergency triggers against one user utterance.

    The check is intentionally lightweight and does not call any model.
    """
    normalized_text = _normalize_text(user_text)
    if not normalized_text:
        return EmergencyTriggerResult(
            triggered=False,
            rule_id=None,
            confidence=0.0,
            matched_terms=(),
            reason="empty_input",
        )

    for rule in _RULES:
        matched_all: list[str] = []
        matched_any: list[str] = []

        all_ok = True
        for phrase in rule.required_all:
            if _match_positive_phrase(normalized_text, phrase):
                matched_all.append(phrase)
            else:
                all_ok = False
                break
        if not all_ok:
            continue

        if rule.required_any:
            matched_any = [
                phrase for phrase in rule.required_any if _match_positive_phrase(normalized_text, phrase)
            ]
            if not matched_any:
                continue

        matched_terms = tuple([*matched_all, *matched_any])
        return EmergencyTriggerResult(
            triggered=True,
            rule_id=rule.rule_id,
            confidence=rule.confidence,
            matched_terms=matched_terms,
            reason=rule.description,
        )

    return EmergencyTriggerResult(
        triggered=False,
        rule_id=None,
        confidence=0.0,
        matched_terms=(),
        reason=None,
    )
