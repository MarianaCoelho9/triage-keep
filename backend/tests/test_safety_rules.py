from voice_agent.agents.triage import evaluate_emergency_trigger


def test_trigger_chest_pain_with_breathing_distress():
    result = evaluate_emergency_trigger(
        "I have chest pain and shortness of breath for 20 minutes."
    )

    assert result.triggered is True
    assert result.rule_id == "chest_pain_breathing_distress"
    assert result.confidence >= 0.9
    assert "chest pain" in result.matched_terms


def test_no_trigger_when_high_risk_terms_are_negated():
    result = evaluate_emergency_trigger(
        "I have a headache, no chest pain and no shortness of breath."
    )

    assert result.triggered is False
    assert result.rule_id is None


def test_trigger_stroke_signs():
    result = evaluate_emergency_trigger(
        "My father has slurred speech and sudden confusion."
    )

    assert result.triggered is True
    assert result.rule_id == "stroke_signs"
    assert any(term in result.matched_terms for term in ("slurred speech", "sudden confusion"))


def test_trigger_anaphylaxis_combo():
    result = evaluate_emergency_trigger(
        "She has swelling lips and difficulty breathing after peanuts."
    )

    assert result.triggered is True
    assert result.rule_id == "possible_anaphylaxis_lips"
    assert "swelling lips" in result.matched_terms


def test_non_emergency_sentence_does_not_trigger():
    result = evaluate_emergency_trigger(
        "I have mild seasonal allergies and a runny nose."
    )

    assert result.triggered is False
    assert result.rule_id is None


def test_empty_input_does_not_trigger():
    result = evaluate_emergency_trigger("  ")

    assert result.triggered is False
    assert result.reason == "empty_input"
