# Safety Guardrails

## Intended Use
- TriageKeep is a decision-support assistant for call handlers.
- It is not a diagnostic device and does not replace clinician judgment.

## Mandatory Disclaimer Behavior
- The system must avoid presenting outputs as a final diagnosis.
- High-risk recommendations must include escalation language (for example: call emergency services now).

## Escalation Rules
- Immediately escalate when conversation includes severe chest pain, severe breathing difficulty, altered consciousness, stroke-like signs, or heavy bleeding.
- If uncertainty is high or data is incomplete, classify risk conservatively and request urgent human review.

## Data Handling
- Treat all conversation content as sensitive medical data.
- Do not hardcode credentials in source code.
- Keep API keys in environment variables and fail fast when required secrets are missing.

## Output Validation
- Model output is untrusted until parsed and validated.
- `/extract` and `/report` must return explicit success/error envelopes.
- On parse failure, return structured errors instead of malformed JSON.

## Human-in-the-Loop
- Operators must be able to review transcript, extracted entities, and final report before action.
- The UI should clearly distinguish user transcript, model questions, and extracted structured fields.
