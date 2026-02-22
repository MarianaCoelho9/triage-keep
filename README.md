# TriageKeep

AI-assisted voice triage prototype for the MedGemma Impact Challenge.

## Architecture

- `frontend/` (Next.js): Dispatcher UI (voice controls, transcript, extraction dashboard, final report panel).
- `backend/` (FastAPI): Voice/event pipeline and REST APIs.
- `backend/voice_agent/services`:
  - STT: `google/medasr`
  - LLM: MedGemma (llama.cpp)
  - TTS: Kokoro-82M
- Pipeline flow:
  - `Audio -> STT -> interaction/extraction workflows -> TTS -> websocket events`

## API Overview

- `POST /analyze`: returns next assistant question.
- `POST /extract`: returns structured envelope:
  - `{ "success": boolean, "data": {...}, "error": {...}|null }`
- `POST /report`: returns structured envelope:
  - `{ "success": boolean, "data": {...}, "error": {...}|null }`
- `POST /transcribe`, `POST /synthesize`
- `WS /ws/audio`

## Quick Start

### Backend

```bash
cd backend
uv sync
PYTHONPATH=. uv run python main.py
```

If you run `uvicorn` directly, include websocket keepalive tuning:

```bash
PYTHONPATH=. uv run uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

### Interaction Mode

Set `NEXT_PUBLIC_VOICE_INTERACTION_MODE` in `frontend/.env.local`:

- `ptt` (default): tap mic to start and tap again to stop/commit.
- `auto_turn`: hands-free turn commit based on in-browser VAD (silence timeout + cooldown guards).

## Demo Runbook

1. Start backend and frontend.
2. Open the UI and verify websocket status turns `Online`.
3. Click microphone and speak a short triage scenario.
4. Use one of the flows:
   - `ptt`: stop recording to send `COMMIT`.
   - `auto_turn`: pause after speaking and verify commit happens automatically.
5. Verify:
   - transcript messages are displayed,
   - extraction data appears in the left dashboard,
   - assistant audio is played.
6. Click `End Connection` to generate the final report.
7. Verify report renders sections (administrative, patient info, assessment, disposition, plan).
8. Click `Start New Conversation` and repeat.

## Demo Video

[Watch the demo video](docs/triage-keep_demo.mov)

## UI Screenshots

### App Startup

![App startup](docs/app_startup.png)

### Triage UI Example

![Triage UI example](docs/ui_example.png)

## Quality Gates

### Backend

```bash
cd backend
PYTHONPATH=. uv run pytest -q
```

### Frontend

```bash
cd frontend
npm run lint
```

## Safety

See `docs/safety_guardrails.md` for non-diagnostic behavior, escalation rules, and secure configuration requirements.

## Auto-Turn Pilot Docs

- Baseline template: `docs/auto-turn-baseline.md`
- Pilot metrics + go/no-go template: `docs/auto-turn-pilot-results.md`
