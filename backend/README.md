# TriageKeep Backend

FastAPI service for voice triage HTTP APIs and websocket streaming.

## Local Setup

1. Download the **MedGemma 27B GGUF** model file and place it in `backend/models/` (example: `backend/models/medgemma-27b-it-Q3_K_M.gguf`).

2. Configure environment:

```bash
cd backend
cp .env.example .env
```

Set `MEDGEMMA_GGUF_PATH` in `.env` to the model path, for example:

```bash
MEDGEMMA_GGUF_PATH=/absolute/path/to/triage-keep/backend/models/medgemma-27b-it-Q3_K_M.gguf
```

3. Install dependencies:

```bash
uv sync
```

4. Start server:

```bash
PYTHONPATH=. uv run python main.py
```

Default server address: `http://127.0.0.1:8000`.

If running uvicorn directly, include websocket keepalive options:

```bash
PYTHONPATH=. uv run uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
```

## API Endpoints

- `GET /`
  - Health/status (`{ "status": "online", "system": "TriageKeep Brain" }`).
- `POST /analyze`
  - Single interaction turn from `user_input` + `chat_history`.
- `POST /extract`
  - Extraction envelope: `{ success, data, error }`.
- `POST /report`
  - Final report envelope: `{ success, data, error }`.
- `POST /report/fhir`
  - Maps report JSON to FHIR `Bundle` envelope: `{ success, data, error }`.
- `POST /transcribe`
  - Upload audio file and return transcript.
- `POST /synthesize`
  - Synthesize a text query parameter to `audio/wav`.
- `WS /ws/audio`
  - Real-time audio pipeline.

## WebSocket Contract (`/ws/audio`)

### Client -> Server

- Binary PCM audio chunks.
- Text control message: `COMMIT` to finalize current turn.

### Server -> Client

- Binary WAV chunks (assistant TTS output).
- JSON text events:
  - `{ "type": "user", "text": ... }`
  - `{ "type": "agent", "text": ... }`
  - `{ "type": "extraction", "data": {...} }`
  - `{ "type": "extraction_status", "status": "...", "revision": n, "is_final"?: true }`
  - `{ "type": "report_status", "status": "running|completed|failed" }`
  - `{ "type": "report", "success": bool, "data": {...}, "error": {...}|null }`

## Current Runtime Architecture

- App entrypoint: `main.py` + `app_factory.py`.
- HTTP routes: `routes/http.py`.
- Websocket routes: `routes/ws.py`, shared session handling in `routes/ws_shared.py`.
- Voice pipeline: `voice_agent/pipelines/triage_pipeline.py`.
  - `stt_stream` -> `agent_stream` -> `tts_stream`
- Triage agent behavior:
  - interaction-first assistant response
  - debounced incremental extraction updates
  - optional emergency rule gate
  - end-session report generation via `END_SESSION` signal.

## Configuration

### Core Service Selection

- `INFERENCE_PROFILE`: `local` or `space` (default is platform dependent).
- `STT_BACKEND`: `auto`, `mlx`, `torch`.
- `TTS_DEVICE`: `auto`, `cuda`, `mps`, `cpu`.
- `LLM_SERVICE`: `medgemma_llamacpp` (alias `llamacpp` also accepted).
- `LLM_INTERACTION_SERVICE`, `LLM_EXTRACTION_SERVICE`: optional per-workflow override.

### MedGemma / llama.cpp

- `MEDGEMMA_GGUF_PATH` (required).
- `MODEL_PATH` (legacy alias; used only if `MEDGEMMA_GGUF_PATH` is unset).
- `MEDGEMMA_N_CTX` (default `4096`)
- `MEDGEMMA_N_THREADS` (default `0`)
- `MEDGEMMA_N_BATCH` (default `128`)
- `MEDGEMMA_MAX_NEW_TOKENS` (default `320`)
- `MEDGEMMA_GPU_LAYERS` (default `-1`)
- `MEDGEMMA_CONTEXT_MARGIN` (default `64`)
- `MEDGEMMA_ENABLE_JSON_GRAMMAR` (default `true`)

### Agent/Workflow Controls

- `MEDGEMMA_EXTRACTION_TIMEOUT_S` (default `35`)
- `MEDGEMMA_EXTRACTION_FINAL_TIMEOUT_S` (default `60`)
- `MEDGEMMA_EXTRACTION_DEBOUNCE_S` (default `3`)
- `MEDGEMMA_EXTRACTION_MAX_DELTA_TURNS` (default `8`)
- `MEDGEMMA_EXTRACTION_ENABLE_STATUS_EVENTS` (default `true`)
- `MEDGEMMA_ENABLE_EMERGENCY_RULE_GATE` (default `true`)
- `MEDGEMMA_EMERGENCY_ESCALATION_MESSAGE` (optional override)
- `MEDGEMMA_SESSION_END_SIGNAL` (default `END_SESSION`)
- `MEDGEMMA_REPORT_AUTO_TIMEOUT_S` (default `90`)
- `MEDGEMMA_REPORT_STATUS_HEARTBEAT_S` (default `2`)
- `MEDGEMMA_REPORT_HTTP_LOCK_TIMEOUT_S` (default `3.0`)

### Websocket Keepalive

- `MEDGEMMA_WS_PING_INTERVAL_S` (default `60`)
- `MEDGEMMA_WS_PING_TIMEOUT_S` (default `120`)

## Static Assets

The app serves `backend/data/` at `/static`. Example: `backend/data/intro.wav` is available at `/static/intro.wav`.

## Tests

```bash
cd backend
PYTHONPATH=. uv run pytest -q
```
