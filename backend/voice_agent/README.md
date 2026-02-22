# Voice Agent Architecture

Modular voice triage pipeline used by the backend websocket endpoint.

## Current Module Layout

```text
voice_agent/
├── agents/triage/
│   ├── agent.py                       # Interaction-first agent stream
│   ├── workflows.py                   # Interaction/extraction/report workflow functions
│   ├── safety_rules.py                # Emergency trigger rules
│   ├── utils.py                       # Parsing + JSON extraction helpers
│   └── prompts/                       # Prompt templates
├── config/settings.py                 # Service factory + env resolution
├── core/
│   ├── events.py                      # Typed event dataclasses
│   ├── schemas.py                     # API schemas
│   ├── types.py                       # Shared type aliases
│   ├── error_mapping.py               # Report error normalization
│   └── logging_utils.py               # Structured logs + turn/session context
├── fhir/
│   ├── mapping.py                     # Report -> FHIR Bundle mapping
│   └── validation.py                  # Lightweight bundle validation
├── pipelines/triage_pipeline.py       # Pipeline composition entrypoint
└── services/
    ├── llm/medgemma_llamacpp.py       # llama.cpp MedGemma backend
    ├── stt/{medasr.py,medasr_mlx.py,stream.py}
    └── tts/{kokoro.py,stream.py}
```

## Runtime Flow

1. `stt_stream` consumes audio bytes and waits for `COMMIT` to finalize a turn.
2. `agent_stream` handles each `STTOutputEvent`:
   - generates interaction response first,
   - schedules debounced incremental extraction,
   - emits report status/report events after end-session signal.
3. `tts_stream` converts `AgentChunkEvent` text into `TTSChunkEvent` audio.
4. Pipeline yields a unified event stream used by websocket handlers.

## Event Types

- `STTOutputEvent`: final transcript text for a user turn.
- `AgentChunkEvent`: assistant text.
- `ExtractionEvent`: normalized structured extraction state.
- `ExtractionStatusEvent`: scheduler state (`scheduled`, `running`, `completed`, etc.).
- `ReportStatusEvent`: report lifecycle (`running`, `completed`, `failed`).
- `ReportEvent`: final report envelope (`success`, `data`, `error`).
- `TTSChunkEvent`: synthesized WAV bytes.

`STTChunkEvent` exists but partial transcript emission is disabled in the current pipeline (`emit_partial_transcripts=False`).

## Public Imports

```python
from voice_agent import voice_agent_pipeline, get_services
from voice_agent.core import AgentChunkEvent, ExtractionEvent, TTSChunkEvent
```

## Minimal Example

```python
from voice_agent import voice_agent_pipeline
from voice_agent.core import AgentChunkEvent, ExtractionEvent, TTSChunkEvent

async def consume(audio_stream):
    async for event in voice_agent_pipeline(audio_stream):
        if isinstance(event, AgentChunkEvent):
            print(event.text)
        elif isinstance(event, ExtractionEvent):
            print(event.data)
        elif isinstance(event, TTSChunkEvent):
            await send_audio(event.audio)
```

## Configuration Notes

- Service creation is centralized in `voice_agent/config/settings.py`.
- Interaction and extraction can share or split LLM instances via env vars.
- Thread locks protect llama.cpp calls when workflows run close together.
- End-session report generation is controlled by `MEDGEMMA_SESSION_END_SIGNAL` (default `END_SESSION`).
