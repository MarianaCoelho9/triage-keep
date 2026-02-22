# TriageKeep Backend

## Voice Agent Components

### Speech-to-Text (STT) Agent
**File**: [`backend/voice_agent/stt.py`](voice_agent/stt.py)
*   **Purpose**: Converts continuous audio input into text events.
*   **Model**: Uses a local instance of **MedASR** (`google/medasr`) via `MedASRService`.
*   **Mechanism**:
    *   Implements an async generator `stt_stream` that consumes a stream of audio bytes.
    *   **Buffering**: It accumulates audio chunks into a buffer.
    *   **Transcription**: Periodically runs the `MedASR` model on the accumulated buffer (simulating streaming since the model itself is not streaming-native).
*   **Output**: Yields `STTOutputEvent` containing the transcribed text.

### Text-to-Speech (TTS) Agent
**File**: [`backend/voice_agent/tts.py`](voice_agent/tts.py)
*   **Purpose**: Converts text chunks from the agent into audio output.
*   **Model**: Uses the **Kokoro-82M** model via `TTSService` (in `services.py`).
*   **Mechanism**:
    *   Implements an async generator `tts_stream` that listens for `AgentChunkEvent`s in the event stream.
    *   **Synthesis**: When text is received, it offloads the synthesis task to a separate thread (using `asyncio.to_thread`) to prevent blocking the event loop while the model generates audio.
*   **Output**: Yields `TTSChunkEvent` containing the synthesized audio bytes (WAV format).

### Agent (The "Brain")
**File**: [`backend/voice_agent/agent.py`](voice_agent/agent.py)
*   **Purpose**: Semantic routing and response generation.
*   **Model**: Uses **MedGemma** via `MedGemmaLlamaCppService` (llama.cpp + GGUF).
*   **Mechanism**:
    *   Implements `agent_stream` which consumes `STTOutputEvent`s.
    *   Maintains conversation history.
    *   Queries the LLM and yields `AgentChunkEvent`s with the response text.

### Voice Pipeline
**File**: [`backend/voice_agent/pipeline.py`](voice_agent/pipeline.py)
*   **Purpose**: Chains the three components into a single stream.
*   **Flow**: `Audio Stream` -> `stt_stream` -> `agent_stream` -> `tts_stream` -> `Output Events`.

### WebSocket Endpoint
**File**: [`backend/main.py`](main.py)
*   **Endpoint**: `/ws/audio`
*   **Functionality**:
    *   Accepts a WebSocket connection.
    *   Receives audio chunks (bytes) from the client.
    *   Passes them through the `voice_agent_pipeline`.
    *   Sends typed events back to the client:
        *   audio chunks (`TTSChunkEvent`)
        *   transcript updates (`STTOutputEvent`)
        *   agent replies (`AgentChunkEvent`)
        *   extraction/report events (`ExtractionEvent`, `ExtractionStatusEvent`, `ReportEvent`, `ReportStatusEvent`)

### MedGemma Runtime Options
**Supported LLM backend**: `LLM_SERVICE=medgemma_llamacpp` (or `llamacpp`)
*   Requires a local GGUF file. Set `MEDGEMMA_GGUF_PATH=/absolute/path/to/model.gguf`.
*   If you only have the Hugging Face weights, convert to GGUF using llama.cpp's
    `convert-hf-to-gguf.py` script and then point `MEDGEMMA_GGUF_PATH` at the output.
*   Optional tuning:
    *   `MEDGEMMA_N_CTX` (default 4096)
    *   `MEDGEMMA_N_THREADS` (default 0 = auto)
    *   `MEDGEMMA_N_BATCH` (default 256)
    *   `MEDGEMMA_MAX_NEW_TOKENS` (default 256)
    *   `MEDGEMMA_GPU_LAYERS` (default -1 = full offload on Apple Silicon)

### Architecture
All agents follow the **LangChain Voice Agent** pattern, operating as independent async generators that are chained together in `voice_agent_pipeline`.
