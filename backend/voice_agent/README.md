# Voice Agent Architecture

A modular voice-based medical triage system with parallel processing capabilities.

## Overview

The voice agent provides a complete pipeline for medical triage conversations:
- **Speech-to-Text (STT)**: Transcribes patient audio using Google MedASR
- **Agent Processing**: Dual parallel workflows for interaction and data extraction
- **Text-to-Speech (TTS)**: Synthesizes responses using Kokoro-82M
- **Real-time Extraction**: Extracts structured clinical data during conversation

## Architecture

```
voice_agent/
├── core/                      # Core abstractions & event types
│   ├── events.py             # VoiceAgentEvent, STTOutputEvent, ExtractionEvent, etc.
│   ├── pipeline.py           # BasePipeline interface
│   └── types.py              # Common type definitions
│
├── services/                  # I/O Service implementations
│   ├── stt/                  # Speech-to-Text
│   │   ├── base.py           # BaseSTTModel (abstract)
│   │   ├── medasr.py         # MedASR implementation
│   │   └── stream.py         # STT streaming logic
│   ├── tts/                  # Text-to-Speech
│   │   ├── base.py           # BaseTTSModel (abstract)
│   │   ├── kokoro.py         # Kokoro TTS implementation
│   │   └── stream.py         # TTS streaming logic
│   └── llm/                  # Language Models
│       ├── base.py           # BaseLLMModel (abstract)
│       └── medgemma_llamacpp.py # MedGemma llama.cpp model
│
├── agents/                    # Agent workflows & logic
│   └── triage/
│       ├── agent.py          # Main triage agent with parallel workflows
│       ├── workflows.py      # Interaction, extraction, report workflows
│       ├── utils.py          # Response parsing utilities
│       └── prompts/
│           ├── interaction.py # Conversation prompts
│           ├── extraction.py  # Data extraction prompts
│           └── report.py      # Report generation prompts
│
├── pipelines/                 # Pipeline orchestration
│   └── triage_pipeline.py    # Main voice agent pipeline
│
└── config/                    # Configuration management
    └── settings.py           # Service factory and configuration
```

## Key Features

### 1. Dual Workflow Processing
The agent processes each user input through two sequential workflows:
- **Interaction Workflow**: Generates conversational responses for the patient (runs first)
- **Extraction Workflow**: Extracts structured clinical data in real-time (runs second)

**Note**: Workflows run sequentially rather than in parallel due to thread-safety constraints with Transformers pipelines. This ensures stability while maintaining both functionalities.

### 2. Event-Driven Architecture
All components communicate through typed events:
- `STTOutputEvent`: Final transcribed text
- `AgentChunkEvent`: Agent's conversational response
- `ExtractionEvent`: Structured clinical data (JSON)
- `TTSChunkEvent`: Synthesized audio

### 3. Modular Service Layer
Services are abstracted with base classes, making it easy to:
- Swap STT/TTS/LLM providers
- Add new service implementations
- Mock services for testing

## Usage

### Basic Pipeline

```python
from backend.voice_agent import voice_agent_pipeline

async def handle_audio(audio_stream):
    async for event in voice_agent_pipeline(audio_stream):
        if isinstance(event, AgentChunkEvent):
            # Handle agent response
            print(f"Agent: {event.text}")
        elif isinstance(event, ExtractionEvent):
            # Handle extracted clinical data
            print(f"Extracted: {event.data}")
        elif isinstance(event, TTSChunkEvent):
            # Send audio to client
            await websocket.send(event.audio)
```

### Service Configuration

```python
from backend.voice_agent import get_services

# Get all services
services = get_services()
stt = services["stt"]
llm = services["llm"]
tts = services["tts"]

# Or get individual services
from backend.voice_agent import get_llm_service
llm = get_llm_service()
```

### Environment Variables

- `LLM_SERVICE`: Choose LLM provider (`medgemma_llamacpp` or `llamacpp`)
  - Default: `medgemma_llamacpp`

## Event Flow

```
Audio Input
    ↓
[STT Stream] → STTOutputEvent
    ↓
[Agent Stream] → Parallel Processing:
    ├─ Interaction Workflow → AgentChunkEvent
    └─ Extraction Workflow → ExtractionEvent
    ↓
[TTS Stream] → TTSChunkEvent
    ↓
Audio Output
```

## Workflows

### Interaction Workflow
**Purpose**: Conduct medical triage conversation  
**Input**: User message + chat history  
**Output**: Next triage question  
**Prompt**: `SIMPLE_TRIAGE_PROMPT`

Follows medical triage protocol:
1. Identify main complaint
2. Gather critical information (duration, severity, symptoms)
3. Screen for red flags
4. Provide guidance

### Extraction Workflow
**Purpose**: Extract structured clinical data  
**Input**: Complete chat history  
**Output**: JSON with clinical entities  
**Prompt**: `TRIAGE_EXTRACTION_PROMPT`

Extracts:
- Main complaint
- Additional symptoms
- Medical history
- Severity/risk level (low/medium/high)

### Report Workflow
**Purpose**: Generate comprehensive triage report  
**Input**: Complete chat history  
**Output**: Structured JSON report  
**Prompt**: `TRIAGE_REPORT_PROMPT`

Includes:
- Administrative data
- Patient demographics
- Clinical assessment
- Triage disposition
- Care plan

## Migration from V1

### Import Changes

**Before:**
```python
from backend.voice_agent.pipeline_v2 import voice_agent_pipeline_v2
```

**After:**
```python
from backend.voice_agent import voice_agent_pipeline
```

### Service Access Changes

**Before:**
```python
from services import get_services
```

**After:**
```python
from backend.voice_agent.config import get_services
```

## Development

### Adding a New Service

1. Create base class in `services/{type}/base.py`
2. Implement service in `services/{type}/{name}.py`
3. Register in `config/settings.py`
4. Export in `services/{type}/__init__.py`

### Adding a New Agent

1. Create agent directory in `agents/{agent_name}/`
2. Implement agent logic in `agent.py`
3. Add workflows in `workflows.py`
4. Create prompts in `prompts/`
5. Export in `agents/{agent_name}/__init__.py`

### Testing

Services can be easily mocked for testing:

```python
class MockLLM(BaseLLMModel):
    def generate(self, prompt: str) -> str:
        return "Mock response"

services = {"llm": MockLLM(), ...}
```

## Performance

- **Sequential Processing**: Interaction and extraction run one after another (LLM thread-safety)
- **Async I/O**: Non-blocking audio streaming
- **Thread Offloading**: CPU-intensive tasks (STT, LLM, TTS) run in thread pool
- **Resource Safety**: Single LLM lock prevents concurrent model access issues

## Version

Current version: **2.0.0**

### Changelog

**v2.0.0** (Current)
- Complete modular reorganization
- Dual workflow processing (interaction + extraction)
- Improved service abstraction
- Better separation of concerns
- Consolidated from V1 and V2
- Thread-safe LLM access with async locks

**v1.0.0** (Deprecated)
- Single-threaded agent processing
- Monolithic service file
- No real-time extraction
