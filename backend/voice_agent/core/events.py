from dataclasses import dataclass
from typing import Literal
from typing import Any

@dataclass
class VoiceAgentEvent:
    """Base class for all voice agent events."""
    pass

@dataclass
class STTChunkEvent(VoiceAgentEvent):
    """Partial transcript event."""
    text: str
    is_final: bool = False

    @property
    def payload(self):
        return {"transcript": self.text, "is_final": self.is_final}

@dataclass
class STTOutputEvent(VoiceAgentEvent):
    """Final transcript event."""
    text: str
    turn_id: int | None = None

    @property
    def payload(self):
        return {"transcript": self.text, "is_final": True}

@dataclass
class AgentChunkEvent(VoiceAgentEvent):
    """Text chunk from the AI agent."""
    text: str
    turn_id: int | None = None

@dataclass
class TTSChunkEvent(VoiceAgentEvent):
    """Audio chunk from TTS."""
    audio: bytes

@dataclass
class ExtractionEvent(VoiceAgentEvent):
    """Structured data extraction event."""
    data: dict[str, Any]


@dataclass
class ExtractionStatusEvent(VoiceAgentEvent):
    """Extraction scheduler status event."""

    status: Literal[
        "scheduled",
        "running",
        "stale_discarded",
        "completed",
        "timed_out",
    ]
    revision: int
    is_final: bool = False


@dataclass
class ReportStatusEvent(VoiceAgentEvent):
    """Final report generation status event."""

    status: Literal["running", "completed", "failed"]


@dataclass
class ReportEvent(VoiceAgentEvent):
    """Final report envelope event."""

    success: bool
    data: dict[str, Any]
    error: dict[str, Any] | None = None
