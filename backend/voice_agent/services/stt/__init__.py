"""Speech-to-Text service module."""
from .base import BaseSTTModel
from .medasr import MedASRService
from .stream import stt_stream

try:
    from .medasr_mlx import MedASRMLXService
except Exception:  # pragma: no cover - optional backend on non-Apple environments
    MedASRMLXService = None  # type: ignore[assignment]

__all__ = [
    "BaseSTTModel",
    "MedASRService",
    "MedASRMLXService",
    "stt_stream",
]
