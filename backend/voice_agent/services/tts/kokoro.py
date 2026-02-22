"""Kokoro TTS implementation."""
import os
import platform
import torch
import numpy as np
import io
import soundfile as sf
from .base import BaseTTSModel


class TTSService(BaseTTSModel):
    """
    TTS Service using hexgrad/Kokoro-82M.
    """

    def __init__(self, device: str = "auto"):
        # macOS specific fix for espeak-ng via phonemizer/espeakng-loader
        if platform.system() == "Darwin":
            # Common Homebrew paths for espeak-ng
            lib_path = "/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib"
            data_path = "/opt/homebrew/share/espeak-ng-data"

            if os.path.exists(lib_path):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
                print(f"[TTS] Configured PHONEMIZER_ESPEAK_LIBRARY: {lib_path}")

            if os.path.exists(data_path):
                os.environ["ESPEAK_DATA_PATH"] = data_path
                print(f"[TTS] Configured ESPEAK_DATA_PATH: {data_path}")

        from kokoro import KPipeline

        print("[TTS] Loading Kokoro-82M...")
        resolved_device = self._resolve_device(device)
        self.pipeline = KPipeline(lang_code="a", device=resolved_device)
        print(f"[TTS] Model loaded on {resolved_device}.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").strip().lower()
        if normalized not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(
                "TTS device must be one of: auto, cuda, mps, cpu."
            )

        if normalized == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if normalized == "cuda" and not torch.cuda.is_available():
            print("[TTS] CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        if normalized == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not has_mps:
                print("[TTS] MPS requested but unavailable. Falling back to CPU.")
                return "cpu"
        return normalized

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text using Kokoro-82M.
        
        :param text: Text to convert to speech
        :return: WAV audio bytes
        """
        generator = self.pipeline(text, voice="af_heart", speed=1.0)
        audio_chunks = [audio for _, _, audio in generator if audio is not None]

        if not audio_chunks:
            return b""

        full_audio = np.concatenate(audio_chunks)
        buffer = io.BytesIO()
        # Kokoro-82M default sample rate is 24000
        sf.write(buffer, full_audio, 24000, format="WAV")
        return buffer.getvalue()
