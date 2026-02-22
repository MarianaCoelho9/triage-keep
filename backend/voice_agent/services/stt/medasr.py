"""MedASR Speech-to-Text implementation."""
import torch
from transformers import pipeline
from .base import BaseSTTModel


class MedASRService(BaseSTTModel):
    """
    Wrapper for google/medasr (HuggingFace).
    Local STT using Google's MedASR medical speech recognition model.
    """

    def __init__(self):
        print("[MedASR] Loading model...")
        # Note: 'automatic-speech-recognition' pipeline usually expects a file path or numpy array.
        # We will handle raw bytes by converting to numpy in the transcribe method.
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="google/medasr",
            trust_remote_code=True,
            device=device,
        )
        print(f"[MedASR] Model loaded on {device}.")

    def transcribe(self, audio_input) -> str:
        """
        Transcribes audio input.
        :param audio_input: File path (str) or numpy array (np.ndarray).
        """
        try:
            # The pipeline 'automatic-speech-recognition' can handle paths and numpy arrays.
            # If passing bytes directly, we might need a decoder, but for now we prioritize paths/arrays.
            result = self.pipe(audio_input)
            return result.get("text", "")
        except Exception as e:
            print(f"[MedASR] Error: {e}")
            return ""
