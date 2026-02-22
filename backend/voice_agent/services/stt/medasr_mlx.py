"""MedASR Speech-to-Text MLX implementation."""
import numpy as np
import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

from .base import BaseSTTModel
from .mlx_engine.model import load_mlx_model
from .mlx_engine.decode import CTCTextDecoder, DecoderConfig

class MedASRMLXService(BaseSTTModel):
    """
    Local STT using Apple MLX-optimized MedASR model.
    """

    def __init__(self, model_id="ainergiz/medasr-mlx-fp16", decode_mode="greedy"):
        print(f"[MedASR-MLX] Downloading/Loading model {model_id}...")
        
        # Download the MLX model snapshot
        self.model_dir = snapshot_download(model_id)
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(f"{self.model_dir}/processor")
        except Exception:
            self.processor = AutoProcessor.from_pretrained("google/medasr")
            
        # Initialize decoder
        config = DecoderConfig(mode=decode_mode, hf_model_id="google/medasr")
        self.decoder = CTCTextDecoder(processor=self.processor, config=config)
        
        # Load MLX model
        self.model = load_mlx_model(self.model_dir)
        print("[MedASR-MLX] Model loaded successfully.")

    def transcribe(self, audio_input) -> str:
        """
        Transcribes audio input using MLX.
        :param audio_input: File path (str) or numpy array (np.ndarray float32).
        """
        try:
            if isinstance(audio_input, str):
                from .mlx_engine.audio_utils import load_audio_mono
                speech, sr = load_audio_mono(audio_input, target_sr=16000)
                speech = speech.astype(np.float32)
            else:
                speech = audio_input
                sr = 16000
                
            features = self.processor(
                speech,
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors="np",
            )
            
            input_features = mx.array(features["input_features"])
            attention_mask = mx.array(features["attention_mask"].astype(np.bool_))

            logits = self.model(input_features=input_features, attention_mask=attention_mask)
            logits_np = np.asarray(logits)
            pred_ids = np.asarray(mx.argmax(logits, axis=-1))
            
            text = self.decoder.decode(logits_np, pred_ids=pred_ids)
            return text.strip()
        except Exception as e:
            print(f"[MedASR-MLX] Error: {e}")
            return ""
