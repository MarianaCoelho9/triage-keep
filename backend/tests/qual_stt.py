import time
import os
import sys

# Add the backend root to the python path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from huggingface_hub import hf_hub_download
from voice_agent.services.stt import MedASRService, MedASRMLXService

def run_comparison():
    print("=== MedASR Quality & Speed Comparison ===\n")
    
    print("[1] Downloading sample medical audio...")
    audio_path = hf_hub_download(repo_id="google/medasr", filename="test_audio.wav")
    print(f"    Audio downloaded: {audio_path}\n")

    print("[2] Initializing Default Hugging Face MedASR...")
    start_init_hf = time.time()
    hf_service = MedASRService()
    init_time_hf = time.time() - start_init_hf
    print(f"    Init Time: {init_time_hf:.2f}s\n")

    print("[3] Initializing MLX MedASR...")
    start_init_mlx = time.time()
    mlx_service = MedASRMLXService()
    init_time_mlx = time.time() - start_init_mlx
    print(f"    Init Time: {init_time_mlx:.2f}s\n")

    print("--- Transcribing ---")
    
    print("\nRunning Default Hugging Face MedASR...")
    # Run once to warmup (PyTorch/Transformers often slow on first run)
    hf_service.transcribe(audio_path)
    start_infer_hf = time.time()
    hf_text = hf_service.transcribe(audio_path)
    infer_time_hf = time.time() - start_infer_hf
    print(f"Hugging Face Inference Time: {infer_time_hf:.2f}s")
    print(f"Transcript: {hf_text}")

    print("\nRunning MLX MedASR...")
    # Run once to warmup
    mlx_service.transcribe(audio_path)
    start_infer_mlx = time.time()
    mlx_text = mlx_service.transcribe(audio_path)
    infer_time_mlx = time.time() - start_infer_mlx
    print(f"MLX Inference Time: {infer_time_mlx:.2f}s")
    print(f"Transcript: {mlx_text}")
    
    print("\n=== SUMMARY ===")
    print(f"HF Inference Time : {infer_time_hf:.2f}s")
    print(f"MLX Inference Time: {infer_time_mlx:.2f}s")
    speedup = infer_time_hf / infer_time_mlx if infer_time_mlx > 0 else 0
    print(f"Speedup: {speedup:.2f}x faster with MLX")
    
    if hf_text == mlx_text:
        print("Quality: EXACT MATCH")
    else:
        print("Quality: Transcript differs slightly.")

if __name__ == "__main__":
    run_comparison()
