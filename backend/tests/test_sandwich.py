import asyncio
import numpy as np
import soundfile as sf
# We need to create a dummy audio file for the HF pipeline to consume easily
# or teach the user how to pass numpy arrays.

from voice_agent.config import get_services

def create_dummy_audio():
    # Generate 1 sec of silence/noise as a test file
    sr = 16000
    audio = np.random.uniform(-0.1, 0.1, sr)
    sf.write("test_audio.wav", audio, sr)
    return "test_audio.wav"

async def run_sandwich_test():
    print("--- Initializing Services (Download Models) ---")
    services = get_services()
    stt = services["stt"]
    llm = services["llm"]
    
    print("\n--- Starting 'Sandwich' Real Inference Test ---")
    
    # 1. MedASR
    print("\n[STT] Testing MedASR with dummy audio...")
    audio_path = create_dummy_audio()
    # The pipeline handles file paths natively
    transcript = stt.transcribe(audio_path) 
    print(f"[STT Output] Transcript: '{transcript}'")

    # 2. MedGemma
    print("\n[LLM] Testing MedGemma reasoning...")
    val_transcript = "Patient complains of chest pain and left arm numbness."
    prompt = f"Analyze this symptom description and suggest triage urgency: {val_transcript}"
    response_text = llm.generate(prompt)
    print(f"[LLM Output] Response: '{response_text}'")

    # 3. TTS
    print("\n[TTS] Testing TTS synthesis...")
    tts = services["tts"]
    audio_data = tts.synthesize(response_text)
    print(f"[TTS Output] Generated {len(audio_data)} bytes of audio.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(run_sandwich_test())
