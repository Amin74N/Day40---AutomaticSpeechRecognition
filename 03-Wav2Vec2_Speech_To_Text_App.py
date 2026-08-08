from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from pathlib import Path
import librosa
import torch
import time
# ----------------------------------------------------------------------------------------------------------------------
AUDIO_DIR = Path("audio")
AUDIO_FILE = AUDIO_DIR / "English.wav"
MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000
RESULTS_FILE = Path("results.txt")
# ----------------------------------------------------------------------------------------------------------------------
# 1)Load Model:
def load_model():
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    asr_pipeline = pipeline("automatic-speech-recognition", model=MODEL_NAME)
    print("Model loaded successfully")
    print("--------------------------------------------------------------------------------")
    return processor, model, asr_pipeline
# ----------------------------------------------------------------------------------------------------------------------
# 2)Load & Prepare Audio:
def load_audio(audio_path):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    print(f"Audio {audio_path} loaded successfully")
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    original_sample_rate = sample_rate
    print(f"Original sample rate: {original_sample_rate} Hz")
    duration = len(audio) / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")
    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"Resampling audio: {sample_rate} Hz → {TARGET_SAMPLE_RATE} Hz")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    print("--------------------------------------------------------------------------------")
    return audio, original_sample_rate, sample_rate, duration
# ----------------------------------------------------------------------------------------------------------------------
# 3)Speech Recognition using Pipeline:
def transcribe_with_pipeline(audio, sample_rate, asr_pipeline):
    start_time = time.perf_counter()
    result = asr_pipeline({"raw": audio,
                           "sampling_rate": sample_rate})
    execution_time = time.perf_counter() - start_time
    transcript = result["text"]
    return transcript, execution_time
# ----------------------------------------------------------------------------------------------------------------------
# 4)Speech Recognition using Wav2Vec2 Directly:
def transcribe_with_model(audio, sample_rate, processor, model):
    start_time = time.perf_counter()
    inputs = processor(audio,sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    execution_time = time.perf_counter() - start_time
    return transcript, execution_time
# ----------------------------------------------------------------------------------------------------------------------
# 5)Compare Results:
def compare_results(pipeline_text, pipeline_time, model_text, model_time):
    print("Pipeline:")
    print(pipeline_text)
    print(f"Execution time: {pipeline_time:.4f} seconds")
    print("Direct Wav2Vec2:")
    print(model_text)
    print(f"Execution time: {model_time:.4f} seconds")
    print("Transcript Match:", pipeline_text.strip().lower() == model_text.strip().lower())
    print("--------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
# 6)Save Results:
def save_results(audio_path, original_sample_rate, final_sample_rate, duration, pipeline_text, pipeline_time, model_text, model_time):
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("Audio Information:\n")
        file.write(f"File: {audio_path}\n")
        file.write(f"Original Sample Rate: {original_sample_rate} Hz\n")
        file.write(f"Final Sample Rate: {final_sample_rate} Hz\n")
        file.write(f"Duration: {duration:.2f} seconds\n")
        file.write("------------------------------------------------------------------------------------------------\n")
        file.write("Model Information:\n")
        file.write(f"Model: {MODEL_NAME}\n")
        file.write("------------------------------------------------------------------------------------------------\n")
        file.write("Pipeline Result:\n")
        file.write(f"Transcript:\n{pipeline_text}\n")
        file.write(f"Execution Time: {pipeline_time:.4f} seconds\n")
        file.write("------------------------------------------------------------------------------------------------\n")
        file.write("Direct Wav2Vec2 Result:\n")
        file.write(f"Transcript:\n{model_text}\n")
        file.write(f"Execution Time: {model_time:.4f} seconds\n")
        file.write("------------------------------------------------------------------------------------------------\n")
        file.write("Comparison:\n")
        file.write(f"Transcript Match: {pipeline_text.strip().lower() == model_text.strip().lower()}\n")
    print("Results saved successfully")
# ----------------------------------------------------------------------------------------------------------------------
# 7)Main:
processor, model, asr_pipeline = load_model()
audio, original_sample_rate, sample_rate, duration = load_audio(AUDIO_FILE)
pipeline_text, pipeline_time = transcribe_with_pipeline(audio, sample_rate, asr_pipeline)
model_text, model_time = transcribe_with_model(audio, sample_rate, processor, model)
compare_results(pipeline_text, pipeline_time, model_text, model_time)
save_results(AUDIO_FILE, original_sample_rate, sample_rate, duration, pipeline_text, pipeline_time, model_text, model_time)
