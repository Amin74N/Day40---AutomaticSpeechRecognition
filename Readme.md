# Wav2Vec2 Speech Recognition

## Frameworks & Libraries
- Python
- PyTorch
- Hugging Face Transformers
- Wav2Vec2
- Librosa
- SoundFile
- pathlib

## Dataset
- Custom WAV audio files

## Concepts
- Automatic Speech Recognition (ASR)
- Wav2Vec2
- Self-Supervised Learning
- Hugging Face Transformers
- Hugging Face Hub
- Transformers Pipeline
- Wav2Vec2 Processor
- CTC (Connectionist Temporal Classification)
- Logits
- Greedy Decoding
- Sampling Rate
- Resampling
- Audio Preprocessing
- Speech-to-Text
- Inference
- Disabling Gradient Calculation
- Model Evaluation Mode
- Execution Time Measurement
- Pipeline vs Direct Model Inference Comparison
- Experiment Logging

## Learning Outcomes
- Explaining the basic purpose of Automatic Speech Recognition
- Explaining what Wav2Vec2 is and how it is used for ASR
- Understanding the basic idea behind Self-Supervised Learning in speech models
- Loading pretrained models from the Hugging Face Hub
- Using the Hugging Face Transformers Pipeline for speech recognition
- Loading and preprocess audio using Librosa
- Inspecting an audio file's original sampling rate
- Resampling audio to 16 kHz when required by the model
- Using Wav2Vec2Processor to prepare audio for inference
- Running Wav2Vec2 directly using PyTorch
- Explaining the role of logits in model prediction
- Explaining the basic idea of CTC decoding
- Converting model predictions into text
- Comparing Pipeline-based inference with direct model inference
- Measuring and compare inference execution times
- Saving transcripts and experiment metadata to a results file
- Explaining the major differences between Wav2Vec2 and Whisper at a high level
- Building a complete Audio → ASR → Text pipeline using pretrained models
