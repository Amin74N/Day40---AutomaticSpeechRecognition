from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import speech_recognition as sr
import torch
import io
# ----------------------------------------------------------------------------------------------------------------------
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
r = sr.Recognizer()
with sr.Microphone(sample_rate=16000) as source:
    print("You can start speaking now...")
    while True:
        audio = r.listen(source)                               # PyAudio Object
        data  = io.BytesIO(audio.get_wav_data())               # List of Bytes
        clip  = AudioSegment.from_file(data)                   # Numpy Array
        x     = torch.FloatTensor(clip.get_array_of_samples()) # Tensor
        inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis=-1)  # Get Data
        text   = tokenizer.batch_decode(tokens) # Tokens Into A String
        print("You Said:", str(text).lower())
