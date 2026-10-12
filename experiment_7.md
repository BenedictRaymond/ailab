# Experiment 7: Speech-to-Text Conversion Using Wav2Vec2 and PyTorch

Ex.No:7        Perform Speech to Text conversion using Word2Vec and pytorch framework and
               input a .wav audio file to convert to text data.

Date      :

Aim:

To perform Speech to Text conversion using Word2Vec and pytorch framework and input a .wav
audio file to convert to text data.

Procedure:

Step1: Preprocess the Audio Data

Step2: Build/Load the Speech Recognition Model

Step3: Train the Model (or Load Pre-trained Model)

Step4: Feed the Processed Audio to the Model

Step5: Post-process and Output the Text

## Program

```python
pip install transformers torch torchaudio

import torch

import torchaudio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

In [5]:

# Load pre-trained Wav2Vec2.0 model and tokenizer from Hugging Face

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load and preprocess the audio file

def load_audio(file_path):

  waveform, sample_rate = torchaudio.load(file_path)

  return waveform, sample_rate

# Preprocess the waveform

def preprocess_audio(waveform, sample_rate):

  # If sample rate is not 16kHz, resample it

  if sample_rate != 16000:

     transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

     waveform = transform(waveform)

  return waveform

In [7]:

# Convert audio to text using the Wav2Vec2.0 model

def speech_to_text(wav_file):

  waveform, sample_rate = load_audio(wav_file)

  waveform = preprocess_audio(waveform, sample_rate)

  # Tokenize and feed to model

  inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding=True)

  with torch.no_grad():

     logits = model(inputs.input_values).logits

  # Decode the output logits into text

  predicted_ids = torch.argmax(logits, dim=-1)

  transcription = tokenizer.decode(predicted_ids[0])

  return transcription

In [8]:

# Provide the path to your WAV file

wav_file = 'harvard.wav'

transcription = speech_to_text(wav_file)

print(f"Transcription: {transcription}")
```

## Output and Result

Output:

Transcription: THE STALE SMELL OF OLD BEER LINGERS IT TAKES HEAT TO BRING OUT
THE ODOR A COLD DIP RESTORES HEALTH AND ZEST A SALT PICKLE TASTES FINE
WITH HAM TAKOZ AL PASTORE ARE MY FAVORITE A ZESTFUL FOOD IS THE HOT
CROSS BUN

Result: Thus, the Python program to perform Speech to Text conversion using Word2Vec and pytorch
framework and input a .wav audio file to convert to text data was verified successful
