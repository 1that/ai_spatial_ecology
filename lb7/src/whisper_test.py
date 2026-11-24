import whisper
import librosa

model = whisper.load_model("base")
audio, _ = librosa.load("audio_samples/test_short.ogg", sr=16000, mono=True)
result = model.transcribe(audio)
print("Распознанный текст:", result["text"])
print("Определенный язык:", result["language"])
