import whisper
import librosa

# Загрузка модели (base - хороший баланс)
model = whisper.load_model("base")
# Whisper сам читает файлы через ffmpeg, поэтому грузим звук заранее:
# моно, 16 кГц, float32 — именно такой формат ожидает модель
audio, _ = librosa.load("audio_samples/test_short.ogg", sr=16000, mono=True)
# Тестовое распознавание
result = model.transcribe(audio)
print("Распознанный текст:", result["text"])
print("Определенный язык:", result["language"])
