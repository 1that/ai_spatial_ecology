import os
import numpy as np
import json
from datetime import datetime
from gtts import gTTS

from resemblyzer import VoiceEncoder, preprocess_wav

class SpeakerDatabase:
    def __init__(self):
        print("Загрузка модели VoiceEncoder...")
        self.encoder = VoiceEncoder()
        self.embeddings = {}
        self.names = []
        self.audio_paths = {}
        print("Модель загружена")

    def extract_embedding(self, audio_path):
        try:
            wav = preprocess_wav(audio_path)
            embedding = self.encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            print(f"Ошибка при извлечении эмбеддинга: {e}")
            return None

    def add_speaker(self, name, audio_path):
        if not os.path.exists(audio_path):
            print(f"Файл {audio_path} не найден")
            return False
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return False
        self.embeddings[name] = embedding
        self.names.append(name)
        self.audio_paths[name] = audio_path
        print(f"Добавлен диктор: {name} (файл: {os.path.basename(audio_path)})")
        return True

    def identify(self, audio_path, threshold=0.7):
        if not self.embeddings:
            print("База пуста. Сначала добавьте дикторов.")
            return None, 0
        test_embedding = self.extract_embedding(audio_path)
        if test_embedding is None:
            return None, 0
        best_match = None
        best_score = -1
        for name, db_embedding in self.embeddings.items():
            similarity = np.dot(test_embedding, db_embedding)
            if similarity > best_score:
                best_score = similarity
                best_match = name
        if best_score > threshold:
            return best_match, best_score
        else:
            return None, best_score

    def verify(self, audio_path, claimed_name, threshold=0.7):
        if claimed_name not in self.embeddings:
            print(f"Диктор {claimed_name} не найден в базе")
            return False, 0
        test_embedding = self.extract_embedding(audio_path)
        if test_embedding is None:
            return False, 0
        db_embedding = self.embeddings[claimed_name]
        score = np.dot(test_embedding, db_embedding)
        is_match = score > threshold
        return is_match, score

    def get_stats(self):
        return {
            'total_speakers': len(self.names),
            'speakers': self.names,
            'audio_files': self.audio_paths
        }

    def save(self, filepath='voice_results/speaker_database.json'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'speakers': self.names,
            'audio_paths': self.audio_paths,
            'created_at': datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"База сохранена: {filepath}")

os.makedirs('audio_samples', exist_ok=True)
db = SpeakerDatabase()

existing_speakers = [
    ('speaker_female', 'audio_samples/speaker_female.wav'),
    ('speaker_male', 'audio_samples/speaker_male.wav')
]

for name, path in existing_speakers:
    if os.path.exists(path):
        db.add_speaker(name, path)
    else:
        print(f"Файл {path} не найден")

synthetic_speakers = [
    ('inspector_ivanov', 'audio_samples/speaker_ivanov.mp3'),
    ('inspector_petrov', 'audio_samples/speaker_petrov.mp3'),
    ('inspector_sidorova', 'audio_samples/speaker_sidorova.mp3')
]

texts = {
    'inspector_ivanov': 'Докладывает Иванов. Координаты 59.93, 30.25. Аммоний 0.5. Все в норме.',
    'inspector_petrov': 'Петров на связи. Точка 60.1, 29.8. Обнаружена асбестовая труба с повреждением.',
    'inspector_sidorova': 'Сидорова докладывает. Координаты 59.8, 30.3. Разлив нефти, требуется помощь.'
}

for name, path in synthetic_speakers:
    if not os.path.exists(path):
        tts = gTTS(text=texts.get(name, f"Сообщение от {name}"), lang='ru', slow=False)
        tts.save(path)
    db.add_speaker(name, path)

print(f"\nБаза голосов создана. Всего дикторов: {len(db.names)}")
print(f"  Дикторы: {', '.join(db.names)}")

test_files = [
    ('speaker_female.wav', 'speaker_female'),
    ('speaker_male.wav', 'speaker_male'),
    ('speaker_ivanov.mp3', 'inspector_ivanov'),
    ('speaker_petrov.mp3', 'inspector_petrov'),
]

for filename, expected in test_files:
    filepath = f'audio_samples/{filename}'
    if os.path.exists(filepath):
        name, score = db.identify(filepath, threshold=0.6)
        print(f"\nФайл: {filename}")
        print(f"  Ожидаемый: {expected}")
        print(f"  Результат: {name if name else 'НЕ ИЗВЕСТЕН'}")
        print(f"  Уверенность: {score:.4f}")

db.save('voice_results/speaker_database.json')