import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

print("\nЗагрузка модели Whisper...")
model_whis = whisper.load_model("small")
print("Модель Whisper загружена (размер: small)")

def plot_audio_with_transcription(audio_path, model):
    """
    Визуализация аудиосигнала и спектрограммы с текстом распознавания
    """
    # Загружаем аудио
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    # Распознаем (передаем массив, а не путь — иначе Whisper ищет ffmpeg)
    result = model.transcribe(audio)
    # Визуализируем
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # Осциллограмма
    ax1.plot(audio)
    ax1.set_title(f'Аудиосигнал: {os.path.basename(audio_path)}', fontsize=12)
    ax1.set_xlabel('Отсчет')
    ax1.set_ylabel('Амплитуда')
    ax1.grid(True, alpha=0.3)
    # Спектрограмма
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    ax2.set_title('Спектрограмма', fontsize=12)
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    plt.suptitle(f'Распознанный текст: {result["text"][:100]}...', fontsize=10, style='italic')
    plt.tight_layout()
    # Получаем имя файла без расширения
    base_name = os.path.basename(audio_path)[:-4]
    # Создаем папку для этого файла
    os.makedirs(f'voice_results/{base_name}', exist_ok=True)
    # Сохраняем в подпапку
    save_name = f'voice_results/{base_name}/spectrogram.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nРезультат распознавания:")
    print(f"  Текст: {result['text']}")
    print(f"  Язык: {result['language']}")
    print(f"  Длительность: {result['segments'][-1]['end'] if result['segments'] else 0:.2f} сек")
    return result

print("\nТестирование модели на аудиофайлах...")
print("=" * 60)

results = {}
folder_name = 'audio_samples'
for f in os.listdir(folder_name):
    filepath = os.path.join(folder_name, f)
    if os.path.exists(filepath):
        print(f"\nОбработка: {filepath}")
        print("-" * 40)
        try:
            result = plot_audio_with_transcription(filepath, model_whis)
            results[filepath] = result
        except Exception as e:
            print(f"  Ошибка при обработке {filepath}: {e}")
    else:
        print(f"\nФайл {filepath} не найден, пропускаем")

# Сравнительный анализ результатов
for file, result in results.items():
    print(f"\n{file}:")
    print(f"  Язык: {result['language']}")
    print(f"  Длина текста: {len(result['text'])} символов")
    print(f"  Количество сегментов: {len(result['segments'])}")
    print(f"  Пример текста: {result['text'][:100]}...")

# Сохранение результатов
import json
os.makedirs('voice_results', exist_ok=True)
with open('voice_results/transcription_results.json', 'w', encoding='utf-8') as f:
    serializable_results = {}
    for file, result in results.items():
        serializable_results[file] = {
            'text': result['text'],
            'language': result['language'],
            'segments': [
                {'start': s['start'], 'end': s['end'], 'text': s['text']}
                for s in result['segments']
            ]
        }
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
