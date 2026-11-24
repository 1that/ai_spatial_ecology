import os
import whisper
import librosa
import json

# Исследование параметров транскрибации
def experiment_parameters(audio_path, model):
    """
    Эксперимент с различными параметрами Whisper
    """
    print(f"\n🔊 Аудиофайл: {os.path.basename(audio_path)}")
    print("-" * 50)
    # Загружаем звук заранее (моно, 16 кГц): Whisper без ffmpeg не читает файлы сам
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    # Базовый случай
    result_base = model.transcribe(audio)
    print(f"\nБазовый результат:")
    print(f"  Текст: {result_base['text'][:100]}...")
    print(f"  Язык: {result_base['language']}")
    # Влияние температуры
    results_temp = {}
    for temp in [0.0, 0.5, 1.0]:
        result = model.transcribe(audio, temperature=temp)
        results_temp[temp] = result['text']
        print(f"  t={temp}: {result['text'][:80]}...")
    # Влияние beam size
    results_beam = {}
    for beam in [1, 3, 5]:
        result = model.transcribe(audio, beam_size=beam)
        results_beam[beam] = result['text']
        print(f"  beam={beam}: {result['text'][:80]}...")
    # Явное указание языка
    try:
        result_ru = model.transcribe(audio, language='ru')
        print(f"  Язык 'ru': {result_ru['text'][:80]}...")
    except:
        print(f"  Язык 'ru': не поддерживается для этого файла")
    try:
        result_en = model.transcribe(audio, language='en')
        print(f"  Язык 'en': {result_en['text'][:80]}...")
    except:
        print(f"  Язык 'en': не поддерживается для этого файла")
    # Временные метки
    result_timestamps = model.transcribe(audio, word_timestamps=True)
    print("  Временные метки слов (первые 5):")
    for segment in result_timestamps['segments'][:3]:
        print(f"    {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text'][:50]}...")
    # Сводная таблица
    print(f"{'Параметр':<20} {'Результат'}")
    print("-" * 50)
    print(f"{'Базовый':<20} {result_base['text'][:50]}...")
    print(f"{'Температура 0.0':<20} {results_temp[0.0][:50]}...")
    print(f"{'Температура 0.5':<20} {results_temp[0.5][:50]}...")
    print(f"{'Температура 1.0':<20} {results_temp[1.0][:50]}...")
    print(f"{'Beam size 1':<20} {results_beam[1][:50]}...")
    print(f"{'Beam size 3':<20} {results_beam[3][:50]}...")
    print(f"{'Beam size 5':<20} {results_beam[5][:50]}...")
    return results_temp, results_beam

# Запуск экспериментов с параметрами...
# Выбираем файлы для экспериментов
experiment_files = []
folder_name = 'audio_samples'
for file_name in os.listdir(folder_name):
    filepath = os.path.join(folder_name, file_name)
    experiment_files.append(filepath)
print(f"Файлы для экспериментов: {[os.path.basename(f) for f in experiment_files]}")
model = whisper.load_model("small")
# Проводим эксперименты для каждого файла
all_results = {}
for filepath in experiment_files:
    results_temp, results_beam = experiment_parameters(filepath, model)
    all_results[os.path.basename(filepath)] = {
        'temperature': results_temp,
        'beam_size': results_beam
    }
# Сохранение результатов экспериментов...
# Сохраняем в папку voice_results
os.makedirs('voice_results', exist_ok=True)
with open('voice_results/parameters_experiment.json', 'w', encoding='utf-8') as f:
    # Преобразуем результаты в сериализуемый формат
    serializable_results = {}
    for filename, res in all_results.items():
        serializable_results[filename] = {
            'temperature': {str(k): v for k, v in res['temperature'].items()},
            'beam_size': {str(k): v for k, v in res['beam_size'].items()}
        }
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
print("Результаты экспериментов сохранены в voice_results/parameters_experiment.json")