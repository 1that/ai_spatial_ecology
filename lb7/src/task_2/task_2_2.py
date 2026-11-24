import os
from gtts import gTTS
import json
from datetime import datetime

os.makedirs('audio_generation', exist_ok=True)
os.makedirs('voice_results', exist_ok=True)

def create_test_voice_report(text, filename="test_voice.mp3", lang='ru', output_dir='audio_generation'):
    filepath = os.path.join(output_dir, filename)
    tts = gTTS(text=text, lang=lang, slow=False)
    try:
        tts.save(filepath)
    except Exception as e:
        print(f"Ошибка при создании {filepath}: {e}")
        return None
    size = os.path.getsize(filepath) / 1024
    print(f"Создан аудиофайл: {filepath} ({size:.1f} КБ)")
    print(f"  Текст: {text[:80]}...")
    return filepath

test_reports = [
    "Проба воды в точке 59.93, 30.25. Аммоний 0.5 миллиграмма на литр, фосфаты 0.3, pH 7.2",
    "Обнаружена асбестовая труба с повреждением 30 процентов, координаты 60.1, 29.8",
    "Разлив нефти в точке 59.8, 30.3, срочно требуется помощь, объем около 2 кубометров",
    "Проба из реки Нева, координаты 59.95, 30.35. Аммоний 0.8, фосфаты 0.6, pH 6.8, температура воды 12 градусов",
    "Бетонная труба с трещиной, координаты 59.88, 30.42, повреждение 15 процентов",
    "Точка мониторинга номер 7, координаты 60.0, 30.0. Кислород 7.2, БПК5 2.1, превышений не обнаружено",
    "Внимание, загрязнение воды в точке 59.91, 30.28. Аммоний 2.5, нитриты 0.2, требуется проверка",
    "Металлическая труба с коррозией, координаты 60.05, 29.95, повреждение 40 процентов, требуется замена"
]

created_files = []
for i, report in enumerate(test_reports, 1):
    filename = f"test_eco_report_{i:02d}.mp3"
    filepath = create_test_voice_report(report, filename, lang='ru')
    created_files.append({
        'id': i,
        'filename': filename,
        'text': report,
        'path': filepath
    })
print(f"\nСоздано {len(created_files)} тестовых аудиофайлов")

eng_reports = [
    "Water sample at coordinates 59.93, 30.25. Ammonium 0.5, phosphates 0.3, pH 7.2",
    "Asbestos pipe with 30 percent damage detected at 60.1, 29.8",
    "Oil spill at point 59.8, 30.3, urgent assistance required",
    "Concrete pipe with crack at 59.88, 30.42, damage 15 percent"
]
eng_files = []
for i, report in enumerate(eng_reports, 1):
    filename = f"test_eco_report_en_{i:02d}.mp3"
    filepath = create_test_voice_report(report, filename, lang='en')
    eng_files.append({
        'id': i,
        'filename': filename,
        'text': report,
        'path': filepath
    })
print(f"Создано {len(eng_files)} англоязычных тестовых файлов")

short_reports = [
    "Координаты 59.93, 30.25, аммоний 0.5",
    "Асбестовая труба, повреждение 30 процентов",
    "Разлив нефти, точка 59.8, 30.3, срочно",
    "Бетонная труба, трещина",
    "Проба воды, pH 7.2, кислород 6.5"
]
short_files = []
for i, report in enumerate(short_reports, 1):
    filename = f"test_short_eco_{i:02d}.mp3"
    filepath = create_test_voice_report(report, filename, lang='ru')
    short_files.append({
        'id': i,
        'filename': filename,
        'text': report,
        'path': filepath
    })
print(f"Создано {len(short_files)} коротких тестовых файлов")

all_files = {
    'full_reports': created_files,
    'english_reports': eng_files,
    'short_reports': short_files,
    'total_files': len(created_files) + len(eng_files) + len(short_files),
    'generated_at': datetime.now().isoformat()
}
with open('voice_results/generated_audio_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(all_files, f, ensure_ascii=False, indent=2)
print("Метаданные сохранены в voice_results/generated_audio_metadata.json")

print(f"\nПапка: audio_generation/")
print(f"  Полные отчеты (рус.): {len(created_files)} файлов")
for f in created_files:
    print(f"    - {f['filename']}")
print(f"\n  Англоязычные отчеты: {len(eng_files)} файлов")
for f in eng_files:
    print(f"    - {f['filename']}")
print(f"\n  Короткие отчеты: {len(short_files)} файлов")
for f in short_files:
    print(f"    - {f['filename']}")