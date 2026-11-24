# Извлечение структурированных данных
import re
import json
from datetime import datetime
import os

def parse_voice_report(text):
    """
    Извлечение структурированных данных из голосового отчета эколога
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'coordinates': None,
        'measurements': {},
        'material': None,
        'damage': None,
        'problems': [],
        'raw_text': text
    }
    # Извлечение координат
    coord_pattern = r'(\d{2}\.\d{1,6})[,\s]+(\d{2}\.\d{1,6})'
    coord_match = re.search(coord_pattern, text)
    if coord_match:
        result['coordinates'] = {
            'lat': float(coord_match.group(1)),
            'lon': float(coord_match.group(2))
        }
    # Извлечение показателей
    # Число: целое или дробное, с точкой или запятой (Whisper иногда пишет "0,5")
    num = r'\s*[:\s]*(\d+(?:[.,]\d+)?)'
    # [а-я]* — любое окончание слова ("нитриты", "нитритов", "аммония" и т.д.)
    patterns = {
        'NH4': r'(?:амм?они[а-я]*|\bNH4)' + num,
        'NO2': r'(?:нитрит[а-я]*|\bNO2)' + num,
        'NO3': r'(?:нитрат[а-я]*|\bNO3)' + num,
        'PO4': r'(?:фосфат[а-я]*|фосфор[а-я]*|\bPO4)' + num,
        'pH': r'(?:(?<![a-zа-я])pH|пэ-?аш|п-?аш)' + num,
        # \b не дает найти "O2" внутри "NO2"
        'oxygen': r'(?:кислород[а-я]*|\bO2)' + num,
        # "БПК5" / "БПК 5." — цифра 5 из названия не должна попасть в значение
        'BOD5': r'\bБПК\s*5?\.?' + num,
        'oil': r'(?:нефтепродукт[а-я]*|нефт[ьи])' + num
    }
    for param, pattern in patterns.items():
        # IGNORECASE вместо text.lower(): иначе шаблоны с заглавными
        # буквами (NH4, pH, БПК) никогда не совпадали бы
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result['measurements'][param] = float(match.group(1).replace(',', '.'))
    # Извлечение материала
    materials = ['асбест', 'пнд', 'полиэтилен', 'бетон', 'керамика', 'металл', 'чугун', 'геотекстиль']
    for material in materials:
        if material in text.lower():
            result['material'] = material
            break
    # Извлечение повреждений
    # Учитываем и "30%", и "30 процентов"
    damage_match = re.search(r'(\d+)\s*(?:%|процент)', text, re.IGNORECASE)
    if damage_match:
        result['damage'] = int(damage_match.group(1))
    # Поиск проблем: по основе слова, чтобы находить любые формы ("трещины", "утечку")
    problems = {'разлив': 'разлив', 'утечк': 'утечка', 'трещин': 'трещина',
                'корроз': 'коррозия', 'загрязн': 'загрязнение'}
    for stem, problem in problems.items():
        if stem in text.lower():
            result['problems'].append(problem)
    return result

# Код ниже выполняется только при прямом запуске скрипта,
# а при импорте parse_voice_report из другого файла пропускается
if __name__ == '__main__':
    # Загрузка результатов распознавания из Этапа 1...
    # Путь к файлу с результатами транскрибации
    results_file = 'voice_results/transcription_results.json'
    # Разбираем только собственные голосовые отчеты
    report_files = ['otchet_female.mp3', 'otchet_male.mp3']
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        transcription_results = {
            path: data for path, data in all_results.items()
            if os.path.basename(path) in report_files
        }
        print(f"Загружены результаты для {len(transcription_results)} аудиофайлов")
        # Показываем, какие файлы загружены
        for filename in transcription_results.keys():
            print(f"  - {filename}")
    else:
        print(f"Файл {results_file} не найден")
        print("  Сначала выполните Этап 1 для получения транскрипций")
        transcription_results = {}

    # Парсинг реальных голосовых сообщений
    parsed_real_results = []
    for filename, data in transcription_results.items():
        text = data['text']
        print(f"\nФайл: {filename}")
        print(f"  Текст: {text.strip()}")
        parsed = parse_voice_report(text)
        parsed['source_file'] = filename
        parsed_real_results.append(parsed)
        # Краткий вывод
        if parsed['coordinates']:
            print(f"  Координаты: {parsed['coordinates']}")
        if parsed['measurements']:
            print(f"  Показатели: {parsed['measurements']}")
        if parsed['material']:
            print(f"  Материал: {parsed['material']}")
        if parsed['damage']:
            print(f"  Повреждение: {parsed['damage']}%")
        if parsed['problems']:
            print(f"  Проблемы: {parsed['problems']}")

    # Тестирование на синтетических фразах (для демонстрации)
    test_phrases = [
        "Проба воды в точке 59.93, 30.25, аммоний 0.5, нитриты 0.08, фосфаты 0.3, pH 7.2",
        "Обнаружена асбестовая труба с повреждением 30 процентов, координаты 60.1, 29.8",
        "Разлив нефти в точке 59.8, 30.3, срочно требуется помощь, нефтепродукты 2.5",
        "Проба из реки Нева, координаты 59.95, 30.35, аммоний 0.8, фосфаты 0.6, pH 6.8",
        "ИЗВ 2.3, точка 59.88, 30.42, превышение нитритов 0.15",
        "Найдена бетонная труба, трещина, срочно!"
    ]
    synthetic_results = []
    for i, phrase in enumerate(test_phrases, 1):
        parsed = parse_voice_report(phrase)
        parsed['test_id'] = i
        parsed['source'] = 'synthetic'
        synthetic_results.append(parsed)
    print(f"Протестировано {len(test_phrases)} синтетических сообщений")

    # Сохранение результатов парсинга...
    os.makedirs('voice_results', exist_ok=True)
    # Сохраняем реальные результаты
    with open('voice_results/parsed_real_reports.json', 'w', encoding='utf-8') as f:
        json.dump(parsed_real_results, f, ensure_ascii=False, indent=2)
    print("Реальные результаты: voice_results/parsed_real_reports.json")
    # Сохраняем синтетические результаты
    with open('voice_results/parsed_synthetic_reports.json', 'w', encoding='utf-8') as f:
        json.dump(synthetic_results, f, ensure_ascii=False, indent=2)
    print("Синтетические результаты: voice_results/parsed_synthetic_reports.json")

    # Статистика парсинга
    def print_stats(results, name):
        total = len(results)
        if total == 0:
            print(f"\n{name}: нет данных")
            return
        has_coords = sum(1 for r in results if r['coordinates'])
        has_measurements = sum(1 for r in results if r['measurements'])
        has_material = sum(1 for r in results if r['material'])
        has_problems = sum(1 for r in results if r['problems'])
        print(f"\n{name}:")
        print(f"  Координаты: {has_coords}/{total} ({has_coords / total * 100:.0f}%)")
        print(f"  Измерения: {has_measurements}/{total} ({has_measurements / total * 100:.0f}%)")
        print(f"  Материал: {has_material}/{total} ({has_material / total * 100:.0f}%)")
        print(f"  Проблемы: {has_problems}/{total} ({has_problems / total * 100:.0f}%)")

    print_stats(parsed_real_results, "Реальные аудиофайлы")
    print_stats(synthetic_results, "Синтетические сообщения")