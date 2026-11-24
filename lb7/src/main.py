import os
import re
import time
from difflib import SequenceMatcher
import librosa
from voice_enhanced_pipeline import VoiceEnhancedPipeline, parse_coordinates

# Тестовые голосовые сообщения: (текст для озвучивания, имя файла).
# Общий список для задач 5.1 и 6.1, чтобы эталонные тексты совпадали с аудио
TEST_AUDIOS = [
    ("Проба воды. Координаты 45.35, 47.85. Аммоний 0.5, фосфаты 0.3, pH 7.2", "test_voice_1.mp3"),
    ("Обнаружена асбестовая труба. Координаты 45.62, 48.14. Повреждение 30 процентов", "test_voice_2.mp3"),
    ("Разлив нефти. Координаты 45.81, 48.53. Срочно", "test_voice_3.mp3")
]


def task_5_1(pipeline: VoiceEnhancedPipeline):
    # Пересоздание аудиофайлов
    os.makedirs('audio_samples', exist_ok=True)
    # Удаляем старые файлы
    for f in ['test_voice_1.mp3', 'test_voice_2.mp3', 'test_voice_3.mp3']:
        filepath = f'audio_samples/{f}'
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Удален: {f}")

    # Создаем новые файлы
    from gtts import gTTS

    test_audios = TEST_AUDIOS

    for text, filename in test_audios:
        filepath = f'audio_samples/{filename}'
        print(f"Создание: {filename}")
        tts = gTTS(text=text, lang='ru', slow=False)
        tts.save(filepath)
        time.sleep(1)
        print(f"  Создан ({os.path.getsize(filepath)} байт)")

    # Тестирование
    pipeline.init_voice_module()
    pipeline.create_sample_points(10, 'astrahan')

    print("\nДобавление проб голосом...")
    for text, filename in test_audios:
        pipeline.add_sample_by_voice(f'audio_samples/{filename}')

    stats = pipeline.get_voice_stats()
    print(f"\nСтатистика: {stats['with_coords']} из {stats['total']} проб с координатами")

    if pipeline.voice_samples:
        print("\nДобавленные пробы:")
        for vs in pipeline.voice_samples:
            coords = vs['data']['coordinates']
            if coords:
                print(f"  ID {vs['id']}: {coords['lat']}, {coords['lon']}")

    pipeline.summary()
    pipeline.visualize_samples()


import os
import numpy as np
from datetime import datetime
import json

def evaluate_asr_quality(pipeline, test_phrases):
    """
    Оценка качества распознавания речи
    """
    if pipeline.asr_model is None:
        print("Речевой модуль не инициализирован")
        return []
    results = []
    print("\nТестирование распознавания:")
    print("-" * 50)
    for i, (expected_text, audio_path) in enumerate(test_phrases):
        if not os.path.exists(audio_path):
            print(f"Файл {audio_path} не найден, пропускаем")
            continue
        # Распознаем
        # Загружаем звук заранее (моно, 16 кГц): Whisper без ffmpeg не читает файлы сам
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        result = pipeline.asr_model.transcribe(audio, language='ru')
        recognized = result['text']
        # Оценка сходства
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, expected_text.lower(), recognized.lower()).ratio()
        # Проверка наличия координат
        coords = parse_coordinates(recognized)
        results.append({
            'id': i + 1,
            'audio_file': os.path.basename(audio_path),
            'expected': expected_text,
            'recognized': recognized,
            'similarity': similarity,
            'has_coordinates': coords is not None,
            'coordinates': coords
        })
        print(f"\n{i + 1}. Файл: {os.path.basename(audio_path)}")
        print(f"  Ожидалось: {expected_text[:80]}...")
        print(f"  Распознано: {recognized[:80]}...")
        print(f"  Сходство: {similarity:.3f}")
        print(f"  Координаты: {'найдены' if coords else 'не найдены'}")
    return results


def task_6_1(pipeline: VoiceEnhancedPipeline):
    # Модель загружается в task_5_1; если она не вызывалась — загружаем здесь
    if pipeline.asr_model is None:
        pipeline.init_voice_module()

    # Создание тестовых пар: эталонный текст — тот же, что озвучен в task_5_1
    test_pairs = [(text, f'audio_samples/{filename}') for text, filename in TEST_AUDIOS]

    # Оценка качества
    evaluation_results = evaluate_asr_quality(pipeline, test_pairs)

    # Сохранение результатов
    os.makedirs('voice_results', exist_ok=True)
    # Сохраняем результаты
    evaluation_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(evaluation_results),
        'average_similarity': np.mean([r['similarity'] for r in evaluation_results]) if evaluation_results else 0,
        'successful_coords': sum(1 for r in evaluation_results if r['has_coordinates']),
        'results': evaluation_results
    }
    with open('voice_results/asr_quality_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
    print("Результаты сохранены в voice_results/asr_quality_evaluation.json")

    # Выводы
    if evaluation_results:
        avg = evaluation_summary['average_similarity']
        coords_success = evaluation_summary['successful_coords']
        if avg > 0.9:
            quality = "Отличное"
        elif avg > 0.7:
            quality = "Хорошее"
        elif avg > 0.5:
            quality = "Удовлетворительное"
        else:
            quality = "Низкое"
        print(f"\nКачество распознавания: {quality}")
        print(f"  Среднее сходство: {avg:.3f}")
        print(f"  Успешное извлечение координат: {coords_success}/{len(evaluation_results)} ({coords_success / len(evaluation_results) * 100:.0f}%)")
        print("\nДетальные результаты:")
        print("-" * 50)
        for r in evaluation_results:
            print(f"\n{r['audio_file']}:")
            print(f"  Сходство: {r['similarity']:.3f}")
            print(f"  Координаты: {r['coordinates'] if r['coordinates'] else 'нет'}")
            print(f"  Распознано: {r['recognized'][:80]}...")


def task_1_3(pipeline: VoiceEnhancedPipeline, runs=3):
    """
    Влияние температуры на «креативность» Whisper:
    один и тот же файл распознается несколько раз при temperature = 0.0, 0.5, 1.0
    """
    import re
    from difflib import SequenceMatcher

    if pipeline.asr_model is None:
        pipeline.init_voice_module()

    # Файл с наибольшим количеством чисел и его эталонный текст
    reference, filename = TEST_AUDIOS[0]
    audio_path = f'audio_samples/{filename}'
    if not os.path.exists(audio_path):
        print(f"Файл {audio_path} не найден. Сначала выполните task_5_1")
        return
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Эталонные значения: все группы цифр и координаты
    # (Whisper пишет "45.35" как "45, 35", поэтому сравниваем группы цифр, а не числа)
    ref_digits = re.findall(r'\d+', reference)
    ref_words = set(re.findall(r'[а-яёa-z]+', reference.lower()))
    expected_coords = parse_coordinates(reference)

    print(f"\nФайл: {filename}")
    print(f"Эталон: {reference}")
    print(f"Запусков на каждую температуру: {runs}")

    results = {}
    for temp in [0.0, 0.5, 1.0]:
        print(f"\nТемпература {temp}")
        texts, runs_info = [], []
        for run in range(1, runs + 1):
            text = pipeline.asr_model.transcribe(audio, language='ru', temperature=temp)['text'].strip()
            texts.append(text)
            # Сходство с эталоном
            similarity = SequenceMatcher(None, reference.lower(), text.lower()).ratio()
            # Точность чисел: доля эталонных групп цифр, найденных в том же порядке
            digits = re.findall(r'\d+', text)
            matcher = SequenceMatcher(None, ref_digits, digits)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            digits_acc = matched / len(ref_digits)
            # «Галлюцинации»: слова, которых нет в эталоне
            extra_words = [w for w in re.findall(r'[а-яёa-z]+', text.lower()) if w not in ref_words]
            # Правильно ли извлечены координаты
            coords = parse_coordinates(text)
            coords_ok = coords == expected_coords
            runs_info.append({
                'run': run,
                'text': text,
                'similarity': similarity,
                'digits_accuracy': digits_acc,
                'extra_words': extra_words,
                'coordinates': coords,
                'coordinates_correct': coords_ok
            })
            print(f"\nЗапуск {run}: {text}")
            print(f"{'.' * 4}Сходство с эталоном: {similarity:.3f}")
            print(f"{'.' * 4}Точность чисел: {matched}/{len(ref_digits)} ({digits_acc * 100:.0f}%)")
            print(f"{'.' * 4}Слова, отсутствующие в эталоне: {extra_words if extra_words else 'нет'}")
            print(f"{'.' * 4}Координаты: {coords} ({'верно' if coords_ok else 'неверно'})")
        unique = len(set(texts))
        results[str(temp)] = {
            'unique_results': unique,
            'avg_similarity': float(np.mean([r['similarity'] for r in runs_info])),
            'avg_digits_accuracy': float(np.mean([r['digits_accuracy'] for r in runs_info])),
            'avg_extra_words': float(np.mean([len(r['extra_words']) for r in runs_info])),
            'coordinates_correct': sum(r['coordinates_correct'] for r in runs_info),
            'runs': runs_info
        }

    print(f"{'Температура':<12}{'Разных':>8}{'Сходство':>10}{'Числа':>8}{'Лишних слов':>13}{'Координаты':>12}")
    for temp, r in results.items():
        print(f"{temp:<12}{r['unique_results']:>8}/{runs}{r['avg_similarity']:>10.3f}"
              f"{r['avg_digits_accuracy'] * 100:>8.0f}%{r['avg_extra_words']:>13.1f}"
              f"{r['coordinates_correct']:>12}/{runs}")


# Фразы с профессиональной терминологией и ключевые термины в каждой из них.
# Термины записаны в той же форме, в какой они стоят во фразе
ECO_TERM_PHRASES = [
    ("Индекс загрязнения воды ИЗВ составил 2.3, вода умеренно загрязненная.",
     ["индекс загрязнения воды", "ИЗВ"]),
    ("В заливе наблюдается эвтрофикация и цветение воды.",
     ["эвтрофикация", "цветение воды"]),
    ("Сброс сточных вод в водоохранной зоне запрещен.",
     ["сточных вод", "водоохранной зоне"]),
    ("Выявлено превышение ПДК по нефтепродуктам в три раза.",
     ["ПДК", "нефтепродуктам"]),
    ("БПК5 составляет 4 миллиграмма кислорода на литр.",
     ["БПК5"]),
    ("Обнаружены цианобактерии и сине-зеленые водоросли.",
     ["цианобактерии", "сине-зеленые водоросли"]),
    ("Отмечена гипоксия придонного слоя и повышенная мутность.",
     ["гипоксия", "придонного слоя", "мутность"]),
    ("Проведен мониторинг макрозообентоса и фитопланктона.",
     ["макрозообентоса", "фитопланктона"]),
    ("Зафиксирован дефицит растворенного кислорода и рост аммонийного азота.",
     ["растворенного кислорода", "аммонийного азота"]),
]


def normalize_text(text):
    """
    Нормализация для сравнения: нижний регистр, ё -> е, без знаков препинания
    """
    text = text.lower().replace('ё', 'е').replace('-', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def term_found(term, text):
    """
    Есть ли термин в тексте. Пробелы не учитываются, чтобы "БПК 5" совпадало с "БПК5"
    """
    return normalize_text(term).replace(' ', '') in normalize_text(text).replace(' ', '')


def fix_terms(text, glossary, threshold=0.8):
    """
    Пост-обработка: слова текста, похожие на термин из словаря, заменяются на этот термин.
    Похожесть считается по SequenceMatcher для окон из 1..(n+1) слов, где n - число слов в термине
    """
    words = text.split()
    # Сначала длинные термины, чтобы "индекс загрязнения воды" не разбивался на части
    for term in sorted(glossary, key=len, reverse=True):
        if term_found(term, ' '.join(words)):
            continue
        # Короткие термины (аббревиатуры ИЗВ, ПДК) пропускаем: нечеткое сравнение
        # коротких строк ненадежно, например "и в" оказывается похоже на "ИЗВ"
        if len(term) < 5:
            continue
        term_norm = normalize_text(term).replace(' ', '')
        n = len(term.split())
        best = None  # (похожесть, начало окна, длина окна)
        for size in range(1, n + 2):
            for start in range(len(words) - size + 1):
                window = normalize_text(' '.join(words[start:start + size])).replace(' ', '')
                if not window:
                    continue
                score = SequenceMatcher(None, window, term_norm).ratio()
                if score >= threshold and (best is None or score > best[0]):
                    best = (score, start, size)
        if best:
            _, start, size = best
            words[start:start + size] = [term]
    return ' '.join(words)


def task_1_4(pipeline: VoiceEnhancedPipeline):
    """
    Тестирование Whisper на «экологическом сленге»
    """
    from gtts import gTTS

    if pipeline.asr_model is None:
        pipeline.init_voice_module()

    # Словарь терминов предметной области
    glossary = sorted({term for _, terms in ECO_TERM_PHRASES for term in terms})
    # Подсказка для Whisper: модель учитывает её при распознавании
    prompt = "Экологический мониторинг водных объектов. Термины: " + ", ".join(glossary) + "."

    os.makedirs('audio_samples', exist_ok=True)
    variants = ['Без обработки', 'Словарь замен', 'Подсказка Whisper']
    errors = {v: 0 for v in variants}
    total_terms = 0
    results = []

    for i, (phrase, terms) in enumerate(ECO_TERM_PHRASES, 1):
        audio_path = f'audio_samples/eco_terms_{i:02d}.mp3'
        # Озвучиваем фразу (только если файла еще нет)
        if not os.path.exists(audio_path):
            gTTS(text=phrase, lang='ru', slow=False).save(audio_path)
            time.sleep(1)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        # 1. Обычное распознавание
        base = pipeline.asr_model.transcribe(audio, language='ru', temperature=0.0)['text'].strip()
        # 2. Пост-обработка словарем замен
        fixed = fix_terms(base, glossary)
        # 3. Распознавание с подсказкой из терминов
        prompted = pipeline.asr_model.transcribe(audio, language='ru', temperature=0.0,
                                                 initial_prompt=prompt)['text'].strip()
        texts = dict(zip(variants, [base, fixed, prompted]))

        print(f"\n{i}. Эталон:      {phrase}")
        for v in variants:
            print(f"   {v + ':':<19}{texts[v]}")
        phrase_result = {'phrase': phrase, 'texts': texts, 'terms': {}}
        for term in terms:
            found = {v: term_found(term, texts[v]) for v in variants}
            for v in variants:
                errors[v] += not found[v]
            phrase_result['terms'][term] = found
        total_terms += len(terms)
        results.append(phrase_result)

    summary = {}
    print("\n")
    for v in variants:
        percent = errors[v] / total_terms * 100
        summary[v] = {'errors': errors[v], 'error_percent': percent}
        print(f"{v:<19}{errors[v]:>3} ошибок  ({percent:.1f}%)")


def evaluate_recognition(reference, text):
    """
    Сравнение распознанного текста с эталоном: сходство, точность чисел, координаты
    """
    import contextlib
    import io
    similarity = SequenceMatcher(None, reference.lower(), text.lower()).ratio()
    # Точность чисел: доля эталонных групп цифр, найденных в том же порядке
    ref_digits = re.findall(r'\d+', reference)
    matcher = SequenceMatcher(None, ref_digits, re.findall(r'\d+', text))
    matched = sum(block.size for block in matcher.get_matching_blocks())
    # parse_coordinates печатает промежуточные сообщения — скрываем их
    with contextlib.redirect_stdout(io.StringIO()):
        coords_ok = parse_coordinates(text) == parse_coordinates(reference)
    return {
        'similarity': similarity,
        'digits_matched': matched,
        'digits_total': len(ref_digits),
        'coordinates_correct': coords_ok
    }


def task_4_2(pipeline: VoiceEnhancedPipeline):
    """
    Качество распознавания «до» и «после» шумоподавления (noisereduce)
    для записей с шумом ветра, дождя и техники
    """
    import noisereduce as nr
    import soundfile as sf

    if pipeline.asr_model is None:
        pipeline.init_voice_module()

    reference = TEST_AUDIOS[0][0]
    test_files = [
        ('Без шума', 'audio_samples/test_voice_1.mp3'),
        ('Ветер', 'audio_samples/test_voice_1_wind.wav'),
        ('Дождь', 'audio_samples/test_voice_1_rain.wav'),
        ('Техника', 'audio_samples/test_voice_1_machinery.wav'),
    ]
    print(f"\nЭталон: {reference}")

    rows = []
    for noise_type, path in test_files:
        if not os.path.exists(path):
            print(f"\nФайл {path} не найден, пропускаем")
            continue
        noisy, sr = librosa.load(path, sr=16000, mono=True)
        # Лучший метод из задачи 4.1 с теми же параметрами
        denoised = nr.reduce_noise(y=noisy, sr=sr, stationary=True)
        # Сохраняем очищенную запись, чтобы ее можно было прослушать
        denoised_path = os.path.splitext(path)[0] + '_denoised.wav'
        sf.write(denoised_path, denoised, sr)

        print(f"\n{'=' * 60}\n{noise_type}: {os.path.basename(path)}\n{'=' * 60}")
        row = {'noise': noise_type}
        for stage, audio in [('До', noisy), ('После', denoised)]:
            text = pipeline.asr_model.transcribe(audio, language='ru', temperature=0.0)['text'].strip()
            metrics = evaluate_recognition(reference, text)
            row[stage] = metrics
            print(f"  {stage} шумоподавления: {text}")
            print(f"    Сходство: {metrics['similarity']:.3f}, "
                  f"числа: {metrics['digits_matched']}/{metrics['digits_total']}, "
                  f"координаты: {'верно' if metrics['coordinates_correct'] else 'неверно'}")
        rows.append(row)

    # Сводная таблица
    print(f"\n{'=' * 60}\nСводная таблица\n{'=' * 60}")
    print(f"{'Шум':<10}{'Сходство до':>13}{'после':>8}{'Числа до':>10}{'после':>8}"
          f"{'Коорд. до':>11}{'после':>8}")
    for r in rows:
        before, after = r['До'], r['После']
        print(f"{r['noise']:<10}{before['similarity']:>13.3f}{after['similarity']:>8.3f}"
              f"{before['digits_matched']:>7}/{before['digits_total']}"
              f"{after['digits_matched']:>5}/{after['digits_total']}"
              f"{'да' if before['coordinates_correct'] else 'нет':>11}"
              f"{'да' if after['coordinates_correct'] else 'нет':>8}")


def main():
    pipeline = VoiceEnhancedPipeline("Экологический мониторинг")
    # task_5_1(pipeline)
    # task_6_1(pipeline)
    # task_1_3(pipeline)
    # task_1_4(pipeline)
    task_4_2(pipeline)
    


if __name__ == '__main__':
    main()