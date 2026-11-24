import os


def dual_male_female(i_f: str):
    import soundfile as sf
    # Делим на два файла
    if os.path.exists(i_f):
        # Загружаем файл: data — массив отсчётов, sr — частота дискретизации
        data, sr = sf.read(i_f)
        # Точка раздела: 1 минута 54 секунды (в отсчётах)
        split = (1 * 60 + 54) * sr
        # Режем файл
        male_version = data[:split]            # От начала до 1:54
        female_version = data[split + sr:]     # От 1:55 до конца
        # Сохраняем в формате .wav (он лучше для SpeechBrain и Whisper)
        sf.write(os.path.join('audio_samples', 'speaker_male.wav'), male_version, sr)
        sf.write(os.path.join('audio_samples', 'speaker_female.wav'), female_version, sr)
    else:
        print(f"Файл {i_f} не найден. Проверь название!")


dual_male_female(os.path.join('audio_samples', 'test_dual.ogg'))
