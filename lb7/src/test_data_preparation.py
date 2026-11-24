import os


def dual_male_female(i_f: str):
    import soundfile as sf
    if os.path.exists(i_f):
        data, sr = sf.read(i_f)
        split = (1 * 60 + 54) * sr
        male_version = data[:split]
        female_version = data[split + sr:]
        sf.write(os.path.join('audio_samples', 'speaker_male.wav'), male_version, sr)
        sf.write(os.path.join('audio_samples', 'speaker_female.wav'), female_version, sr)
    else:
        print(f"Файл {i_f} не найден. Проверь название!")


dual_male_female(os.path.join('audio_samples', 'test_dual.ogg'))
