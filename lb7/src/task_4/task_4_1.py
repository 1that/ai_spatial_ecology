import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')
try:
    import noisereduce as nr
except ImportError:
    nr = None
    print("Библиотека noisereduce не установлена, метод будет пропущен")

# Фиксируем генератор случайных чисел, чтобы шум (и результаты) повторялись
np.random.seed(42)

def add_noise_to_audio(audio_path, noise_level=0.3, output_path=None):
    """
    Добавление шума к аудиофайлу
    """
    signal, sr = librosa.load(audio_path, sr=16000)
    noise = np.random.randn(len(signal)) * noise_level
    noisy = signal + noise
    # Громкость не меняем: иначе SNR сравнивал бы сигналы разного масштаба
    if output_path:
        # Нормируем только копию для записи в WAV, чтобы не было переполнения int16
        max_val = np.max(np.abs(noisy))
        to_save = noisy / max_val if max_val > 0 else noisy
        wavfile.write(output_path, sr, (to_save * 32767).astype(np.int16))
    return noisy, sr

def align_signals(original, processed):
    """
    Выравнивание длины сигналов (обрезаем до меньшей длины)
    """
    min_len = min(len(original), len(processed))
    return original[:min_len], processed[:min_len]

def calculate_snr(clean, denoised):
    """
    Расчет SNR с выравниванием длины
    """
    clean_trim, denoised_trim = align_signals(clean, denoised)
    noise = clean_trim - denoised_trim
    signal_power = np.sum(clean_trim ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power == 0:
        return 100.0  # идеальный случай
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def spectral_subtraction(y, sr, noise_frames=10, alpha=0.5):
    """
    Шумоподавление методом спектрального вычитания
    """
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)
    noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    magnitude_denoised = np.maximum(magnitude - alpha * noise_estimate, 0)
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    # length=len(y): длина результата совпадает с исходным сигналом
    denoised = librosa.istft(D_denoised, length=len(y))
    return denoised

def wiener_filter(y, sr, noise_frames=10):
    """
    Шумоподавление фильтром Винера
    """
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)
    noise_power = np.mean(magnitude[:, :noise_frames] ** 2, axis=1, keepdims=True)
    signal_power = magnitude ** 2 - noise_power
    signal_power = np.maximum(signal_power, 0)
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
    magnitude_denoised = magnitude * np.sqrt(wiener_gain)
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    denoised = librosa.istft(D_denoised, length=len(y))
    return denoised

def nr_denoise(y, sr, stationary=True):
    """
    Шумоподавление с помощью библиотеки noisereduce
    """
    if nr is not None:
        return nr.reduce_noise(y=y, sr=sr, stationary=stationary)
    else:
        return y  # возвращаем исходный сигнал

def compare_denoising_methods(audio_path, noise_level=0.3, save_plots=True):
    """
    Сравнение различных методов шумоподавления
    """
    # Загружаем оригинал
    original, sr = librosa.load(audio_path, sr=16000)
    # Добавляем шум
    noisy, _ = add_noise_to_audio(audio_path, noise_level)
    # Применяем методы
    print("Применение методов шумоподавления...")
    # Спектральное вычитание
    ss_result = spectral_subtraction(noisy, sr)
    # Фильтр Винера
    wf_result = wiener_filter(noisy, sr)
    # noisereduce
    nr_result = nr_denoise(noisy, sr)
    # Вычисляем SNR
    original_snr = calculate_snr(original, noisy)
    ss_snr = calculate_snr(original, ss_result)
    wf_snr = calculate_snr(original, wf_result)
    nr_snr = calculate_snr(original, nr_result)
    print(f"\nОценка качества (SNR, дБ):")
    print(f"  Зашумленный: {original_snr:.2f} дБ")
    print(f"  Спектральное вычитание: {ss_snr:.2f} дБ (улучшение: {ss_snr - original_snr:.2f})")
    print(f"  Фильтр Винера: {wf_snr:.2f} дБ (улучшение: {wf_snr - original_snr:.2f})")
    print(f"  noisereduce: {nr_snr:.2f} дБ (улучшение: {nr_snr - original_snr:.2f})")
    # Визуализация
    # Если длины сигналов совпадают для отображения
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    # Обрезаем все сигналы до одинаковой длины для отображения (первые 10000 отсчетов)
    display_len = min(10000, len(original), len(noisy), len(ss_result), len(wf_result), len(nr_result))
    axes[0].plot(original[:display_len])
    axes[0].set_title('Оригинальный сигнал', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(noisy[:display_len])
    axes[1].set_title(f'Зашумленный сигнал (уровень шума = {noise_level})', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(ss_result[:display_len])
    axes[2].set_title('Спектральное вычитание', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(wf_result[:display_len])
    axes[3].set_title('Фильтр Винера', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(nr_result[:display_len])
    axes[4].set_title('noisereduce', fontsize=12)
    axes[4].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        os.makedirs('voice_results', exist_ok=True)
        plot_name = f'voice_results/denoising_{os.path.basename(audio_path)[:-4]}.png'
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {plot_name}")
    plt.show()
    return {
        'original_snr': original_snr,
        'spectral_subtraction': ss_snr,
        'wiener_filter': wf_snr,
        'noisereduce': nr_snr
    }

# Тестирование
# Список файлов для тестирования
test_files = [
    'audio_samples/speaker_female.wav',
    'audio_samples/speaker_male.wav',
    'audio_samples/test_short.ogg'
]
# Фильтруем существующие файлы
existing_files = [f for f in test_files if os.path.exists(f)]
if not existing_files:
    print("Нет доступных аудиофайлов. Сначала выполните этап 1 и 2.")
else:
    all_results = {}
    for filepath in existing_files:
        print(f"\n{'=' * 60}")
        print(f"Обработка: {os.path.basename(filepath)}")
        print('=' * 60)
        results = compare_denoising_methods(filepath, noise_level=0.3)
        all_results[os.path.basename(filepath)] = results

    # Эксперимент: влияние уровня шума на качество
    if existing_files:
        test_file = existing_files[0]
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"\nТестовый файл: {os.path.basename(test_file)}")
        for noise_lvl in noise_levels:
            print(f"\nУровень шума: {noise_lvl}")
            original, sr = librosa.load(test_file, sr=16000)
            noisy, _ = add_noise_to_audio(test_file, noise_lvl)
            # Применяем лучший метод (noisereduce)
            denoised = nr_denoise(noisy, sr)
            snr_before = calculate_snr(original, noisy)
            snr_after = calculate_snr(original, denoised)
            print(f"  SNR до: {snr_before:.2f} дБ")
            print(f"  SNR после: {snr_after:.2f} дБ")
            print(f"  Улучшение: {snr_after - snr_before:.2f} дБ")

    # Сохранение результатов
    os.makedirs('voice_results', exist_ok=True)
    import json
    from datetime import datetime
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_files': existing_files,
        'methods': ['spectral_subtraction', 'wiener_filter', 'noisereduce'],
        'results': all_results
    }
    with open('voice_results/denoising_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)