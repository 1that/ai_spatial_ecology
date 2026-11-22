import glob
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


MODELS = {
    'FP32': 'tflite_models/model_fp32.tflite',
    'FP16': 'tflite_models/model_fp16.tflite',
    'Dynamic': 'tflite_models/model_dynamic.tflite',
    'Int8': 'tflite_models/model_int8.tflite'
}
TEST_DIR = 'pipe_dataset/test'
OUTPUT_DIR = 'class_data'
IMG_SIZE = 224
N_IMAGES = 10
N_PASSES = 5
N_THREADS = 4
SEED = 42


def load_image(path):
    image = img_to_array(load_img(path, target_size=(IMG_SIZE, IMG_SIZE))) / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)


def prepare_input(image, input_details):
    if input_details['dtype'] == np.uint8:
        scale, zero_point = input_details['quantization']
        return np.clip(np.round(image / scale + zero_point), 0, 255).astype(np.uint8)
    return image


def predict(interpreter, input_data):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]


def measure_times(interpreters, inputs):
    for name, interpreter in interpreters.items():
        predict(interpreter, inputs[name][0])

    times = {name: [] for name in interpreters}
    for _ in range(N_PASSES):
        for name, interpreter in interpreters.items():
            for input_data in inputs[name]:
                start = time.perf_counter()
                predict(interpreter, input_data)
                times[name].append((time.perf_counter() - start) * 1000)
    return times


def plot_inference_times(results, filename='inference_time.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results['format'], results['mean_ms'], yerr=results['std_ms'], width=0.55,
                  color='#2a78d6', error_kw={'ecolor': '#555555', 'capsize': 4, 'linewidth': 1})
    labels = [f'{mean:.1f} мс' for mean in results['mean_ms']]
    ax.bar_label(bars, labels=labels, padding=6, color='#333333')
    ax.set_title(f'Среднее время предсказания одного изображения (CPU, потоков: {N_THREADS})')
    ax.set_xlabel('Тип квантизации')
    ax.set_ylabel('Среднее время, мс')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"График сохранен: {path}")


def run_inference_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*', '*.jpg')))
    sample_paths = random.Random(SEED).sample(test_paths, N_IMAGES)

    interpreters = {}
    inputs = {}
    images = [load_image(path) for path in sample_paths]
    for name, path in MODELS.items():
        interpreter = tf.lite.Interpreter(model_path=path, num_threads=N_THREADS)
        interpreter.allocate_tensors()
        interpreters[name] = interpreter
        input_details = interpreter.get_input_details()[0]
        inputs[name] = [prepare_input(image, input_details) for image in images]

    times = measure_times(interpreters, inputs)

    rows = []
    for name in MODELS:
        rows.append({
            'format': name,
            'mean_ms': np.mean(times[name]),
            'std_ms': np.std(times[name])
        })
    results = pd.DataFrame(rows)

    print(results.round(4).to_string(index=False))

    results.to_csv(os.path.join(OUTPUT_DIR, 'inference_benchmark.csv'), index=False)
    plot_inference_times(results)
    return results
