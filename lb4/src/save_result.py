import os
import shutil
import json
import pickle
from datetime import datetime

MODEL = 'model/'
TFLITE = 'tflite_models/'


def save_pract5_results():
    print("\n" + "=" * 60)
    print("Этап 6: сохранение результатов")
    print("=" * 60)

    os.makedirs('pract5_results', exist_ok=True)
    print(f"Создана папка: pract5_results/")

    print("\nКопирование моделей...")
    models_to_copy = [
        f'{MODEL}pipe_model_final.keras',
        f'{TFLITE}model_fp32.tflite',
        f'{TFLITE}model_fp16.tflite',
        f'{TFLITE}model_dynamic.tflite',
        f'{TFLITE}model_int8.tflite',
        f'{MODEL}labels.txt'
    ]
    for model_file in models_to_copy:
        if os.path.exists(model_file):
            shutil.copy(model_file, f'pract5_results/{os.path.basename(model_file)}')
            size = os.path.getsize(model_file) / 1024 / 1024
            print(f"  {model_file} ({size:.2f} MB)")
        else:
            print(f"  {model_file} не найден, пропускаем")

    print("\nКопирование графиков...")
    graphs_to_copy = [
        f'{MODEL}training_stage1.png',
        f'{MODEL}training_complete.png',
        'pipe_dataset_samples.png'
    ]
    for graph_file in graphs_to_copy:
        if os.path.exists(graph_file):
            shutil.copy(graph_file, f'pract5_results/{os.path.basename(graph_file)}')
            print(f"  {graph_file}")
        else:
            print(f"  {graph_file} не найден, пропускаем")

    print("\nКопирование результатов объединения с сегментацией...")
    integration_to_copy = [
        'class_data/integrated_map.html',
        'class_data/inspections_integrated.geojson',
        'class_data/risk_alerts.json'
    ]
    for integration_file in integration_to_copy:
        if os.path.exists(integration_file):
            shutil.copy(integration_file, f'pract5_results/{os.path.basename(integration_file)}')
            print(f"  {integration_file}")
        else:
            print(f"  {integration_file} не найден, пропускаем")

    print("\nСохранение метаданных...")

    try:
        with open(f'{MODEL}labels.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        n_classes = len(classes)
        print(f"Классы ({n_classes}): {classes}")
    except:
        classes = ['asbestos', 'hdp', 'ceramics', 'concrete', 'metal', 'geotextile']
        n_classes = len(classes)
        print(f"labels.txt не найден, использую стандартные классы")

    model_sizes = {}
    for model in ['model_fp32.tflite', 'model_fp16.tflite', 'model_dynamic.tflite', 'model_int8.tflite']:
        if os.path.exists(TFLITE + model):
            model_sizes[model] = os.path.getsize(TFLITE + model) / 1024 / 1024

    history_path = f'{MODEL}/training_history.pkl'
    with open(history_path, 'rb') as f:
        history_data = pickle.load(f)
    final_accuracy = history_data['final_accuracy']
    print(f"Точность модели (fp16): {final_accuracy:.4f} (из сохраненной истории)")

    metadata = {
        'lab': '5',
        'name': 'Mobile AI: классификация материалов дренажных труб',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'original_model': 'MobileNetV2',
            'fine_tuned': True,
            'quantization_types': list(model_sizes.keys())
        },
        'model_sizes_mb': model_sizes,
        'n_classes': n_classes,
        'classes': classes,
        'accuracy_fp16': final_accuracy,
        'files_included': models_to_copy + graphs_to_copy + integration_to_copy
    }

    with open('pract5_results/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Метаданные сохранены в pract5_results/metadata.json")

    print("\n" + "=" * 60)
    print("Итоговая статистика")
    print("=" * 60)
    total_size = 0
    for file in os.listdir('pract5_results'):
        file_path = os.path.join('pract5_results', file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1024 / 1024
            total_size += size
            print(f"  {file:35}: {size:.2f} MB")
