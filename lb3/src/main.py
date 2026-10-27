from water_quality_pipeline_extended import WaterQualityPipelineExtended
from water_segmetation import WaterSegmentation
from segmentation_pipeline import SegmentationPipeline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import json
import os
import glob
from shapely.geometry import Polygon, box

MODELS = {'Dice Loss': 'unet_dice', 'Tversky Loss': 'unet_tversky', 'Combined Loss': 'unet_combined'}

def task_2_1(pipeline: WaterQualityPipelineExtended):
    pipeline.test_methods_on_patches(n_samples=3)


def task_2_2(pipeline: WaterQualityPipelineExtended):
    pipeline.plot_confusion_matrix()


def task_3_1(pipeline: WaterQualityPipelineExtended):
    pipeline.create_dataset_from_pract2(stride=64)
    pipeline.visualize_dataset(n_samples=3)


def task_3_3(pipeline: WaterQualityPipelineExtended):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.load_dataset()
    # pipeline.plot_dataset_split()
    # pipeline.visualize_augmentation(n_samples=2)


def task_5_1(pipeline: WaterQualityPipelineExtended):
    pipeline.load_dataset()
    X_val, y_val = pipeline.dataset['X_val'], pipeline.dataset['y_val']
    models, thresholds = {}, {}
    for name, file in MODELS.items():
        models[name] = tf.keras.models.load_model(f'{pipeline.train_dir}/{file}.keras', compile=False)
        thresholds[name], _ = pipeline.find_optimal_threshold(models[name], X_val, y_val, model_name=file, thresholds=np.arange(0.3, 0.85, 0.05))
    return models, thresholds


def task_5_3(pipeline: WaterQualityPipelineExtended):
    models, thresholds = task_5_1(pipeline)
    X_test, y_test = pipeline.dataset['X_test'], pipeline.dataset['y_test']
    np.random.seed(42)
    pipeline.visualize_predictions(models, X_test, y_test, n_samples=4)
    np.random.seed(42)
    pipeline.visualize_predictions_with_thresholds(models, X_test, y_test, thresholds, n_samples=4)


def task_6_1(test_image='../../lb2/src/data/delta_volga.tif',
             output_mask='data/full_mask.tif',
             output_geojson='data/water_bodies.geojson'):
    seg_pipeline = SegmentationPipeline()
    info_path = f'{seg_pipeline.train_dir}/best_model_info.json'

    # Если лучшая модель еще не выбрана - подбираем пороги на val и берем модель с наибольшим IoU
    if not os.path.exists(info_path):
        seg_pipeline.load_dataset()
        X_val, y_val = seg_pipeline.dataset['X_val'], seg_pipeline.dataset['y_val']
        best = None
        for file in MODELS.values():
            model = tf.keras.models.load_model(f'{seg_pipeline.train_dir}/{file}.keras', compile=False)
            threshold, iou = seg_pipeline.find_optimal_threshold(model, X_val, y_val, model_name=file, thresholds=np.arange(0.3, 0.85, 0.05))
            if best is None or iou > best[2]:
                best = (file, threshold, iou)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({'best_model_file': f'{best[0]}.keras',
                       'best_threshold': round(float(best[1]), 2),
                       'val_iou': round(float(best[2]), 4)}, f, ensure_ascii=False, indent=2)

    seg_pipeline.load_best_model(info_path)

    # Применяем к тестовому снимку test_image
    seg_pipeline.segment_image(
        image_path=test_image,
        output_path=output_mask,
        threshold=seg_pipeline.best_threshold  # используем оптимальный порог
    )

    # Конвертируем маску в GeoJSON
    seg_pipeline.mask_to_geojson(
        mask_path=output_mask,
        output_path=output_geojson,
        min_area=1000
    )

    with rasterio.open(output_mask) as src:
        mask = src.read(1)

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='Blues')
    plt.title('Сегментированная маска всего снимка')
    plt.axis('off')
    plt.savefig(seg_pipeline.output_dir + 'full_mask.png', dpi=150, bbox_inches='tight')
    plt.show()


def task_6_2():
    seg_pipeline = SegmentationPipeline()
    # Пороги моделей подбираются на val (как в задаче 5.1)
    models, thresholds = task_5_1(seg_pipeline)
    X_test, y_test = seg_pipeline.dataset['X_test'], seg_pipeline.dataset['y_test']

    # Метрики каждой модели на test с ее оптимальным порогом
    results = {}
    for name, model in models.items():
        probs = model.predict(X_test, batch_size=4, verbose=0)[..., 0]
        metrics = [seg_pipeline.segmenter.calculate_metrics(
                       y_test[i], seg_pipeline.segmenter.postprocess_mask((probs[i] > thresholds[name]).astype(np.float32)))
                   for i in range(len(y_test))]
        results[name] = {'IoU': float(np.mean([m['IoU'] for m in metrics])),
                         'threshold': round(float(thresholds[name]), 2)}
        print(f"{name:15}: IoU на test = {results[name]['IoU']:.4f} (порог {results[name]['threshold']:.2f})")

    seg_pipeline.save_pract34_results(models, results,
                                      seg_pipeline.dataset['X_train'], seg_pipeline.dataset['X_val'], X_test)


def task_6_3_1(test_image='data/delta_volga.tif',
               output_mask='data/full_mask.tif',
               output_geojson='data/water_bodies.geojson'):
    seg_pipeline = SegmentationPipeline()
    seg_pipeline.load_best_model(f'{seg_pipeline.train_dir}/best_model_info.json')
    return seg_pipeline.segment_and_export(test_image, output_mask=output_mask,
                                           output_geojson=output_geojson, min_area=1000)


def task_6_3_2(series_dir='data/date_series'):
    seg_pipeline = SegmentationPipeline()
    seg_pipeline.load_best_model(f'{seg_pipeline.train_dir}/best_model_info.json')

    image_paths = sorted(glob.glob(f'{series_dir}/delta_volga(*).tif'))
    reference_paths = sorted(glob.glob(f'{series_dir}/dynamic_world_label(*).tif'))

    top_left = (45.801636, 48.603861)
    bottom_right = (45.749001, 48.694187)
    
    roi = box(top_left[1], bottom_right[0], bottom_right[1], top_left[0])
    seg_pipeline.plot_roi(image_paths, roi)

    gdf = seg_pipeline.segment_time_series(image_paths, roi, reference_paths=reference_paths)
    gdf.to_file('data/water_time_series.geojson', driver='GeoJSON')
    return gdf


def main():
    pipeline = WaterQualityPipelineExtended()
    # task_3_1(pipeline)
    # task_2_1(pipeline)
    # task_2_2(pipeline)
    # task_3_3(pipeline)
    # task_5_1(pipeline)
    # task_5_3(pipeline)
    # task_6_1()
    # task_6_2()
    task_6_3_1()
    # task_6_3_2()


if __name__ == '__main__':
    main()
