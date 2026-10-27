from water_quality_pipeline_extended import WaterQualityPipelineExtended
import json
import os
import pickle
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import shape
import tensorflow as tf


class SegmentationPipeline(WaterQualityPipelineExtended):
    """
    Расширение пайплайна методами сегментации для работы с новыми снимками.
    Позволяет загрузить обученную модель, применить ее к любому спутниковому снимку
    и сохранить результат в виде GeoTIFF и GeoJSON.
    """

    def __init__(self, name="Интеграция сегментации"):
        super().__init__(name)
        self.segmentation_model = None
        self.segmentation_masks = None
        self.best_threshold = None  # добавили поле для порога

    def load_segmentation_model(self, model_path='model_train/unet_dice.keras'):
        """
        Загрузка обученной модели сегментации.
        compile=False: функции потерь нужны только для обучения,
        для предсказания модель загружается без них
        """
        self.segmentation_model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Модель загружена: {model_path}")
        return self.segmentation_model

    def load_best_model(self, info_path='model_train/best_model_info.json'):
        """
        Загружает лучшую модель и ее оптимальный порог из JSON-файла
        вида {"best_model_file": "unet_tversky.keras", "best_threshold": 0.45}
        """
        with open(info_path, 'r', encoding='utf-8') as f:
            best_info = json.load(f)

        self.load_segmentation_model(f'{self.train_dir}/{best_info["best_model_file"]}')
        self.best_threshold = best_info['best_threshold']
        print(f"Оптимальный порог для этой модели: {self.best_threshold:.2f}")
        return self.best_threshold

    def segment_image(self, image_path, output_path=None, patch_size=256, overlap=0.5, threshold=None):
        """
        Сегментация спутникового снимка скользящим окном.
        Снимок режется на патчи с перекрытием overlap (доля от размера патча),
        предсказания всех патчей считаются батчами и усредняются в местах перекрытия.
        threshold=None - используется best_threshold (если загружен) или 0.5
        """
        if self.segmentation_model is None:
            raise RuntimeError("Сначала загрузите модель: load_segmentation_model()")
        if threshold is None:
            threshold = self.best_threshold if self.best_threshold is not None else 0.5

        with rasterio.open(image_path) as src:
            # Читаем RGB каналы (B4, B3, B2) и нормализуем так же, как при создании датасета
            img = np.stack([self.norm_band(src.read(i).astype(np.float32)) for i in (1, 2, 3)],
                           axis=-1).astype(np.uint8).astype(np.float32) / 255.0
            crs, transform = src.crs, src.transform

        h, w = img.shape[:2]
        step = max(1, int(patch_size * (1 - overlap)))

        # Координаты левых верхних углов патчей, последний патч прижат к краю снимка
        y_coords = list(range(0, h - patch_size + 1, step))
        x_coords = list(range(0, w - patch_size + 1, step))
        if y_coords[-1] != h - patch_size:
            y_coords.append(h - patch_size)
        if x_coords[-1] != w - patch_size:
            x_coords.append(w - patch_size)

        patches = np.array([img[y:y + patch_size, x:x + patch_size] for y in y_coords for x in x_coords])

        # Батчевое предсказание
        print(f"Обработка {len(patches)} патчей...")
        predictions = self.segmentation_model.predict(patches, batch_size=16, verbose=1)[..., 0]

        # Аккумуляция результатов с весами
        mask_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        coords = [(y, x) for y in y_coords for x in x_coords]
        for (y, x), pred in zip(coords, predictions):
            mask_accum[y:y + patch_size, x:x + patch_size] += pred
            weight_accum[y:y + patch_size, x:x + patch_size] += 1.0

        # Усредняем перекрывающиеся области
        self.segmentation_masks = mask_accum / np.maximum(weight_accum, 1e-8)

        # Сохраняем бинарную маску с заданным порогом и той же постобработкой, что при подборе порога
        if output_path:
            binary = self.segmenter.postprocess_mask(
                (self.segmentation_masks > threshold).astype(np.float32)).astype(np.uint8)
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=h,
                width=w,
                count=1,
                dtype=np.uint8,
                crs=crs,
                transform=transform,
                tiled=True, blockxsize=256, blockysize=256,
                compress='lzw', photometric='palette'
            ) as dst:
                dst.write(binary, 1)
                dst.write_colormap(1, {0: (0, 0, 0, 255), 1: (255, 255, 255, 255)})
                dst.build_overviews([2, 4, 8], Resampling.nearest)
            print(f"Маска сохранена: {output_path} (порог {threshold:.2f})")

        return self.segmentation_masks

    def mask_to_geojson(self, mask_path, output_path='water_bodies.geojson', min_area=1000):
        """
        Конвертация бинарной маски (GeoTIFF) в векторный слой GeoJSON.
        Фильтрует полигоны по минимальной площади (в квадратных метрах).

        Параметры:
        - mask_path: путь к GeoTIFF с бинарной маской (0 или 1).
        - output_path: путь для сохранения GeoJSON.
        - min_area: минимальная площадь полигона в кв. метрах
          (площадь считается в системе координат снимка, она должна быть метровой, например UTM).

        Возвращает:
        - gdf: GeoDataFrame с полигонами водных объектов.
        """
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

        results = []
        for geom, _ in features.shapes(mask.astype(np.int16), mask=mask == 1, transform=transform):
            geom_shapely = shape(geom)
            if geom_shapely.area >= min_area:
                results.append(geom_shapely)

        gdf = gpd.GeoDataFrame(geometry=results, crs=crs)
        gdf['area_m2'] = gdf.geometry.area
        gdf = gdf.to_crs('EPSG:4326')
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"GeoJSON сохранен: {output_path} (объектов: {len(gdf)})")
        return gdf

    def segment_and_export(self, image_path, output_mask='segmentation_mask.tif',
                           output_geojson='water_bodies.geojson',
                           threshold=None, min_area=1000):
        """
        Полный цикл: сегментация снимка, экспорт в GeoTIFF и GeoJSON и расчет статистики
        водных объектов. Статистика сохраняется рядом с GeoJSON (*_stats.json)
        """
        self.segment_image(image_path, output_path=output_mask, threshold=threshold)
        gdf = self.mask_to_geojson(output_mask, output_geojson, min_area=min_area)
        print("Интеграция завершена: получены растровая и векторная маски.")

        stats = self.calculate_water_statistics(gdf)
        stats_path = os.path.splitext(output_geojson)[0] + '_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Статистика сохранена: {stats_path}")
        return stats

    def calculate_water_statistics(self, gdf):
        """
        Статистика водных объектов по GeoDataFrame из mask_to_geojson
        (площадь берется из колонки area_m2, посчитанной в метровой системе координат снимка)
        1 га = 10 000 м²
        """
        total_m2 = float(gdf['area_m2'].sum())
        n_objects = len(gdf)
        mean_m2 = total_m2 / n_objects if n_objects else 0.0

        stats = {
            'total_area_m2': round(total_m2, 1),
            'total_area_ha': round(total_m2 / 1e4, 2),
            'n_objects': n_objects,
            'mean_area_m2': round(mean_m2, 1),
            'mean_area_ha': round(mean_m2 / 1e4, 2),
        }
        print("\nСтатистика водных объектов:")
        print(f"Общая площадь: {stats['total_area_m2']:,.0f} м2 ({stats['total_area_ha']:,.2f} га)".replace(',', ' '))
        print(f"Количество водоемов: {n_objects}")
        print(f"Средний размер объекта: {stats['mean_area_m2']:,.0f} м2 ({stats['mean_area_ha']:,.2f} га)".replace(',', ' '))
        return stats

    def plot_roi(self, image_paths, roi, dates=None, filename='roi.png'):
        """
        Область интереса (красный контур) поверх RGB-снимков серии.
        roi - shapely-геометрия в EPSG:4326 или GeoDataFrame/GeoSeries
        """
        import re
        import matplotlib.pyplot as plt

        if dates is None:
            dates = [re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(p)).group() for p in image_paths]
        roi = gpd.GeoSeries([roi], crs='EPSG:4326') if not isinstance(roi, (gpd.GeoSeries, gpd.GeoDataFrame)) else roi

        fig, axes = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5.5), squeeze=False)
        for ax, path, date in zip(axes[0], image_paths, dates):
            with rasterio.open(path) as src:
                rgb = np.stack([self.norm_band(src.read(i).astype(np.float32)) for i in (1, 2, 3)],
                               axis=-1).astype(np.uint8)
                b, crs = src.bounds, src.crs
            ax.imshow(rgb, extent=(b.left, b.right, b.bottom, b.top))
            roi.to_crs(crs).boundary.plot(ax=ax, color='red', linewidth=2)
            ax.set_xlim(b.left, b.right)
            ax.set_ylim(b.bottom, b.top)
            ax.set_title(date)
            ax.axis('off')

        roi_area = roi.to_crs(crs).area.sum() / 1e6
        fig.suptitle(f'Область интереса, площадь {roi_area:.1f} км^2', fontsize=14, y=1.03)
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150, bbox_inches='tight')
        plt.show()

    def segment_time_series(self, image_paths, roi, dates=None, reference_paths=None,
                            threshold=None, filename='water_dynamics.png'):
        """
        Сегментация серии снимков одной территории за разные даты.

        Параметры:
        - image_paths: список путей к снимкам (GeoTIFF, каналы B4, B3, B2, ...)
        - roi: общая область интереса (shapely-геометрия в EPSG:4326 или GeoDataFrame/GeoSeries)
        - dates: список дат; если None - дата берется из имени файла (первая дата вида YYYY-MM-DD)
        - reference_paths: необязательные маски Dynamic World для сравнения (класс 0 - вода)
        - threshold: порог бинаризации (None - best_threshold модели)

        Возвращает:
        - gdf: GeoDataFrame с колонками date, water_area (км²), n_objects
          и геометрией водных объектов внутри ROI на каждую дату
        """
        import re
        import matplotlib.pyplot as plt
        from rasterio.features import geometry_mask
        from shapely.geometry import MultiPolygon

        if threshold is None:
            threshold = self.best_threshold if self.best_threshold is not None else 0.5
        if dates is None:
            dates = [re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(p)).group() for p in image_paths]
        roi = gpd.GeoSeries([roi], crs='EPSG:4326') if not isinstance(roi, (gpd.GeoSeries, gpd.GeoDataFrame)) else roi

        rows = []
        for i, (path, date) in enumerate(zip(image_paths, dates)):
            print(f"\n[{i + 1}/{len(image_paths)}] {date}: {os.path.basename(path)}")
            with rasterio.open(path) as src:
                crs, transform, shape_hw = src.crs, src.transform, (src.height, src.width)
            pixel_area = abs(transform.a * transform.e)

            # Маска ROI в системе координат снимка (True - внутри области)
            roi_mask = ~geometry_mask(roi.to_crs(crs).geometry, out_shape=shape_hw, transform=transform)

            probs = self.segment_image(path, threshold=threshold)
            water = self.segmenter.postprocess_mask((probs > threshold).astype(np.float32)).astype(np.uint8)
            water[~roi_mask] = 0

            # Водные объекты - связные полигоны воды (мелкий шум уже удален postprocess_mask)
            polygons = [shape(g) for g, _ in features.shapes(water.astype(np.int16), mask=water == 1, transform=transform)]

            row = {'date': pd.to_datetime(date),
                   'water_area': sum(p.area for p in polygons) / 1e6,
                   'n_objects': len(polygons),
                   'geometry': MultiPolygon(polygons) if polygons else None}

            if reference_paths is not None:
                with rasterio.open(reference_paths[i]) as ref:
                    ref_water = (ref.read(1) == 0) & roi_mask
                row['water_area_dw'] = ref_water.sum() * pixel_area / 1e6

            rows.append(row)
            print(f"Площадь воды: {row['water_area']:.2f} км2, объектов: {row['n_objects']}")

        gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs).sort_values('date').reset_index(drop=True)

        # График динамики площади воды
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(gdf['date'], gdf['water_area'], 'o-', color='tab:blue', label='U-Net')
        if 'water_area_dw' in gdf:
            ax1.plot(gdf['date'], gdf['water_area_dw'], 's--', color='tab:gray', label='Dynamic World')
        ax1.set_ylabel('Площадь воды, км^2')
        ax1.set_xlabel('Дата')
        ax1.set_xticks(gdf['date'])
        ax1.set_xticklabels(gdf['date'].dt.strftime('%Y-%m-%d'))
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_title('Динамика площади водной поверхности')
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150, bbox_inches='tight')
        plt.show()

        print(gdf.drop(columns='geometry').to_string(index=False))
        return gdf

    def save_pract34_results(self, models, results, X_train, X_val, X_test):
        """
        Сохранение результатов практической работы №3-4.

        Параметры:
        - models: словарь обученных моделей {имя: модель}
        - results: словарь с метриками моделей {имя: {'IoU': ..., 'threshold': ...}}
        - X_train, X_val, X_test: массивы данных (для статистики)
        """
        os.makedirs(self.train_dir, exist_ok=True)

        # 1. Сохраняем все модели
        for name, model in models.items():
            # Приводим имя к безопасному виду для имени файла
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            model_path = os.path.join(self.train_dir, f'{safe_name}.keras')
            model.save(model_path)
            print(f"Модель сохранена: {model_path}")

        # 2. Сохраняем результаты сравнения (метрики)
        results_path = os.path.join(self.train_dir, 'segmentation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Результаты сравнения сохранены: {results_path}")

        # 3. Находим лучшую модель (по максимальному IoU)
        best_model_name = max(results, key=lambda n: results[n]['IoU'])
        best_iou = results[best_model_name]['IoU']

        # 4. Метаданные эксперимента
        metadata = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'best_iou': float(best_iou),  # преобразуем numpy float в обычный float
            'best_model': best_model_name,
            'best_threshold': float(results[best_model_name].get('threshold', 0.5)),
            'tversky_params': {'alpha': 0.7, 'beta': 0.3},
            'date': pd.Timestamp.now().isoformat()
        }

        metadata_path = os.path.join(self.train_dir, 'pract34_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Метаданные сохранены: {metadata_path}")

        print("\nВсе файлы сохранены в папке:", self.train_dir)
        print("  - все обученные модели (*.keras)")
        print("  - segmentation_results.pkl")
        print("  - pract34_metadata.json")
        print("  - графики (threshold_analysis_*.png, test_predictions*.png)")
