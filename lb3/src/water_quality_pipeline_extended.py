from water_quality_pipeline import WaterQualityPipeline
from water_segmetation import WaterSegmentation
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import folium
import webbrowser
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import pandas as pd
from statsmodels.tsa.seasonal import STL
import random
import itertools
from shapely.geometry import Point, box
from rasterio.mask import mask
from rasterio.plot import show
import warnings
from PIL import Image
from matplotlib.patches import Patch, Rectangle
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import warnings
import cv2
from skimage.morphology import remove_small_objects, closing, disk
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import albumentations as A



class WaterQualityPipelineExtended(WaterQualityPipeline):
    def __init__(self, name="Практическая работа №3"):
        super().__init__(name)
        self.raster: DatasetReader | None = None
        self.bands: np.ndarray | None = None
        self.indices: dict[str, np.ndarray] = {}
        self.clusters: np.ndarray | None = None
        self.points_gdf: gpd.GeoDataFrame | None = None
        self.segmenter = WaterSegmentation()


    def load_raster(self, raster_path: str):
        """Загрузка спутникового снимка"""

        self.raster = rasterio.open(raster_path)

        if self.raster is None:
            raise FileExistsError("Неудалось открыть")

        self.bands = self.raster.read()
        
        print(f"Снимок загружен: {raster_path}")
        print(f"Размер: {self.raster.width} x {self.raster.height}")
        print(f"Каналов: {self.raster.count}")
        print(f"CRS: {self.raster.crs}")

        return self.raster


    def calculate_ndvi(self, red_idx=0, nir_idx=3):
        """Вычисление NDVI"""

        if self.bands is None:
            raise ValueError("Сначала загрузите снимок методомload_raster()")
        
        red = self.bands[red_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndvi = (nir - red) / (nir + red + 1e-10)
        self.indices['NDVI'] = ndvi
        return ndvi

    
    def calculate_ndwi(self, green_idx=1, nir_idx=3):
        """Вычисление NDWI"""
        
        if self.bands is None:
            raise ValueError("Сначала загрузите снимок методом load_raster()")
        
        green = self.bands[green_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndwi = (green - nir) / (green + nir + 1e-10)
        self.indices['NDWI'] = ndwi
        return ndwi

    
    def calculate_ndbi(self, swir_idx=4, nir_idx=3):
        """Вычисление NDBI"""

        if self.bands is None:
         raise ValueError("Сначала загрузите снимок методом load_raster()")
        
        swir = self.bands[swir_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndbi = (swir - nir) / (swir + nir + 1e-10)
        self.indices['NDBI'] = ndbi
        return ndbi

    
    def calculate_all_indices(self):
        """Вычисление всех значений индексов"""

        self.calculate_ndvi()
        self.calculate_ndwi()
        self.calculate_ndbi()
        return self.indices


    def create_water_sample_points(self, n_points=50, output_file='data/water_points.geojson'):
        """
        Создание точек проб с 4 заданными кластерами
        Алгоритм:
        1. Делит область снимка на 4 части по координатам
        2. В каждой части создает точки с разными диапазонами NH4,
        PO4, pH
        3. Формирует 4 кластера для демонстрации DBSCAN
        Параметры кластеров (гауссовы распределения):
        - Кластер 0 (чистая вода): NH4=0.05-0.25, PO4=0.02-0.10, pH=7.8-8.3
        - Кластер 1 (органика): NH4=1.2-2.2, PO4=0.25-0.55, pH=6.8-7.4
        - Кластер 2 (аммоний): NH4=3.5-5.8, PO4=0.60-1.10, pH=6.3-6.9
        - Кластер 3 (промышленное): NH4=7.0-13.0, PO4=1.50-3.20, pH=5.4-6.3
        """

        if self.raster is None:
            raise ValueError("Сначала загрузите снимок методом load_raster()")
        left, bottom, right, top = self.raster.bounds
        print(f"Границы снимка: X: {left:.1f}-{right:.1f}, Y: {bottom:.1f}-{top:.1f}")
        
        np.random.seed(42)
        group_size = n_points // 4
        locations = locations = [
            'Участок 1',
            'Участок 2',
            'Участок 3',
            'Участок 4',
            'Участок 5'
        ]
        
        """
        Задаем 4 четких кластера - для учебных целей (гауссовы распределения - сфера):
        увидеть работу алгоритма, поэкспериментировать с параметрами,
        получить красивую карту с цветными точками
        """
        cluster_params = [
            {'NH4': (0.05, 0.25), 'PO4': (0.02, 0.10), 'pH': (7.8, 8.3)},
            {'NH4': (1.2, 2.2), 'PO4': (0.25, 0.55), 'pH': (6.8, 7.4)},
            {'NH4': (3.5, 5.8), 'PO4': (0.60, 1.10), 'pH': (6.3, 6.9)},
            {'NH4': (7.0, 13.0), 'PO4': (1.50, 3.20), 'pH': (5.4, 6.3)}
        ]

        all_x, all_y, all_nh4, all_po4, all_ph, all_loc = [], [], [], [], [], []
        for i, params in enumerate(cluster_params):
            n_in_group = group_size if i < 3 else n_points - 3 * group_size

            x_start = left + i * (right - left) / 4.5
            x_end = left + (i + 1) * (right - left) / 4.5
            y_start = bottom + i * (top - bottom) / 4.5
            y_end = bottom + (i + 1) * (top - bottom) / 4.5
            x_group = np.random.uniform(x_start, x_end, n_in_group)
            y_group = np.random.uniform(y_start, y_end, n_in_group)
            
            nh4_group = np.random.uniform(params['NH4'][0],
            params['NH4'][1], n_in_group)
            po4_group = np.random.uniform(params['PO4'][0],
            params['PO4'][1], n_in_group)
            ph_group = np.random.uniform(params['pH'][0],
            params['pH'][1], n_in_group)
            loc_group = np.random.choice(locations, n_in_group)
            
            all_x.extend(x_group)
            all_y.extend(y_group)
            all_nh4.extend(nh4_group)
            all_po4.extend(po4_group)
            all_ph.extend(ph_group)
            all_loc.extend(loc_group)

        self.points_gdf = gpd.GeoDataFrame(
            data={
                    'id': range(1, n_points + 1),
                    'NH4_mgL': np.round(all_nh4, 3),
                    'PO4_mgL': np.round(all_po4, 3),
                    'pH': np.round(all_ph, 2),
                    'location': all_loc
                },
            geometry=gpd.points_from_xy(all_x, all_y),
            crs=self.raster.crs
        )

        print(f"Создано {len(self.points_gdf)} точек с 4 кластерами:")
        print(f"NH4: {self.points_gdf['NH4_mgL'].min():.2f} - {self.points_gdf['NH4_mgL'].max():.2f}")
        print(f"Группы по NH4: {self.points_gdf['NH4_mgL'].round(1).value_counts().sort_index().values}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.points_gdf.to_file(output_file, driver='GeoJSON')
        print(f"Сохранено в файле: {output_file}")

        return self.points_gdf


    def normalize(self, band):
        low, high = np.percentile(band, (2, 98))
        if high <= low:
            return np.zeros_like(band)
        return np.clip((band - low) / (high - low), 0, 1)


    def plot_rgb_with_points(self, filename='rgb_points.png'):
        """RGB-снимок с точками проб, цвет точки зависит от NH4"""

        if self.raster is None or self.bands is None or self.points_gdf is None:
            raise ValueError("Сначала загрузите снимок и создайте точки проб")

        r, g, b = self.bands[0], self.bands[1], self.bands[2]
        rgb = np.dstack((self.normalize(r), self.normalize(g), self.normalize(b)))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)

        points_in_image = 0
        for idx, row in self.points_gdf.iterrows():
            row_idx, col = self.raster.index(row.geometry.x, row.geometry.y)
            if 0 <= col < rgb.shape[1] and 0 <= row_idx < rgb.shape[0]:
                points_in_image += 1
                color = 'red' if row['NH4_mgL'] > 1.0 else 'orange' if row['NH4_mgL'] > 0.5 else 'green'
                ax.scatter(col, row_idx, c=color, s=50, alpha=0.6)

        ax.axis('off')
        ax.set_title(f'Точки проб ({points_in_image})')

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_index(self, name, cmap, vmin, vmax, title, filename=None):
        """Карта одного спектрального индекса"""

        if name not in self.indices:
            raise ValueError(f"Индекс {name} не рассчитан")

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.indices[name], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.axis('off')

        filename = filename or f'{name.lower()}.png'
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_nh4_histogram(self, filename='nh4_hist.png'):
        """Гистограмма концентрации NH4"""

        if self.points_gdf is None:
            raise ValueError("Сначала создайте точки проб")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.points_gdf['NH4_mgL'], bins=8, alpha=0.7, color='steelblue')
        ax.axvline(x=self.points_gdf['NH4_mgL'].mean(), color='red', linestyle='--', label='Среднее')
        ax.set_title('Распределение NH4')
        ax.set_xlabel('NH4, мг/л')
        ax.set_ylabel('Количество проб')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_locations(self, filename='locations.png'):
        """Распределение проб по локациям"""

        if self.points_gdf is None:
            raise ValueError("Сначала создайте точки проб")

        loc_counts = self.points_gdf['location'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(loc_counts.index.astype(str).tolist(), loc_counts.to_numpy(), alpha=0.7, color='steelblue')
        ax.set_title('Распределение проб по локациям')
        ax.set_ylabel('Количество проб')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_with_points(self):
        """Все графики по очереди, каждый в отдельном окне"""

        if not self.indices or self.points_gdf is None:
            raise ValueError("Сначала рассчитайте индексы и создайте точки проб")

        self.plot_rgb_with_points()
        self.plot_index('NDVI', 'RdYlGn', -0.2, 0.8, 'NDVI (растительность)')
        self.plot_index('NDWI', 'Blues', -0.5, 0.5, 'NDWI (водные объекты)')
        self.plot_index('NDBI', 'Reds', -0.5, 0.5, 'NDBI (застройка)')
        self.plot_nh4_histogram()
        self.plot_locations()


    def find_optimal_eps_auto(self, X_scaled, min_samples=4):
        """Вычисление оптимального eps"""

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples - 1])
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(distances)
        plt.title(f'k-distance график (k={min_samples})')
        plt.xlabel('Точки (отсортированы)')
        plt.ylabel(f'{min_samples}-е расстояние')
        plt.grid(True, alpha=0.3)
        
        try:
            kneedle = KneeLocator(range(len(distances)), distances, curve='convex',
                                   direction='increasing', S=1.0)
            if kneedle.knee is not None and kneedle.knee < len(distances) * 0.8:
                auto_eps = distances[kneedle.knee]
            else:
                auto_eps = np.percentile(distances, 40)
        except (ImportError, Exception):
            print("KneeLocator недоступен, используем перцентиль")
            auto_eps = np.percentile(distances, 40)
        
        best_eps = auto_eps
        best_score = -1
        eps_candidates = np.linspace(auto_eps * 0.6, auto_eps * 1.8, 7)

        plt.subplot(1, 2, 2)
        silhouette_scores = []

        for eps_test in eps_candidates:
            try:
                dbscan_test = DBSCAN(eps=eps_test, min_samples=min_samples)
                labels_test = dbscan_test.fit_predict(X_scaled)
                n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
                
                if n_clusters_test >= 2:
                    score = silhouette_score(X_scaled, labels_test)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            except:
                silhouette_scores.append(-1)
 
        valid_indices = [i for i, s in enumerate(silhouette_scores) if s > -1]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: silhouette_scores[i])
            best_eps = eps_candidates[best_idx]
            best_score = silhouette_scores[best_idx]
        
        print("\nМассив значений eps:")
        print(eps_candidates)
        print("Массив silhouette scores:")
        print(np.array(silhouette_scores))
        
        for i, eps_test in enumerate(eps_candidates):
            plt.scatter(eps_test, silhouette_scores[i], 
                        s=120, c='green' if i == np.argmax([s for s in silhouette_scores if s > -1]) else 'blue', 
                        label=f'eps={eps_test:.3f}' if i < 3 else "")
        plt.axvline(best_eps, color='red', linestyle='--', linewidth=2, label=f'Лучший: {best_eps:.3f} (score={best_score:.3f})')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score')
        plt.title('Автоподбор eps')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir + 'clustering', dpi=150)
        plt.show()

        print(f"Найден eps: {best_eps:.3f}")
        print(f"Silhouette: {best_score:.3f}")
        print(f"Диапазон расстояний: {distances[5]:.3f} - {np.percentile(distances, 75):.3f}")
        
        return best_eps
        

    def most_polluted_cluster(self, points_gdf, clusters):
        """Кластер с наибольшей средней концентрацией NH4 (без шума)"""

        nh4 = pd.Series(points_gdf['NH4_mgL'].to_numpy(), index=np.asarray(clusters))
        nh4 = nh4[nh4.index != -1]
        if nh4.empty:
            return None
        
        return nh4.groupby(level=0).mean().idxmax()


    def ndvi_near_water(self, x, y, radius_m=100):
        """NDVI в точке и наличие воды (NDWI > 0) в радиусе"""

        if self.raster is None or 'NDVI' not in self.indices or 'NDWI' not in self.indices:
            return None, False

        row, col = self.raster.index(x, y)
        height, width = self.indices['NDVI'].shape
        if not (0 <= row < height and 0 <= col < width):
            return None, False

        r = max(1, int(radius_m / self.raster.res[0]))
        window = self.indices['NDWI'][max(0, row - r):row + r + 1, max(0, col - r):col + r + 1]

        return float(self.indices['NDVI'][row, col]), bool((window > 0).any())


    def get_recommendations(self, cluster, high_cluster, x, y):
        """Рекомендации для всплывающего окна точки"""

        recommendations = []
        if high_cluster is not None and cluster == high_cluster:
            recommendations.append('Рекомендуется отбор повторной пробы')
        
        if cluster == -1:
            recommendations.append('Проверить координаты: возможна ошибка GPS')
        
        ndvi, near_water = self.ndvi_near_water(x, y)
        if ndvi is not None and ndvi < 0.2 and near_water:
            recommendations.append('Возможен сброс сточных вод')

        return recommendations


    def visualize_clusters_map(self, points_gdf=None, clusters=None, filename='clusters_map.html'):
        """Визуализация карты с кластерами проб"""

        if points_gdf is None:
            points_gdf = self.points_gdf
        if clusters is None:
            clusters = self.clusters
        
        if points_gdf is None or clusters is None:
            raise ValueError("Нет данных для визуализации кластеров")
        
        os.makedirs(os.path.dirname(filename) if
        os.path.dirname(filename) else '.', exist_ok=True)
        print("\nКонвертация координат для отображения на карте...")
        print(f"Исходная CRS точек: {getattr(points_gdf, 'crs', 'Не установлена')}")

        points_wgs84 = points_gdf.copy()

        current_crs = points_gdf.crs
        if current_crs and str(current_crs) != 'EPSG:4326':
            try:
                points_wgs84 = points_gdf.to_crs('EPSG:4326')
                print(f"Конвертировано в: {points_wgs84.crs}")
            except Exception as e:
                print(f"Ошибка конвертации: {e}")
                print("Пробуем принудительно установить CRS...")
                points_wgs84 = points_gdf.copy()
                points_wgs84.crs = 'EPSG:4326'
        
        print(f"\nДиапазон координат после конвертации:")
        print(f" Долгота (X): {points_wgs84.geometry.x.min():.6f} - {points_wgs84.geometry.x.max():.6f}")
        print(f" Широта (Y): {points_wgs84.geometry.y.min():.6f} - {points_wgs84.geometry.y.max():.6f}")
        
        center_lat = points_wgs84.geometry.y.mean()
        center_lon = points_wgs84.geometry.x.mean()
        
        print(f"Центр карты: {center_lat:.6f}, {center_lon:.6f}")
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'gray', 'brown', 'lightgray']

        high_cluster = self.most_polluted_cluster(points_gdf, clusters)

        points_added = 0
        for idx, row in points_wgs84.iterrows():
            cluster = clusters[idx]
            color = 'pink' if cluster == -1 else colors[cluster % len(colors)]

            radius = 5 + row['NH4_mgL'] * 3

            geom = points_gdf.loc[idx].geometry
            recommendations = self.get_recommendations(cluster, high_cluster, geom.x, geom.y)
            recommendations_html = ''.join(f'- {r}<br>' for r in recommendations)

            popup_text = f"""
                <b>Точка {row.get('id', idx)}</b><br>
                NH₄: {row['NH4_mgL']:.2f} мг/л<br>
                PO₄: {row['PO4_mgL']:.2f} мг/л<br>
                pH: {row['pH']:.2f}<br>
                Кластер: {cluster}<br>
                Локация: {row.get('location', 'Не указана')}<br>
                {'<b>Рекомендации:</b><br>' + recommendations_html if recommendations else ''}
            """

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            points_added += 1

        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px;
            z-index: 1000; background-color: white; padding: 10px;
            border-radius: 5px; border: 2px solid grey; font-size:
            12px;">
                <b>Легенда</b><br>
                <span style="color: pink;">●</span> Шум (-1)<br>
                <span style="color: blue;">●</span> Кластер 0<br>
                <span style="color: red;">●</span> Кластер 1<br>
                <span style="color: green;">●</span> Кластер 2<br>
                <i>Размер = f(NH₄)</i>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник'
        ).add_to(m)
        folium.LayerControl().add_to(m)

        m.save(filename)
        print(f"\nКарта кластеров сохранена: {filename}")
        print(f"Добавлено точек на карту: {points_added} из {len(points_wgs84)}")
        
        webbrowser.open('file://' + os.path.realpath(filename))
        
        return m


    def analyze_ndvi_anomalies(self, n_years=10, seed=42):
        """
        Анализ NDVI: STL + поиск аномалий
        1. Создает временной ряд по паттернам MODIS леса
        2. Добавляет случайные аномалии (засухи/вырубки) - для
        иллюстрации
        3. STL-декомпозиция - остатки - выявление аномалий ±2σ
        """
        np.random.seed(seed)
        random.seed(seed)
        
        dates = pd.date_range(start='2015-01-01', periods=n_years * 12, freq='ME')
        months = dates.month
        seasonal_base = np.array([0.25, 0.28, 0.35, 0.45, 0.55, 0.70, 
                                  0.78, 0.75, 0.65, 0.50, 0.35, 0.28])
        seasonal = seasonal_base[months - 1]
        
        trend = np.linspace(0, -0.12, n_years * 12)
        
        noise = np.random.normal(0, 0.015, len(dates))

        anomalies = np.zeros(len(dates))
        anomaly_years = random.sample(range(2015, 2015 + n_years), min(2, n_years // 3))
        for year in anomaly_years:
            start_month = random.randint(4, 8)
            months_in_year = random.randint(4, 8)
            idx_start = (year - 2015) * 12 + start_month
            anomalies[idx_start:idx_start + months_in_year] -= np.random.uniform(0.15, 0.30)
        
        ndvi = np.clip(trend + seasonal + noise + anomalies, 0, 1)
        ts_df = pd.DataFrame({'date': dates, 'ndvi': ndvi}).set_index('date')
        ts_df.to_csv(os.path.join(self.output_dir, 'realistic_ndvi.csv'))
        
        print(f"Реалистичный NDVI {ts_df.index.min().year}-{ts_df.index.max().year}")
        print(f"Случайные аномалии добавлены в годы: {anomaly_years}")
        
        stl = STL(ts_df.ndvi, period=12, robust=True)
        result = stl.fit()
        
        resid_mean, resid_std = result.resid.mean(), result.resid.std()
        upper, lower = resid_mean + 2 * resid_std, resid_mean - 2 * resid_std

        anomalies_mask = (result.resid > upper) | (result.resid < lower)

        self.plot_stl_component(result.observed, 'Наблюдаемый NDVI', 'NDVI', 'ndvi_stl_observed.png')
        self.plot_stl_component(result.trend, 'Тренд NDVI', 'Тренд', 'ndvi_stl_trend.png')
        self.plot_stl_component(result.seasonal, 'Сезонность NDVI', 'Сезонность', 'ndvi_stl_seasonal.png')
        self.plot_stl_residuals(result.resid, lower, upper, anomalies_mask)
        self.plot_ndvi_with_anomalies(ts_df.ndvi, anomalies_mask)

        print("\nАномалии обнаружены автоматически")
        print("=" * 70)
        anomaly_df = ts_df[anomalies_mask].copy()
        anomaly_df['resid'] = result.resid[anomalies_mask]
        anomaly_df['zscore'] = (anomaly_df.resid - resid_mean) / resid_std
        print(anomaly_df.round(3))
        
        print("\nИнтерпретация:")
        for idx in anomaly_df.index:
            z = anomaly_df.loc[idx, 'zscore']
            year = idx.year
            if z < -2.5:
                event = "Сильная засуха/ вырубка"
            elif z < -1.8:
                event = "Умеренная засуха/болезнь"
            else:
                event = "Аномалия (анализ спутников/полей)"
            print(f" {idx.strftime('%Y-%m')}: z={z:.2f} → {event}")

        self.results['analyze_ndvi_anomalies'] = {
            'df': ts_df, 'stl': result, 'anomalies_mask':
            anomalies_mask,
            'anomaly_years_true': anomaly_years,
            'anomaly_years_detected': anomaly_df.index.year.unique()
        }
        return ts_df, anomaly_df


    def plot_stl_component(self, series, title, ylabel, filename):
        """Одна компонента STL-разложения: наблюдаемый ряд, тренд или сезонность"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(series)
        ax.set_title(title)
        ax.set_xlabel('Дата')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()


    def plot_stl_residuals(self, resid, lower, upper, anomalies_mask, filename='ndvi_stl_resid.png'):
        """Остатки STL с коридором ±2σ и найденными аномалиями"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(resid)
        ax.fill_between(resid.index, lower, upper, color='gray', alpha=0.2, label='Коридор ±2σ')
        ax.scatter(resid.index[anomalies_mask], resid[anomalies_mask],
                   color='red', s=40, label='Аномалии ±2σ', zorder=5)
        ax.set_title('Остатки STL-разложения')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Остатки')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()


    def plot_ndvi_with_anomalies(self, ndvi, anomalies_mask, filename='ndvi_anomalies.png'):
        """Исходный ряд NDVI с отмеченными аномалиями"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(ndvi, label='NDVI')
        ax.scatter(ndvi.index[anomalies_mask], ndvi[anomalies_mask],
                   color='red', s=60, label='Обнаруженные аномалии', zorder=5)
        ax.set_title('NDVI с найденными аномалиями')
        ax.set_xlabel('Дата')
        ax.set_ylabel('NDVI')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()


    def plot_ndvi_histograms(self, ndvi_by_label, filename='ndvi_hist.png'):
        """Гистограммы NDVI нескольких снимков на одном графике"""

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green', 'red']
        for (label, ndvi), color in zip(ndvi_by_label.items(), colors):
            median = float(np.median(ndvi))
            ax.hist(ndvi.ravel(), bins=100, range=(-1, 1), density=True,
                    alpha=0.5, color=color, label=label)
            ax.axvline(median, color=color, linestyle='--', label=f'Медиана: {median:.2f}')
    
        ax.set_title('Распределение NDVI')
        ax.set_xlabel('NDVI')
        ax.set_ylabel('Плотность')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def norm_band(self, band):
        """
        Функция нормализации каналов для визуализации (отсекаем 2% и 98% процентили)
        """
        p2, p98 = np.percentile(band, (2, 98))
        return np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1) * 255


    def create_dataset_from_pract2(self, raster_path='data/delta_volga.tif', 
                                output_dir='water_dataset', patch_size=256, stride=32,
                                save_4band=True,
                                dynamic_world=True, dynamic_world_path='data/dynamic_world.tif'):
        """
        Использует спутниковый снимок из Практической работы №2
        и создает набор патчей (изображений) с соответствующими
        масками воды.
        Параметры:
        - raster_path: путь к GeoTIFF снимку
        - output_dir: папка для сохранения датасета
        - patch_size: размер патча (обычно 256x256)
        - stride: шаг нарезки (меньше шаг = больше патчей)
        - save_4band: сохранять ли 4-канальные данные (RGB + NIR)
        - dynamic_world: эталон по Dynamic World того же снимка (иначе NDWI > 0.15)
        - dynamic_world_path: путь к маске Dynamic World (download_dynamic_world.py)
        """
        print("\n" + "=" * 60)
        print("Задача 3.4: Создание датасет для сегментации")
        print("=" * 60)
        self.dataset_dir = output_dir

        # Создаем необходимые папки
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        os.makedirs(f'{output_dir}/masks', exist_ok=True)
        if save_4band:
            os.makedirs(f'{output_dir}/images_4band', exist_ok=True)

        # Удаляем патчи прошлого запуска, чтобы не смешивать датасеты
        for sub in ['images', 'masks', 'images_4band']:
            if os.path.isdir(f'{output_dir}/{sub}'):
                for fname in os.listdir(f'{output_dir}/{sub}'):
                    if fname.startswith('patch_'):
                        os.remove(f'{output_dir}/{sub}/{fname}')

        try:
            with rasterio.open(raster_path) as src:
                # Читаем каналы спутникового снимка
                red = src.read(1).astype(np.float32) # B4 - красный
                green = src.read(2).astype(np.float32) # B3 - зеленый
                blue = src.read(3).astype(np.float32) # B2 - синий
                nir = src.read(4).astype(np.float32) # B8 - ближний инфракрасный

                if dynamic_world:
                    # Эталонная маска по Dynamic World (класс 0 - вода) на дату снимка
                    water_mask = self.load_reference_mask(src, dynamic_world_path)
                else:
                    # Создаем эталонную маску по NDWI (индекс воды)
                    # NDWI = (Green - NIR) / (Green + NIR) дает значения >0.15 для воды
                    ndwi = (green - nir) / (green + nir + 1e-10)
                    water_mask = (ndwi > 0.10).astype(np.uint8)

                # Создаем RGB изображение для визуализации
                rgb = np.stack([self.norm_band(red), self.norm_band(green), 
                                self.norm_band(blue)], axis=-1).astype(np.uint8)
                
                # Нарезаем снимок на патчи
                h, w = rgb.shape[:2]
                patch_idx = 0
                positions = []
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        mask_patch = water_mask[y:y + patch_size, x:x + patch_size]
                        
                        # Сохраняем только патчи, где есть вода (больше 1% пикселей)
                        if np.sum(mask_patch) > patch_size ** 2 * 0.01:
                
                            # Сохраняем RGB патч
                            Image.fromarray(rgb[y:y + patch_size, x:x + patch_size]).save(
                                f'{output_dir}/images/patch_{patch_idx:04d}.png'
                            )
                    
                            # Сохраняем 4-канальные данные для NDWI (R,G,B,NIR)
                            if save_4band:
                                bands = np.stack([
                                    red[y:y + patch_size, x:x + patch_size],
                                    green[y:y + patch_size, x:x + patch_size],
                                    blue[y:y + patch_size, x:x + patch_size],
                                    nir[y:y + patch_size, x:x + patch_size]
                                ])

                                np.save(f'{output_dir}/images_4band/patch_{patch_idx:04d}.npy', bands)

                            # Сохраняем маску воды
                            Image.fromarray((mask_patch * 255).astype(np.uint8)).save(
                                f'{output_dir}/masks/patch_{patch_idx:04d}.png'
                            )

                            # Запоминаем положение патча на снимке (для пространственного разделения)
                            positions.append({'name': f'patch_{patch_idx:04d}.png',
                                              'x': x, 'y': y, 'size': patch_size})
                            patch_idx += 1

                pd.DataFrame(positions).to_csv(f'{output_dir}/patches.csv', index=False)

                print(f"Создано {patch_idx} патчей")
                print(f" - images/: RGB изображения для визуализации")
                print(f" - masks/: бинарные маски воды (1 - вода, 0 - суша)")
                print(f" - patches.csv: положение патчей на снимке")
                
                if save_4band:
                    print(f" - images_4band/: 4-канальные данные для NDWI")
                
                return patch_idx
        except Exception as e:
            print(f"Ошибка при создании датасета: {e}")
            return 0


    def load_reference_mask(self, src, reference_path, water_classes=(0,)):
        """
        Маска воды по карте классов Dynamic World (0 - вода).
        Маска должна быть на той же сетке, что и снимок
        """
        with rasterio.open(reference_path) as ref:
            if ref.crs != src.crs or ref.transform != src.transform or ref.shape != src.shape:
                raise ValueError("Сетка маски не совпадает со снимком")
            classes = ref.read(1)
        return np.isin(classes, water_classes).astype(np.uint8)


    def visualize_dataset(self, n_samples=3, filename='dataset_samples.png'):
        """
        Визуализация случайных патчей датасета: RGB, маска воды и наложение маски на снимок
        """
        images_dir = f'{self.dataset_dir}/images'
        masks_dir = f'{self.dataset_dir}/masks'
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            print(f"Датасет не найден: {self.dataset_dir}")
            return

        files = sorted(f for f in os.listdir(images_dir) if f.endswith('.png'))
        if not files:
            print("В датасете нет патчей")
            return

        samples = random.sample(files, min(n_samples, len(files)))

        fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)), squeeze=False)
        for row, name in enumerate(samples):
            image = np.array(Image.open(f'{images_dir}/{name}'))
            mask = np.array(Image.open(f'{masks_dir}/{name}')) > 0

            overlay = image.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)

            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f'{name}: RGB')
            axes[row, 1].imshow(mask, cmap='gray')
            axes[row, 1].set_title(f'Маска воды ({mask.mean() * 100:.1f}% воды)')
            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title('Наложение маски')
            for ax in axes[row]:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/{filename}', dpi=150)
        plt.show()


    def test_methods_on_patches(self, n_samples=3):
        """
        в class WaterQualityPipelineExtended(WaterQualityPipeline):
        Тестирование методов сегментации на патчах
        Сравнивает различные пороговые методы сегментации:
        - Глобальный порог по NDWI
        - Метод Оцу
        - Адаптивный порог
        - NDWI с порогом 0.15
        """
        print("\n" + "=" * 60)
        print("Задача 1.2: Тестирование методов сегментации")
        print("=" * 60)
        image_dir = f'{self.dataset_dir}/images'
        mask_dir = f'{self.dataset_dir}/masks'
        npy_dir = f'{self.dataset_dir}/images_4band'
        if not all(os.path.exists(d) for d in [image_dir, mask_dir, npy_dir]):
            print("Сначала создайте датасет с 4-канальными данными")
            return
        files = sorted(set(os.listdir(image_dir)) & set(os.listdir(mask_dir)))
        n_samples = min(n_samples, len(files))
        selected = random.sample(files, n_samples)
        selected.sort()
        # Загружаем данные
        images_rgb, true_masks, images_4band = [], [], []
        for fname in selected:
            images_rgb.append(plt.imread(f'{image_dir}/{fname}'))
            mask = plt.imread(f'{mask_dir}/{fname}')
            true_masks.append(mask[:, :, 0] if mask.ndim == 3 else mask)
            npy_path = f'{npy_dir}/{fname[:-4]}.npy'
            if os.path.exists(npy_path):
                images_4band.append(np.load(npy_path))
            else:
                images_4band.append(np.zeros((256, 256, 4), dtype=np.float32))
        # Применяем все методы
        method_results = {
            'Глобальный (0.5)': [self.segmenter.global_threshold(img4) for img4 in images_4band],
            'Оцу': [self.segmenter.otsu_threshold(img4) for img4 in images_4band],
            'Адаптивный': [self.segmenter.adaptive_threshold(img4) for img4 in images_4band],
            'NDWI (0.15)': [self.segmenter.ndwi_threshold(img4) for img4 in images_4band]
        }
        # Добавляем постобработку
        postprocessed = {}
        for method_name, preds in method_results.items():
            post_preds = [self.segmenter.postprocess_mask(pred) for pred in preds]
            postprocessed[f'{method_name}+Пост'] = post_preds
        # Визуализируем результаты
        self._show_results(images_rgb, true_masks, method_results, 'segmentation_results.png')
        self._show_results(images_rgb, true_masks, postprocessed, 'segmentation_results_post.png')
        method_results.update(postprocessed)
        # Выводим метрики
        print("\n" + "=" * 60)
        print("Итоговые метрики (средний IoU по патчам)")
        print("=" * 60)
        for method_name in method_results.keys():
            ious = [
                self.segmenter.calculate_metrics(true_masks[i], method_results[method_name][i])['IoU']
                for i in range(len(images_rgb))
            ]
            print(f"{method_name:25}: {np.mean(ious):.3f}")


    def _show_results(self, images, true_masks, methods, filename='segmentation_results.png'):
        """
        в класс WaterQualityPipelineExtended
        Вспомогательный метод для визуализации результатов
        Создает сетку изображений: оригинал, идеальная маска и
        результаты всех методов.
        """
        n_samples = len(images)
        n_methods = len(methods)
        fig, axes = plt.subplots(n_samples, n_methods + 2, figsize=(4 * (n_methods + 2), 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        print("\nIoU по методам (для первого патча):")
        print("-" * 50)
        for i in range(n_samples):
            # Оригинал
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('RGB', fontsize=12)
            axes[i, 0].axis('off')
            # Идеальная маска (Ground Truth)
            axes[i, 1].imshow(true_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth', fontsize=12)
            axes[i, 1].axis('off')
            # Результаты всех методов
            for j, (method_name, preds) in enumerate(methods.items()):
                col = j + 2
                pred = preds[i]
                # Приводим к правильной размерности
                if len(pred.shape) == 3 and pred.shape[0] == 4:
                    pred = np.mean(pred, axis=0)
                elif len(pred.shape) == 3:
                    pred = np.mean(pred, axis=-1)
                if len(pred.shape) == 2 and pred.shape == true_masks[i].shape:
                    iou = self.segmenter.calculate_metrics(true_masks[i], pred)['IoU']
                    axes[i, col].imshow(pred, cmap='Blues', vmin=0, vmax=1)
                    axes[i, col].set_title(f'{method_name}\nIoU: {iou:.3f}', fontsize=10)
                    if i == 0:
                        print(f"{method_name:25}: {iou:.3f}")
                else:
                    axes[i, col].imshow(np.zeros_like(true_masks[i]), cmap='Blues')
                    axes[i, col].set_title(f'{method_name}\nОшибка', fontsize=10)
                axes[i, col].axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.show()


    def plot_confusion_matrix(self, patch_name=None, target_water=0.2, filename='confusion_matrix.png'):
        """
        Матрица ошибок в пиксельном пространстве для одного патча и всех пороговых методов.
        Карта ошибок: зеленый - TP (вода найдена), красный - FP (ложная вода),
        желтый - FN (пропущенная вода), серый - TN (фон верно)
        patch_name=None - берется патч с долей воды, ближайшей к target_water
        """
        print("\n" + "=" * 60)
        print("Задача 2.2: Матрица ошибок для патча")
        print("=" * 60)
        image_dir = f'{self.dataset_dir}/images'
        mask_dir = f'{self.dataset_dir}/masks'
        npy_dir = f'{self.dataset_dir}/images_4band'

        def load_mask(name):
            mask = plt.imread(f'{mask_dir}/{name}')
            return (mask[:, :, 0] if mask.ndim == 3 else mask) > 0.5

        if patch_name is None:
            files = sorted(set(os.listdir(image_dir)) & set(os.listdir(mask_dir)))
            patch_name = min(files, key=lambda f: abs(load_mask(f).mean() - target_water))

        image = plt.imread(f'{image_dir}/{patch_name}')
        true_mask = load_mask(patch_name)
        image_4band = np.load(f'{npy_dir}/{patch_name[:-4]}.npy')
        print(f"Патч {patch_name}, вода {true_mask.mean() * 100:.1f}% пикселей")

        methods = {
            'Глобальный (0.5)': self.segmenter.global_threshold(image_4band),
            'Оцу': self.segmenter.otsu_threshold(image_4band),
            'Адаптивный': self.segmenter.adaptive_threshold(image_4band),
            'NDWI (0.15)': self.segmenter.ndwi_threshold(image_4band),
        }

        # Патч и эталонная маска - отдельный рисунок
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].set_title(f'RGB: {patch_name}')
        axes[1].imshow(true_mask, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title(f'Эталон ({true_mask.mean() * 100:.1f}% воды)')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/{filename[:-4]}_patch.png', dpi=150, bbox_inches='tight')
        plt.show()

        colors = {'TP': (0, 0.8, 0), 'FP': (0.9, 0, 0), 'FN': (1, 0.85, 0), 'TN': (0.6, 0.6, 0.6)}
        fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8))
        fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()],
                   loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12, title='Карта ошибок')

        rows = []
        n = true_mask.size
        for col, (name, pred) in enumerate(methods.items()):
            pred = pred > 0.5
            tp, fp = (true_mask & pred).sum(), (~true_mask & pred).sum()
            fn, tn = (true_mask & ~pred).sum(), (~true_mask & ~pred).sum()

            # Карта ошибок
            error_map = np.zeros((*true_mask.shape, 3))
            for key, area in [('TP', true_mask & pred), ('FP', ~true_mask & pred),
                              ('FN', true_mask & ~pred), ('TN', ~true_mask & ~pred)]:
                error_map[area] = colors[key]
            iou = tp / (tp + fp + fn + 1e-10)
            axes[0, col].imshow(error_map)
            axes[0, col].set_title(f'{name}\nIoU = {iou:.3f}')
            axes[0, col].axis('off')

            # Матрица ошибок 2x2: строки - эталон, столбцы - предсказание
            matrix = np.array([[tp, fn], [fp, tn]])
            axes[1, col].imshow(matrix / n, cmap='Blues', vmin=0, vmax=1)
            for (r, c), value in np.ndenumerate(matrix):
                label = [['TP', 'FN'], ['FP', 'TN']][r][c]
                axes[1, col].text(c, r, f'{label}\n{value}\n({value / n * 100:.1f}%)', ha='center', va='center',
                                  color='white' if value / n > 0.5 else 'black', fontsize=11)
            axes[1, col].set_xticks([0, 1], ['Вода', 'Фон'])
            axes[1, col].set_yticks([0, 1], ['Вода', 'Фон'])
            axes[1, col].set_xlabel('Предсказание')
            axes[1, col].set_ylabel('Эталон')

            rows.append({'Метод': name, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'IoU': iou,
                         'Precision': tp / (tp + fp + 1e-10), 'Recall': tp / (tp + fn + 1e-10)})

        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.show()

        table = pd.DataFrame(rows).set_index('Метод')
        print(table.round(3).to_string())
        return table


    def load_dataset(self, data_dir='water_dataset', img_size=(256, 256), train_size=0.7, val_size=0.15,
                     block_size=448):
        """
        Пространственно разделяет данные на три части по клеткам снимка,
        случайно распределенным между выборками:
        - Train (обучение) - около 70%
        - Val (валидация) - около 15%
        - Test (тест) - около 15%
        Патчи, пересекающие границы клеток разных выборок, исключаются, чтобы
        перекрывающиеся патчи не попадали в разные выборки
        """
        print("\n" + "=" * 60)
        print("Загрузка и разделение датасета")
        print("=" * 60)

        image_dir = f'{data_dir}/images'
        mask_dir = f'{data_dir}/masks'
        positions_path = f'{data_dir}/patches.csv'

        if not (os.path.exists(image_dir) and os.path.exists(mask_dir)):
            print("Сначала создайте датасет")
            return None
        if not os.path.exists(positions_path):
            print("Нет файла patches.csv с положением патчей, пересоздайте датасет")
            return None

        common_files = set(os.listdir(image_dir)) & set(os.listdir(mask_dir))
        positions = pd.read_csv(positions_path)
        positions = positions[positions['name'].isin(common_files)].reset_index(drop=True)
        print(f"Найдено {len(positions)} пар изображение-маска")

        images, masks = [], []
        for img_file in positions['name']:
            mask_file = img_file

            # Загрузка и нормализация изображения
            img = plt.imread(f'{image_dir}/{img_file}')
            if img.max() > 1.0:
                if img.max() <= 255:
                    img = img.astype(np.float32) / 255.0
                else:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

            # Ресайз при необходимости
            if img.shape[:2] != img_size:
                img = resize(img, img_size, preserve_range=True, anti_aliasing=True)

            # Загрузка маски
            mask = plt.imread(f'{mask_dir}/{mask_file}')
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape[:2] != img_size:
                mask = resize(mask, img_size, preserve_range=True, anti_aliasing=False)
            mask = (mask > 0.5).astype(np.float32)

            images.append(img.astype(np.float32))
            masks.append(mask)

        X, y = np.array(images), np.array(masks)
        print(f"Размерность данных: X {X.shape}, y {y.shape}")

        # Пространственное разделение на train/val/test по клеткам снимка
        split = self.block_split(positions['x'].to_numpy(), positions['y'].to_numpy(),
                                 y.mean(axis=(1, 2)), int(positions['size'].iloc[0]),
                                 block_size, train_size, val_size)
        if split is None:
            print("Не удалось разделить датасет: слишком мало патчей")
            return None
        (train_idx, val_idx, test_idx), cell_set = split

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_used = len(train_idx) + len(val_idx) + len(test_idx)
        n_cells = [list(cell_set.values()).count(k) for k in range(3)]
        print(f"\nКлетки {block_size}x{block_size}: Train {n_cells[0]}, Val {n_cells[1]}, Test {n_cells[2]}")
        print(f"Исключено патчей на границах клеток разных выборок: {len(X) - n_used}")
        print(f"\nРазделение: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
        for name, part in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            print(f"  {name:5}: {len(part) / n_used * 100:4.1f}% патчей, вода {part.mean() * 100:.1f}% пикселей")

        self.dataset = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        # Сведения о разбиении для визуализации (plot_dataset_split)
        self.split_info = {
            'positions': positions, 'block_size': block_size, 'cell_set': cell_set,
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx
        }

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


    def plot_dataset_split(self, raster_path='data/delta_volga.tif', filename='dataset_split.png'):
        """
        Визуализация разбиения датасета на снимке: клетки
        закрашены цветом своей выборки, точки - центры патчей
        (цвет - выборка, серый крестик - патч исключен на границе клеток разных выборок)
        """
        if not hasattr(self, 'split_info'):
            print("Сначала выполните load_dataset()")
            return

        info = self.split_info
        positions, block_size, cell_set = info['positions'], info['block_size'], info['cell_set']
        patch_size = int(positions['size'].iloc[0])

        with rasterio.open(raster_path) as src:
            rgb = np.stack([self.norm_band(src.read(i).astype(np.float32)) for i in (1, 2, 3)],
                           axis=-1).astype(np.uint8)
        h, w = rgb.shape[:2]

        names = ['Train', 'Val', 'Test']
        colors = ['tab:green', 'tab:orange', 'tab:purple']

        fig, ax = plt.subplots(figsize=(10, 11))
        ax.imshow(rgb)

        # Клетки, закрашенные цветом своей выборки (клетки без патчей не закрашены)
        for (r, c), part in cell_set.items():
            ax.add_patch(Rectangle((c * block_size, r * block_size), block_size, block_size,
                                   facecolor=colors[part], edgecolor='white',
                                   alpha=0.3, linewidth=1))

        # Центры патчей
        centers_x = positions['x'].to_numpy() + patch_size / 2
        centers_y = positions['y'].to_numpy() + patch_size / 2
        used = np.zeros(len(positions), dtype=bool)
        for name, color, key in zip(names, colors, ['train_idx', 'val_idx', 'test_idx']):
            idx = info[key]
            used[idx] = True
            ax.scatter(centers_x[idx], centers_y[idx], s=25, color=color,
                       edgecolor='black', linewidth=0.5, label=f'{name}: {len(idx)} патчей')
        ax.scatter(centers_x[~used], centers_y[~used], s=20, marker='x', color='lightgray',
                   label=f'Исключены: {(~used).sum()} патчей')

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'Разбиение датасета: клетки {block_size}x{block_size} пикселей')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150, bbox_inches='tight')
        plt.show()


    def block_split(self, xs, ys, water, patch_size, block_size=448,
                    train_size=0.7, val_size=0.15, n_trials=200, min_patches=10, seed=42):
        """
        Разделение по клеткам: снимок делится на клетки block_size x block_size,
        клетки случайно распределяются между train/val/test в заданных долях.
        Из n_trials случайных вариантов выбирается тот, где доли патчей ближе
        к заданным, а доля воды в выборках - одинаковая.
        Патч попадает в выборку, только если все клетки под ним относятся к ней
        """
        rng = np.random.default_rng(seed)
        target = np.array([train_size, val_size, 1 - train_size - val_size])
        # Клетки, которые накрывает каждый патч
        cells = [{(r, c) for r in range(y // block_size, (y + patch_size - 1) // block_size + 1)
                         for c in range(x // block_size, (x + patch_size - 1) // block_size + 1)}
                 for x, y in zip(xs, ys)]
        all_cells = sorted(set().union(*cells))
        n_val = round(len(all_cells) * target[1])
        n_test = round(len(all_cells) * target[2])
        best = None
        for _ in range(n_trials):
            # Перемешиваем клетки: первые n_val - val, следующие n_test - test, остальные - train
            order = rng.permutation(len(all_cells))
            cell_set = {all_cells[j]: 1 if i < n_val else 2 if i < n_val + n_test else 0
                        for i, j in enumerate(order)}
            labels = np.full(len(xs), -1)
            for i, patch_cells in enumerate(cells):
                sets = {cell_set[c] for c in patch_cells}
                if len(sets) == 1:
                    labels[i] = sets.pop()
            sizes = np.array([(labels == k).sum() for k in range(3)])
            if (sizes < min_patches).any():
                continue
            water_share = np.array([water[labels == k].mean() for k in range(3)])
            score = (np.abs(sizes / sizes.sum() - target).sum()
                     + 0.5 * np.abs(water_share - water_share.mean()).sum() / water_share.mean()
                     + 0.5 * (labels == -1).mean())
            if best is None or score < best[0]:
                best = (score, labels, cell_set)
        if best is None:
            return None
        labels = best[1]
        return tuple(np.where(labels == k)[0] for k in range(3)), best[2]


    def get_augmentation(self, p=0.5):
        """
        Аугментация позволяет искусственно увеличить размер
        датасета за счет случайных преобразований изображений.
        """
        return A.Compose([
            # Геометрические преобразования
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=(-30, 30), p=0.3),

            # Цветовые преобразования
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),

            # Шумы и размытие
            A.GaussNoise(std_range=(0.012, 0.025), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),

            # Погодные эффекты (для спутниковых снимков)
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),
            A.RandomRain(slant_range=(-10, 10), drop_width=1, p=0.1),
            A.RandomSnow(snow_point_range=(0.1, 0.3), p=0.1),

            # Вырезание (учат модель не полагаться на отдельные пиксели)
            A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(32, 32),
                            hole_width_range=(32, 32), fill=0, p=0.2),
        ], p=p)


    def apply_augmentation(self, images, masks, augmentor=None):
        """
        Применение аугментации к батчу изображений
        Преобразует изображения и соответствующие маски, сохраняя
        их соответствие.
        """
        if augmentor is None:
            augmentor = self.get_augmentation()

        aug_images, aug_masks = [], []
        for i in range(len(images)):
            # Приводим к uint8
            img_uint8 = (images[i] * 255).astype(np.uint8) if images[i].max() <= 1.0 else images[i].astype(np.uint8)
            mask_uint8 = (masks[i] * 255).astype(np.uint8)

            augmented = augmentor(image=img_uint8, mask=mask_uint8)

            # Возвращаем в float32 [0,1] для нейросети
            aug_images.append(augmented['image'].astype(np.float32) / 255.0)
            aug_masks.append(augmented['mask'].astype(np.float32) / 255.0)

        return np.array(aug_images), np.array(aug_masks)


    def visualize_augmentation(self, n_samples=4):
        """
        Демонстрация аугментации на примерах
        Показывает оригиналы и аугментированные версии изображений
        и масок.
        """
        print("\n" + "=" * 60)
        print("Демонстрация аугментации")
        print("=" * 60)

        if not hasattr(self, 'dataset') or 'X_train' not in self.dataset:
            print("Сначала выполните load_dataset()")
            return

        sample_images = self.dataset['X_train'][:n_samples]
        sample_masks = self.dataset['y_train'][:n_samples]

        aug_images, aug_masks = self.apply_augmentation(sample_images, sample_masks)

        fig, axes = plt.subplots(2, n_samples * 2, figsize=(4 * n_samples * 2, 8))
        for i in range(n_samples):
            # Оригиналы
            axes[0, i * 2].imshow(sample_images[i])
            axes[0, i * 2].set_title(f'Оригинал {i + 1}')
            axes[0, i * 2].axis('off')

            axes[0, i * 2 + 1].imshow(sample_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[0, i * 2 + 1].set_title(f'Маска {i + 1}')
            axes[0, i * 2 + 1].axis('off')

            # Аугментированные
            axes[1, i * 2].imshow(aug_images[i])
            axes[1, i * 2].set_title(f'Аугмент. {i + 1}')
            axes[1, i * 2].axis('off')

            axes[1, i * 2 + 1].imshow(aug_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[1, i * 2 + 1].set_title(f'Маска аугм.')
            axes[1, i * 2 + 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/augmentation_demo.png', dpi=150, bbox_inches='tight')
        plt.show()


    def find_optimal_threshold(self, model, X_val, y_val, model_name="", thresholds=np.arange(0.3, 0.9, 0.05)):
        """
        Поиск оптимального порога бинаризации для модели
        Перебирает пороги от 0.3 до 0.85 и выбирает тот,
        при котором IoU на валидационной выборке максимален.
        """
        print(f"\nПоиск оптимального порога для {model_name}...")

        best_threshold, best_iou = 0.5, 0
        threshold_results = []

        n_samples = min(30, len(X_val))
        X_sample, y_sample = X_val[:n_samples], y_val[:n_samples]
        # Предсказание считаем один раз, дальше меняется только порог
        preds = model.predict(X_sample, batch_size=4, verbose=0)[..., 0]

        for threshold in thresholds:
            ious = []
            for i in range(n_samples):
                pred_bin = self.segmenter.postprocess_mask((preds[i] > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(y_sample[i], pred_bin)['IoU']
                ious.append(iou)

            mean_iou = np.mean(ious)
            threshold_results.append((threshold, mean_iou))

            if mean_iou > best_iou:
                best_iou, best_threshold = mean_iou, threshold
                print(f"Порог {threshold:.2f}: IoU = {mean_iou:.4f} (НОВЫЙ ЛУЧШИЙ)")
            else:
                print(f"Порог {threshold:.2f}: IoU = {mean_iou:.4f}")

        self.plot_threshold_analysis(threshold_results, model_name)

        return best_threshold, best_iou


    def plot_threshold_analysis(self, threshold_results, model_name):
        """
        ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТИ IoU ОТ ПОРОГА
        Показывает, как меняется качество сегментации при изменении
        порога, и отмечает оптимальное значение.
        """
        thresholds, ious = zip(*threshold_results)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, ious, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Порог бинаризации')
        plt.ylabel('IoU')
        plt.title(f'Зависимость качества от порога\n{model_name}')
        plt.grid(True, alpha=0.3)

        best_idx = np.argmax(ious)
        best_threshold, best_iou = thresholds[best_idx], ious[best_idx]

        plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7)
        plt.scatter([best_threshold], [best_iou], color='red', s=200, zorder=5)
        # Подпись смещена от точки оптимума в пунктах, а не в единицах IoU,
        # чтобы не выходить за график при малом разбросе IoU
        plt.annotate(
            f'Оптимум: {best_threshold:.2f}\nIoU={best_iou:.4f}',
            xy=(best_threshold, best_iou), xytext=(10, -35), textcoords='offset points',
            fontsize=10, color='red'
        )

        plt.tight_layout()
        plt.savefig(
            f'{self.train_dir}/threshold_analysis_{model_name.replace(" ", "_")}.png',
            dpi=150, bbox_inches='tight'
        )
        plt.show()


    def visualize_predictions(self, models_dict, X_test, y_test, n_samples=4, save=True):
        """
        Визуализация предсказаний моделей с адаптивными порогами
        Показывает для каждого тестового примера:
        - оригинал
        - идеальную маску
        - предсказания всех моделей
        - карту ошибок лучшей модели
        """
        print("\n" + "=" * 60)
        print("Визуализация предсказаний")
        print("=" * 60)

        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        n_models = len(models_dict)

        fig, axes = plt.subplots(len(indices), n_models + 3, figsize=(4 * (n_models + 3), 4 * len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)

        # Эмпирически подобранные пороги для каждой модели
        optimal_thresholds = {'Dice Loss': 0.65, 'Tversky Loss': 0.5, 'Combined Loss': 0.45}

        for row, idx in enumerate(indices):
            img, true_mask = X_test[idx], y_test[idx].squeeze()
            col = 0

            axes[row, col].imshow(img)
            axes[row, col].set_title('Снимок')
            axes[row, col].axis('off')
            col += 1

            axes[row, col].imshow(true_mask, cmap='Blues', vmin=0, vmax=1)
            axes[row, col].set_title('Идеальная маска')
            axes[row, col].axis('off')
            col += 1

            ious = []
            for name, model in models_dict.items():
                pred_raw = model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(name, 0.5)

                # Для Dice Loss используем Otsu если предсказания слишком высокие
                if name == 'Dice Loss' and pred_raw.mean() > 0.7:
                    try:
                        pred_norm = ((pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min() + 1e-8) * 255).astype(np.uint8)
                        threshold = threshold_otsu(pred_norm) / 255.0
                    except:
                        threshold = 0.7

                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(true_mask, pred_bin)['IoU']
                ious.append((name, iou))

                axes[row, col].imshow(pred_bin, cmap='Blues', vmin=0, vmax=1)
                axes[row, col].set_title(f'{name}\nпорог={threshold:.2f}\nIoU={iou:.3f}', fontsize=9)
                axes[row, col].axis('off')
                col += 1

            # Карта ошибок лучшей модели
            if ious:
                best_name, best_iou = max(ious, key=lambda x: x[1])
                best_model = models_dict[best_name]
                pred_raw = best_model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(best_name, 0.5)
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))

                error_map = np.zeros((*pred_bin.shape, 3))
                tp_mask = (true_mask > 0.5) & (pred_bin > 0.5)  # True Positive (зеленый)
                fp_mask = (true_mask <= 0.5) & (pred_bin > 0.5)  # False Positive (красный)
                fn_mask = (true_mask > 0.5) & (pred_bin <= 0.5)  # False Negative (желтый)
                error_map[tp_mask] = [0, 1, 0]
                error_map[fp_mask] = [1, 0, 0]
                error_map[fn_mask] = [1, 1, 0]

                axes[row, -1].imshow(error_map)
                axes[row, -1].set_title(f'Ошибки\n{best_name}: {best_iou:.3f}', fontsize=10)
                axes[row, -1].axis('off')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.dataset_dir}/test_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()


    def visualize_predictions_with_thresholds(self, models_dict, X_test, y_test, optimal_thresholds=None, n_samples=4, save=True):
        """
        Визуализация предсказаний с оптимальными порогами
        Использует найденные ранее оптимальные пороги для каждой модели.
        """
        print("\n" + "=" * 60)
        print("Визуализация - оптимальные пороги")
        print("=" * 60)

        if optimal_thresholds is None:
            optimal_thresholds = {name: 0.5 for name in models_dict.keys()}

        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        n_models = len(models_dict)

        fig, axes = plt.subplots(len(indices), n_models + 3, figsize=(4 * (n_models + 3), 4 * len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)

        for row, idx in enumerate(indices):
            img, true_mask = X_test[idx], y_test[idx].squeeze()
            col = 0

            axes[row, col].imshow(img)
            axes[row, col].set_title('Снимок')
            axes[row, col].axis('off')
            col += 1

            axes[row, col].imshow(true_mask, cmap='Blues', vmin=0, vmax=1)
            axes[row, col].set_title('Идеальная маска')
            axes[row, col].axis('off')
            col += 1

            ious = []
            for name, model in models_dict.items():
                pred_raw = model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(name, 0.5)
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(true_mask, pred_bin)['IoU']
                ious.append((name, iou))

                axes[row, col].imshow(pred_bin, cmap='Blues', vmin=0, vmax=1)
                axes[row, col].set_title(f'{name}\nпорог={threshold:.2f}\nIoU={iou:.3f}', fontsize=9)
                axes[row, col].axis('off')
                col += 1

            # Карта ошибок лучшей модели
            if ious:
                best_name, best_iou = max(ious, key=lambda x: x[1])
                best_model = models_dict[best_name]
                pred_raw = best_model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(best_name, 0.5)
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))

                error_map = np.zeros((*pred_bin.shape, 3))
                tp_mask = (true_mask > 0.5) & (pred_bin > 0.5)
                fp_mask = (true_mask <= 0.5) & (pred_bin > 0.5)
                fn_mask = (true_mask > 0.5) & (pred_bin <= 0.5)
                error_map[tp_mask] = [0, 1, 0]
                error_map[fp_mask] = [1, 0, 0]
                error_map[fn_mask] = [1, 1, 0]

                axes[row, -1].imshow(error_map)
                axes[row, -1].set_title(f'Ошибки\n{best_name}: {best_iou:.3f}', fontsize=10)
                axes[row, -1].axis('off')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.dataset_dir}/test_predictions_optimal.png', dpi=150, bbox_inches='tight')
        plt.show()