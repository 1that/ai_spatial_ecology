from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from water_quality_pipeline_extended import WaterQualityPipelineExtended
import geopandas as gpd
import os


def task_1_2(pipeline: WaterQualityPipelineExtended):
    path_to_file = 'data/delta_volga.tif'
    pipeline.load_raster(path_to_file)


def task_2_1(pipeline: WaterQualityPipelineExtended):
    indices = pipeline.calculate_all_indices()
    for name, values in indices.items():
        print(f'{name}: min = {values.min():.3f}, max = {values.max():.3f}')


def task_3_1(pipeline: WaterQualityPipelineExtended):
    pipeline.create_water_sample_points(n_points=50)
    pipeline.points_gdf = gpd.read_file('data/water_points.geojson')


def task_3_2(pipeline: WaterQualityPipelineExtended):
    pipeline.plot_with_points()


def task_4_1(pipeline: WaterQualityPipelineExtended):
    if (pipeline.points_gdf is None) or len(pipeline.points_gdf) == 0:
        raise ValueError("points_gdf должен быть заполнен данными")
      
    features = ['NH4_mgL', 'PO4_mgL', 'pH']
    X = pipeline.points_gdf[features].values
    X_scaled = StandardScaler().fit_transform(X)
    optimal_eps = pipeline.find_optimal_eps_auto(X_scaled, min_samples=4)
    pipeline.clusters = DBSCAN(eps=optimal_eps, min_samples=4).fit_predict(X_scaled)
    pipeline.points_gdf['cluster'] = pipeline.clusters
    n_clusters = len(set(pipeline.clusters)) - (1 if -1 in pipeline.clusters else 0)
    print(f"Найдено {n_clusters} кластеров (eps={optimal_eps:.3f})")


def task_5(pipeline: WaterQualityPipelineExtended):
    pipeline.visualize_clusters_map(filename='class_data/clusters_map.html')
    os.makedirs('class_data', exist_ok=True)
    pipeline.points_gdf.to_file('class_data/points_with_clusters.geojson', driver='GeoJSON')
    print("\nТочки с кластерами сохранены: class_data/points_with_clusters.geojson")


def task_6_1(pipeline: WaterQualityPipelineExtended):
    pipeline.analyze_ndvi_anomalies(n_years=10, seed=42)
    print("\nИТОГОВАЯ СТАТИСТИКА")
    print("="*50)
    print(f"Всего точек проб: {len(pipeline.points_gdf)}")
    print(f"Средний NDVI: {pipeline.indices['NDVI'].mean():.3f}")
    print(f"Средний NDWI: {pipeline.indices['NDWI'].mean():.3f}")
    print(f"Средний NDBI: {pipeline.indices['NDBI'].mean():.3f}")
    print(f"Диапазон NH4: {pipeline.points_gdf['NH4_mgL'].min():.2f} - {pipeline.points_gdf['NH4_mgL'].max():.2f} мг/л")


def main():
    pipeline = WaterQualityPipelineExtended()
    # task_1_2(pipeline)
    # task_2_1(pipeline)
    # task_3_1(pipeline)
    # task_3_2(pipeline)
    # task_4_1(pipeline)
    # task_5(pipeline)
    # task_6_1(pipeline)


if __name__ == '__main__':
    main()
