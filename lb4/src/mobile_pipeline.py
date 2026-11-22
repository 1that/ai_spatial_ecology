import pandas as pd
import geopandas as gpd
import json
import os
import folium
import rasterio
from datetime import datetime
from shapely.geometry import box
from water_quality_pipeline import WaterQualityPipeline

SAME_LOCATION_RADIUS_M = 100
NEAR_WATER_RADIUS_M = 500

PRIORITY_LEVELS = ['низкий', 'средний', 'высокий', 'критический']
PRIORITY_COLORS = ['green', 'orange', 'red', 'darkred']
WATER_STATUSES = {
    'same_location': 'водоем в той же локации',
    'near': 'водоем рядом',
    'far': 'водоема рядом нет',
    'no_coverage': 'вне снимка, нет данных'
}


class MobilePipeline(WaterQualityPipeline):

    def __init__(self):
        super().__init__()
        self.water_bodies = None
        self.water_coverage = None
        self.integrated = None
        self.risk_alerts = []

    def import_mobile_data(self, geojson_path):
        try:
            mobile_data = gpd.read_file(geojson_path)
            print(f"Загружено {len(mobile_data)} записей из {geojson_path}")

            if mobile_data.crs != 'EPSG:4326':
                mobile_data = mobile_data.to_crs('EPSG:4326')

            mobile_data['source'] = 'DrainEye'

            if 'name' not in mobile_data.columns:
                mobile_data['name'] = 'DrainEye: ' + mobile_data['material']

            if self.samples is None:
                self.samples = mobile_data
            else:
                self.samples = pd.concat([self.samples, mobile_data], ignore_index=True)

            print(f"Импортировано {len(mobile_data)} записей из DrainEye")
            return mobile_data

        except FileNotFoundError:
            print(f"Файл {geojson_path} не найден")
            return None
        except Exception as e:
            print(f"Ошибка при импорте: {e}")
            return None


    def analyze_mobile_data(self):
        if self.samples is None or len(self.samples) == 0:
            print("Нет данных для анализа")
            return

        print("\n" + "=" * 50)
        print("Анализ данных из мобильного приложения")
        print("=" * 50)

        if 'material' in self.samples.columns:
            material_stats = self.samples['material'].value_counts()
            print("\nРаспределение материалов:")
            for material, count in material_stats.items():
                percentage = count / len(self.samples) * 100
                print(f"  {material:12}: {count:3} шт. ({percentage:.1f}%)")

        if 'confidence' in self.samples.columns:
            avg_conf = self.samples['confidence'].mean()
            print(f"\nСредняя уверенность модели: {avg_conf:.2%}")

        print(f"\nВсего записей: {len(self.samples)}")

        if self.samples.geometry is not None:
            bounds = self.samples.total_bounds
            print(f"Географические границы:")
            print(f"  Долгота: {bounds[0]:.4f} - {bounds[2]:.4f}")
            print(f"  Широта: {bounds[1]:.4f} - {bounds[3]:.4f}")

        self.visualize_samples('mobile_data_map.html')
        return self.samples


    def load_water_bodies(self, water_path='data/water_bodies.geojson', mask_path='data/full_mask.tif'):
        with rasterio.open(mask_path) as src:
            water_crs = src.crs
            self.water_coverage = box(*src.bounds)

        self.water_bodies = gpd.read_file(water_path).to_crs(water_crs)
        print(f"Загружено {len(self.water_bodies)} водоемов из {water_path}")
        return self.water_bodies


    def integrate_with_segmentation(self):
        if self.samples is None or len(self.samples) == 0:
            raise ValueError("Сначала импортируйте данные: import_mobile_data()")
        if self.water_bodies is None:
            raise ValueError("Сначала загрузите водоемы: load_water_bodies()")

        inspections = self.samples.to_crs(self.water_bodies.crs)
        nearest = gpd.sjoin_nearest(inspections[['geometry']], self.water_bodies[['geometry']],
                                    distance_col='distance')
        distance = nearest[~nearest.index.duplicated()]['distance']
        covered = inspections.within(self.water_coverage)

        result = self.samples.copy()
        if 'damagePercent' in result.columns:
            result = result.rename(columns={'damagePercent': 'damage_percent'})

        result['water_distance_m'] = distance.where(covered).round(1)
        result['water_status'] = [self._water_status(d) for d in result['water_distance_m']]
        result['base_priority'] = [self._base_priority(d) for d in result['damage_percent']]
        raised = result['water_status'] == 'same_location'
        result['priority'] = (result['base_priority'] + raised.astype(int)).clip(upper=len(PRIORITY_LEVELS) - 1)
        result['priority_name'] = result['priority'].map(lambda p: PRIORITY_LEVELS[p])

        is_risk = (result['material'] == 'asbestos') & result['water_status'].isin(['same_location', 'near'])
        result['risk_alert'] = is_risk

        self.risk_alerts = []
        for _, row in result[is_risk].iterrows():
            self.risk_alerts.append({
                'id': row['id'],
                'material': row['material'],
                'confidence': row['confidence'],
                'water_distance_m': row['water_distance_m'],
                'priority': row['priority_name'],
                'message': f"Асбестоцементная труба в {row['water_distance_m']:.0f} м от водоема: "
                           f"риск загрязнения воды при разрушении трубы"
            })

        self.integrated = result
        self.results['integration'] = {
            'n_inspections': len(result),
            'n_raised': int(raised.sum()),
            'n_alerts': len(self.risk_alerts),
            'n_no_coverage': int((result['water_status'] == 'no_coverage').sum())
        }

        result.to_file(os.path.join(self.output_dir, 'inspections_integrated.geojson'), driver='GeoJSON')
        with open(os.path.join(self.output_dir, 'risk_alerts.json'), 'w', encoding='utf-8') as f:
            json.dump(self.risk_alerts, f, ensure_ascii=False, indent=2)

        self._print_integration_report()
        return result


    def _water_status(self, distance):
        if pd.isna(distance):
            return 'no_coverage'
        if distance <= SAME_LOCATION_RADIUS_M:
            return 'same_location'
        if distance <= NEAR_WATER_RADIUS_M:
            return 'near'
        return 'far'


    def _base_priority(self, damage_percent):
        if pd.isna(damage_percent):
            return 1
        if damage_percent < 30:
            return 0
        return 1 if damage_percent < 60 else 2


    def _print_integration_report(self):
        print("\n" + "=" * 50)
        print("Объединение с данными сегментации")
        print("=" * 50)

        print(f"Предупреждений о риске: {len(self.risk_alerts)}")
        for alert in self.risk_alerts:
            print(f"  {alert['id']}: {alert['message']}")

        no_data = self.integrated[(self.integrated['water_status'] == 'no_coverage')
                                  & (self.integrated['material'] == 'asbestos')]
        for inspection_id in no_data['id']:
            print(f"  {inspection_id}: асбестоцемент вне снимка, близость воды не проверена")


    def visualize_map(self, filename='mobile_map.html'):
        if self.integrated is None:
            raise ValueError("Сначала выполните integrate_with_segmentation()")

        coverage = gpd.GeoSeries([self.water_coverage], crs=self.water_bodies.crs).to_crs('EPSG:4326')
        west, south, east, north = coverage.total_bounds

        m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник'
        ).add_to(m)

        folium.GeoJson(
            coverage.to_json(),
            name='Границы снимка',
            style_function=lambda _: {'color': 'white', 'weight': 2, 'dashArray': '6', 'fillOpacity': 0}
        ).add_to(m)
        folium.GeoJson(
            self.water_bodies.to_crs('EPSG:4326').to_json(),
            name='Водоемы (сегментация)',
            style_function=lambda _: {'color': '#1565c0', 'weight': 1, 'fillColor': '#42a5f5', 'fillOpacity': 0.5}
        ).add_to(m)

        markers = folium.FeatureGroup(name='Инспекции').add_to(m)
        for _, row in self.integrated.iterrows():
            no_data = row['water_status'] == 'no_coverage'
            distance = 'нет данных' if no_data else f"{row['water_distance_m']:.0f} м"
            popup = (f"<b>{row['id']}</b><br>Материал: {row['material']}<br>"
                     f"Уверенность: {row['confidence']:.0%}<br>Повреждение: {row['damage_percent']}%<br>"
                     f"До воды: {distance}<br>{WATER_STATUSES[row['water_status']]}<br>"
                     f"Приоритет: <b>{row['priority_name']}</b>")
            if row['risk_alert']:
                popup += "<br><b style='color:#c62828'>Риск: асбестоцемент рядом с водой</b>"
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"{row['id']}: {row['priority_name']}",
                icon=folium.Icon(color='gray' if no_data else PRIORITY_COLORS[row['priority']],
                                 icon='exclamation-sign' if row['risk_alert'] else 'tint')
            ).add_to(markers)

        legend_rows = ''.join(f"<div><span style='color:{color}'>&#9679;</span> {level}</div>"
                              for level, color in zip(PRIORITY_LEVELS, PRIORITY_COLORS))
        legend = (f"<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:white;"
                  f"color:black;padding:8px 12px;border-radius:4px;font-size:13px'>"
                  f"<b>Приоритет проверки</b>{legend_rows}"
                  f"<div><span style='color:gray'>&#9679;</span> вне снимка</div>"
                  f"<div>! риск: асбестоцемент у воды</div></div>")
        m.get_root().html.add_child(folium.Element(legend))

        folium.LayerControl().add_to(m)
        m.fit_bounds([[south, west], [north, east]])
        path = os.path.join(self.output_dir, filename)
        m.save(path)
        print(f"Совмещенная карта сохранена: {path}")
        return m


def create_test_geojson(output_path='inspections.geojson'):
    test_data = {
        "type": "FeatureCollection",
        "name": "test_inspections",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [47.5215, 45.3342]
                },
                "properties": {
                    "id": "INSP_001",
                    "timestamp": datetime.now().isoformat(),
                    "material": "asbestos",
                    "confidence": 0.95,
                    "damage_percent": 30
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [48.1582, 45.7783]
                },
                "properties": {
                    "id": "INSP_002",
                    "timestamp": datetime.now().isoformat(),
                    "material": "hdp",
                    "confidence": 0.92,
                    "damage_percent": 15
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [48.6449, 45.1091]
                },
                "properties": {
                    "id": "INSP_003",
                    "timestamp": datetime.now().isoformat(),
                    "material": "concrete",
                    "confidence": 0.88,
                    "damage_percent": 45
                }
            }
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Создан тестовый файл: {output_path}")
    return output_path


def create_near_water_geojson(output_path='inspections_near_water.geojson'):
    points = [
        ('INSP_101', [48.66763, 45.79092], 'asbestos', 0.93, 30),
        ('INSP_102', [48.60696, 45.78629], 'hdp', 0.90, 15),
        ('INSP_103', [48.55969, 45.73419], 'asbestos', 0.89, 65),
        ('INSP_104', [48.69975, 45.69582], 'asbestos', 0.91, 20),
    ]
    test_data = {
        "type": "FeatureCollection",
        "name": "test_inspections_near_water",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coordinates},
                "properties": {
                    "id": inspection_id,
                    "timestamp": datetime.now().isoformat(),
                    "material": material,
                    "confidence": confidence,
                    "damage_percent": damage_percent
                }
            }
            for inspection_id, coordinates, material, confidence, damage_percent in points
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Создан тестовый файл: {output_path}")
    return output_path
