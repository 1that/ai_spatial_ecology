import os
import webbrowser
from datetime import datetime
from pathlib import Path
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from jinja2 import Environment, FileSystemLoader
from matplotlib.patches import Patch
from report_translations import DATE_FORMATS, TRANSLATIONS, VALUE_TRANSLATIONS
from water_quality_pipeline import WaterQualityPipeline


class WaterQualityPipelineAnalytics(WaterQualityPipeline):
    """Анализ качества воды: подготовка проб, ПДК и расчет ИЗВ."""

    # ПДК для рыбохозяйственных водоемов (Приказ Росрыболовства от 26.05.2025 № 296)
    PDK = {
        # Растворенный кислород: не менее 6.0 мг/дм³ для водоемов высшей/первой категорий (летом).
        # В зимний период норма ниже -- 4.0 мг/дм³.
        'oxygen': {'min': 6.0, 'type': 'min'},
        # БПК5: Биохимическое потребление кислорода за 5 суток.
        # Это интегральный показатель органического загрязнения. Лимит -- 3.0 мг O₂/дм³.
        'BOD5': 3.0,
        # Аммоний-ион (NH4+): Токсичный продукт распада органики.
        # ПДК составляет 0.5 мг/дм³ (в пересчете на азот аммонийный -- 0.4 мг/дм³).
        'NH4': 0.5,
        # Нитрит-ион (NO2-): Высокотоксичен для рыб (нарушает перенос кислорода кровью).
        # ПДК строго ограничена -- 0.08 мг/дм³.
        'NO2': 0.08,
        # Нитрат-ион (NO3-): Здесь указано значение 9.0 мг/дм³, что соответствует ПДК
        # в пересчете на азот (N). ПДК по самому иону NO3 составляет 40.0 мг/дм³.
        'NO3': 9.0,
        # Фосфаты (PO4 3-): Главный фактор «цветения» воды (эвтрофикации).
        # ПДК -- 0.2 мг/дм³ (в пересчете на фосфор -- 0.066 мг/дм³).
        'PO4': 0.2,
        # Нефтепродукты: Суммарное содержание масел, бензина и др.
        # ПДК -- 0.05 мг/дм³. Превышение дает характерную пленку и запах.
        'oil': 0.05
    }

    # Шесть основных показателей для ИЗВ (кислород, БПК5, NH4, NO2, NO3, нефтепродукты)
    IZV_PARAMS = ['oxygen', 'BOD5', 'NH4', 'NO2', 'NO3', 'oil']

    # Условные значения для показателей, которых нет в исходных данных
    DEFAULT_VALUES = {
        'NO2': 0.05,
        'NO3': 0.5,
        'BOD5': 2.0,
        'oil': 0.03,
        'oxygen': 7.0
    }

    # Цветовая шкала классов качества воды
    CLASS_COLORS = {
        'I (очень чистая)': '#0000FF',
        'II (чистая)': '#00FF00',
        'III (умеренно загрязненная)': '#FFFF00',
        'IV (загрязненная)': '#FFA500',
        'V (грязная)': '#FF0000',
        'VI (очень грязная)': '#8B0000',
        'VII (чрезвычайно грязная)': '#000000'
    }

    # Показатели для гистограмм и их подписи на графиках
    HISTOGRAM_TITLES = {
        'NH4': 'Аммоний (NH₄)',
        'NO2': 'Нитриты (NO₂⁻)',
        'NO3': 'Нитраты (NO₃⁻)',
        'PO4': 'Фосфаты (PO₄³⁻)',
        'BOD5': 'БПК₅',
        'oil': 'Нефтепродукты'
    }

    # Условные коэффициенты формулы хлорофилла-а, выбраны для демонстрации метода.
    # Не откалиброваны по измерениям, поэтому значения в мг/м³ ориентировочные.
    CHL_A_COEFFS = {'a': 0.5, 'b': -2.0, 'c': 0.5}

    # Названия загрязнителей для отчета
    POLLUTANT_NAMES = {
        'NH4': 'аммоний',
        'NO2': 'нитриты',
        'NO3': 'нитраты',
        'PO4': 'фосфаты',
        'BOD5': 'БПК5',
        'oil': 'нефтепродукты'
    }
    
    CONTRIBUTION_STYLES = {
        'NH4': {'name': 'Аммоний (NH₄)', 'color': '#D7263D'},
        'NO2': {'name': 'Нитриты (NO₂)', 'color': '#8E44AD'},
        'NO3': {'name': 'Нитраты (NO₃)', 'color': '#2E86DE'},
        'BOD5': {'name': 'БПК₅', 'color': '#E67E22'},
        'oil': {'name': 'Нефтепродукты', 'color': '#2C3E50'},
        'oxygen': {'name': 'Кислород (O₂)', 'color': '#16A085'}
    }

    MIXED_CONTRIBUTION_COLOR = '#B0B0B0'
    
    DOMINANT_SHARE_THRESHOLD = 80

    # Во сколько раз концентрация должна превысить ПДК, чтобы строка отчета выделялась цветом
    EXCEEDANCE_FACTOR = 2

    def __init__(self,
                 name="Аналитический пайплайн",
                 data_path='class_data/points_with_clusters.geojson',
                 s2_path='data/delta_volga.tif'):
        super().__init__(name)
        self.train_dir = 'reports'
        self.data_path = data_path
        self.s2_path = s2_path
        self.pdk = self.PDK
        self.samples_gdf = None
        self.samples_gdf_wgs84 = None
        self.izv_contributions = None
        self.rgb = None
        self.ndvi = None
        self.ndwi = None
        self.chl_a_estimate = None
        self.chl_a_water = None
        self.report_data = None
        self.html_content = None
        self.export_gdf = None

    def load_data(self):
        """Загружает точки проб из практической работы № 2 и подготавливает колонки."""
        samples_gdf = gpd.read_file(self.data_path)
        print(f"Загружено {len(samples_gdf)} точек проб")
        print(f"Колонки: {samples_gdf.columns.tolist()}")

        # Преобразуем названия колонок
        rename_dict = {
            'NH4_mgL': 'NH4',
            'PO4_mgL': 'PO4',
            'pH': 'pH'
        }
        samples_gdf = samples_gdf.rename(columns={k: v for k, v in rename_dict.items() if k in samples_gdf.columns})

        # Добавляем недостающие колонки
        for col, value in self.DEFAULT_VALUES.items():
            if col not in samples_gdf.columns:
                samples_gdf[col] = value

        # Добавляем даты
        if 'date' not in samples_gdf.columns:
            base_date = datetime(2026, 3, 1)
            samples_gdf['date'] = [
                base_date - pd.Timedelta(days=np.random.randint(0, 90))
                for _ in range(len(samples_gdf))
            ]

        print(f"\nПервые 5 записей:")

        self.samples_gdf = samples_gdf

    def show_pdk(self):
        """Выводит значения ПДК."""
        for param, value in self.pdk.items():
            print(f"  {param}: {value}")

    def check_data(self):
        """Проверяет готовность данных к расчету."""
        print(f"\nВсего проб: {len(self.samples_gdf)}")
        print(f"Необходимые колонки присутствуют:")
        required_cols = ['NH4', 'NO2', 'NO3', 'BOD5', 'oil', 'oxygen']
        for col in required_cols:
            if col in self.samples_gdf.columns:
                print(f"  {col}")
            else:
                print(f"  {col} - ОТСУТСТВУЕТ!")

        print(f"\nПример данных для расчета:")
        print(self.samples_gdf[['id', 'NH4', 'NO2', 'NO3', 'BOD5', 'oil', 'oxygen']].head())

    def calculate_ratios(self, row):
        """Возвращает словарь отношений Cᵢ/ПДКᵢ по показателям ИЗВ для одной пробы."""
        ratios = {}

        for param in self.IZV_PARAMS:
            if param not in row or param not in self.pdk:
                print(f"⚠️ Параметр {param} не найден, пропускаем")
                continue

            value = row[param]
            pdk_value = self.pdk[param]

            if param == 'oxygen':
                # Для кислорода формула обратная: чем меньше, тем хуже
                # Используем min (ПДК: не ниже 6.0 мг/л)
                if value >= pdk_value['min']:
                    ratio = 1.0
                else:
                    ratio = pdk_value['min'] / max(value, 0.1)  # защита от деления на 0
            else:
                # Для остальных: превышение ПДК дает ratio > 1
                ratio = value / pdk_value

            ratios[param] = ratio

        return ratios

    def calculate_izv(self, row):
        """
        Расчет индекса загрязнения воды (ИЗВ) по шести показателям.

        Параметры:
        - row: строка DataFrame с концентрациями

        Возвращает:
        - izv: значение ИЗВ
        - quality_class: класс качества (I-VII)
        """
        ratios = self.calculate_ratios(row)

        if len(ratios) == 0:
            return None, "Н/Д (нет данных)"

        izv = np.mean(list(ratios.values()))
        return izv, self.get_quality_class(izv)

    def get_quality_class(self, izv):
        """Определяет класс качества воды по значению ИЗВ."""
        if izv < 0.3:
            return "I (очень чистая)"
        elif izv < 1.0:
            return "II (чистая)"
        elif izv < 2.5:
            return "III (умеренно загрязненная)"
        elif izv < 4.0:
            return "IV (загрязненная)"
        elif izv < 6.0:
            return "V (грязная)"
        elif izv < 10.0:
            return "VI (очень грязная)"
        return "VII (чрезвычайно грязная)"

    def calculate_izv_all(self):
        """Применяет расчет ИЗВ ко всем пробам и добавляет результаты в GeoDataFrame."""
        print("\nРасчет ИЗВ для всех проб...")
        izv_results = []
        quality_classes = []

        for idx, row in self.samples_gdf.iterrows():
            izv, qclass = self.calculate_izv(row)
            izv_results.append(izv)
            quality_classes.append(qclass)

        self.samples_gdf['IZV'] = izv_results
        self.samples_gdf['quality_class'] = quality_classes

    def show_izv_results(self):
        """Выводит результаты расчета ИЗВ и статистику."""
        samples_gdf = self.samples_gdf

        # Проверяем, сколько проб получили корректный ИЗВ
        valid_count = samples_gdf['IZV'].notna().sum()
        print(f"\nРассчитан ИЗВ для {valid_count} из {len(samples_gdf)} проб")

        # Показываем первые 10 проб
        print("\nПервые 10 проб:")
        display_cols = ['id', 'NH4', 'NO2', 'NO3', 'BOD5', 'oil', 'oxygen', 'IZV', 'quality_class']
        available_cols = [col for col in display_cols if col in samples_gdf.columns]
        print(samples_gdf[available_cols].head(10).to_string(index=False))

        # Статистика по классам качества
        print("\nРаспределение по классам качества:")
        class_counts = samples_gdf['quality_class'].value_counts()
        for qclass, count in class_counts.items():
            percentage = count / valid_count * 100
            print(f"  {qclass}: {count} проб ({percentage:.1f}%)")

        # Статистика по ИЗВ
        print(f"\nСтатистика ИЗВ:")
        print(f"  Минимальный ИЗВ: {samples_gdf['IZV'].min():.3f}")
        print(f"  Максимальный ИЗВ: {samples_gdf['IZV'].max():.3f}")
        print(f"  Средний ИЗВ: {samples_gdf['IZV'].mean():.3f}")
        print(f"  Медианный ИЗВ: {samples_gdf['IZV'].median():.3f}")

        # Пробы с высоким загрязнением (ИЗВ > 2.5)
        high_risk = samples_gdf[samples_gdf['IZV'] > 2.5]
        if len(high_risk) > 0:
            print(f"\nПробы с высоким уровнем загрязнения (ИЗВ > 2.5): {len(high_risk)} шт.")
            print(high_risk[['id', 'NH4', 'NO2', 'IZV', 'quality_class']].head(10).to_string(index=False))

        # Пробы с критическим загрязнением (ИЗВ > 4.0)
        critical = samples_gdf[samples_gdf['IZV'] > 4.0]
        if len(critical) > 0:
            print(f"\nКРИТИЧЕСКИЕ пробы (ИЗВ > 4.0): {len(critical)} шт.")
            print(critical[['id', 'NH4', 'NO2', 'IZV', 'quality_class']].head(10).to_string(index=False))

    def convert_to_wgs84(self):
        """Преобразует координаты проб в WGS84, если данные в другой системе координат."""
        if self.samples_gdf.crs != 'EPSG:4326':
            print("Преобразование координат из UTM в WGS84...")
            self.samples_gdf_wgs84 = self.samples_gdf.to_crs('EPSG:4326')
            print("Координаты преобразованы!")
        else:
            self.samples_gdf_wgs84 = self.samples_gdf
            print("Данные уже в WGS84")

    def create_izv_map(self, output_file='izv_map.html'):
        """Создает интерактивную карту ИЗВ, сохраняет ее в HTML и открывает в браузере."""
        samples_gdf_wgs84 = self.samples_gdf_wgs84

        # Получаем границы для центрирования
        min_lat = samples_gdf_wgs84.geometry.y.min()
        max_lat = samples_gdf_wgs84.geometry.y.max()
        min_lon = samples_gdf_wgs84.geometry.x.min()
        max_lon = samples_gdf_wgs84.geometry.x.max()
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        print(f"\n📐 Центр карты: {center_lat:.4f}, {center_lon:.4f}")
        print(f"  Широта: {min_lat:.4f} - {max_lat:.4f}")
        print(f"  Долгота: {min_lon:.4f} - {max_lon:.4f}")

        # Создаем карту
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')

        # Добавляем спутниковый слой
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник (Esri)'
        ).add_to(m)

        points_added = 0
        print("\nДобавление точек на карту...")

        for idx, row in samples_gdf_wgs84.iterrows():
            if pd.isna(row['IZV']):
                continue

            lat = row.geometry.y
            lon = row.geometry.x

            # Проверяем, что координаты в разумных пределах для дельты Волги
            if not (45.0 <= lat <= 46.0 and 47.0 <= lon <= 49.0):
                print(f"Точка {row.id}: координаты вне области ({lat:.4f}, {lon:.4f})")
                continue

            color = self.CLASS_COLORS.get(row['quality_class'], '#808080')
            radius = 5 + row['IZV'] * 1.5
            radius = min(radius, 15)

            popup_text = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <b>Проба №{row.id}</b><br>
                <hr>
                <b>ИЗВ:</b> {row.IZV:.3f}<br>
                <b>Класс:</b> {row.quality_class}<br>
                <hr>
                <b>Загрязнители:</b><br>
                &nbsp;&nbsp;NH₄: {row.NH4:.3f} мг/л<br>
                &nbsp;&nbsp;NO₂: {row.NO2:.3f} мг/л<br>
                &nbsp;&nbsp;O₂: {row.oxygen:.2f} мг/л
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Проба {row.id}: ИЗВ={row.IZV:.2f}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            points_added += 1

        print(f"\nДобавлено точек: {points_added}")

        # Легенда
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
            background-color: white; padding: 10px; border-radius: 5px;
            border: 2px solid grey;">
            <b>Классы качества</b><br>
            <span style="color:#0000FF;">●</span> I (очень чистая)<br>
            <span style="color:#00FF00;">●</span> II (чистая)<br>
            <span style="color:#FFFF00;">●</span> III (умеренно загрязненная)<br>
            <span style="color:#FFA500;">●</span> IV (загрязненная)<br>
            <span style="color:#FF0000;">●</span> V (грязная)<br>
            <span style="color:#8B0000;">●</span> VI (очень грязная)<br>
            <span style="color:#000000;">●</span> VII (чрезвычайно грязная)<br>
            <hr>
            <i>Всего точек: {points_added}</i>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)

        # Сохраняем
        html_path = self.output_dir + output_file
        m.save(html_path)
        print(f"\nКарта сохранена: {html_path}")
        webbrowser.open('file://' + os.path.realpath(html_path))

    def get_translations(self, lang='ru'):
        """Возвращает словари фраз и значений для выбранного языка."""
        if lang not in TRANSLATIONS:
            raise ValueError(f"Язык {lang} не поддерживается. Доступные языки: {list(TRANSLATIONS)}")
        return TRANSLATIONS[lang], VALUE_TRANSLATIONS.get(lang, {})

    def get_figure_file(self, name, lang='ru'):
        """Возвращает имя файла рисунка с суффиксом языка, например izv_map_static_en.png."""
        return f'{name}_{lang}.png'

    def finish_figure(self, fig, output_file, show):
        """Сохраняет рисунок и показывает его на экране или закрывает без показа."""
        plt.savefig(self.output_dir + output_file, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def create_static_izv_map(self, output_file=None, lang='ru', show=True):
        """Создает статическую карту ИЗВ на выбранном языке и сохраняет ее в файл."""
        t, values = self.get_translations(lang)
        output_file = output_file or self.get_figure_file('izv_map_static', lang)
        samples_gdf_wgs84 = self.samples_gdf_wgs84

        print("\nСоздание статической карты...")
        fig, ax = plt.subplots(figsize=(12, 10))

        for idx, row in samples_gdf_wgs84.iterrows():
            if pd.isna(row['IZV']):
                continue
            color = self.CLASS_COLORS.get(row['quality_class'], 'gray')
            ax.scatter(row.geometry.x, row.geometry.y, c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(t['fig_longitude'])
        ax.set_ylabel(t['fig_latitude'])
        ax.set_title(t['fig_static_map_title'], fontsize=14)
        ax.grid(True, alpha=0.3)

        legend_elements = [Patch(facecolor=color, label=values.get(label, label), alpha=0.7)
                           for label, color in self.CLASS_COLORS.items()
                           if label in samples_gdf_wgs84['quality_class'].unique()]
        ax.legend(handles=legend_elements, loc='upper right', title=t['quality_class'])

        plt.tight_layout()
        self.finish_figure(fig, output_file, show)
        print(f"Статическая карта сохранена: {output_file}")

    def calculate_contributions(self):
        """Рассчитывает долю каждого показателя в сумме отношений Cᵢ/ПДКᵢ и основной показатель пробы."""
        ratios = pd.DataFrame(
            [self.calculate_ratios(row) for _, row in self.samples_gdf.iterrows()],
            index=self.samples_gdf.index
        )
        shares = ratios.div(ratios.sum(axis=1), axis=0) * 100

        contributions = ratios.add_prefix('ratio_').join(shares.add_prefix('share_'))
        contributions['main_param'] = shares.idxmax(axis=1)
        contributions['main_share'] = shares.max(axis=1)
        contributions['is_dominant'] = contributions['main_share'] > self.DOMINANT_SHARE_THRESHOLD

        self.izv_contributions = contributions

    def get_cluster_statistics(self):
        """Возвращает статистику ИЗВ и вкладов показателей по кластерам."""
        data = self.samples_gdf[['cluster', 'IZV']].join(self.izv_contributions)
        share_cols = [f'share_{param}' for param in self.IZV_PARAMS]

        statistics = data.groupby('cluster').agg(
            samples=('IZV', 'size'),
            mean_izv=('IZV', 'mean'),
            dominant_samples=('is_dominant', 'sum')
        )
        mean_shares = data.groupby('cluster')[share_cols].mean()
        statistics['main_param'] = mean_shares.idxmax(axis=1).str.replace('share_', '', regex=False)
        statistics['main_share'] = mean_shares.max(axis=1)
        return statistics

    def show_contribution_statistics(self):
        """Выводит статистику вкладов показателей в ИЗВ."""
        contributions = self.izv_contributions
        dominant = contributions[contributions['is_dominant']]

        print(f"\nПроб с долей одного показателя более {self.DOMINANT_SHARE_THRESHOLD}%: "
              f"{len(dominant)} из {len(contributions)}")
        for param, count in dominant['main_param'].value_counts().items():
            print(f"  {self.CONTRIBUTION_STYLES[param]['name']}: {count}")

        print("\nСтатистика по кластерам:")
        print(self.get_cluster_statistics().round(2).to_string())

    def create_contribution_tooltip(self, row, contribution, cluster_row):
        """Формирует HTML всплывающей подсказки с вкладом показателей в ИЗВ."""
        lines = ''
        shares = sorted(self.IZV_PARAMS, key=lambda param: contribution[f'share_{param}'], reverse=True)

        for param in shares:
            style = self.CONTRIBUTION_STYLES[param]
            share = contribution[f'share_{param}']
            lines += f"""
            <div style="margin-top: 4px;">
                {style['name']}: <b>{share:.1f}%</b> (C/ПДК = {contribution[f'ratio_{param}']:.2f})
                <div style="background: #EEEEEE; height: 6px; width: 220px;">
                    <div style="background: {style['color']}; height: 6px; width: {share:.1f}%;"></div>
                </div>
            </div>
            """

        return f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px;">
            <b>Проба №{row['id']}</b> · кластер {row['cluster']}<br>
            ИЗВ: <b>{row['IZV']:.2f}</b> · {row['quality_class']}<br>
            Средний ИЗВ кластера: {cluster_row['mean_izv']:.2f} ({int(cluster_row['samples'])} проб)
            <hr style="margin: 4px 0;">
            <b>Вклад в индекс:</b>
            {lines}
        </div>
        """

    def create_contribution_map(self, output_file='izv_contribution_map.html'):
        """Создает интерактивную карту вкладов показателей в ИЗВ и статистики по кластерам."""
        if self.samples_gdf_wgs84 is None:
            self.convert_to_wgs84()

        samples = self.samples_gdf_wgs84
        contributions = self.izv_contributions
        cluster_statistics = self.get_cluster_statistics()

        m = folium.Map(tiles='OpenStreetMap')
        m.fit_bounds([
            [samples.geometry.y.min(), samples.geometry.x.min()],
            [samples.geometry.y.max(), samples.geometry.x.max()]
        ])
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник (Esri)'
        ).add_to(m)

        for idx, row in samples.iterrows():
            contribution = contributions.loc[idx]
            cluster_row = cluster_statistics.loc[row['cluster']]
            color = self.MIXED_CONTRIBUTION_COLOR
            if contribution['is_dominant']:
                color = self.CONTRIBUTION_STYLES[contribution['main_param']]['color']

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=min(5 + row['IZV'] * 1.5, 15),
                tooltip=folium.Tooltip(self.create_contribution_tooltip(row, contribution, cluster_row)),
                color='#333333',
                fill=True,
                fillColor=color,
                fillOpacity=0.85,
                weight=1
            ).add_to(m)

        # Легенда содержит только показатели, которые преобладают хотя бы в одной пробе
        dominant_params = contributions.loc[contributions['is_dominant'], 'main_param'].unique()
        legend_rows = ''.join(
            f'<span style="color:{style["color"]};">●</span> {style["name"]}<br>'
            for param, style in self.CONTRIBUTION_STYLES.items()
            if param in dominant_params
        )
        legend_html = f'''
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; font-size: 13px;
            background-color: white; padding: 10px; border-radius: 5px; border: 2px solid grey;">
            <b>Основной показатель (&gt; {self.DOMINANT_SHARE_THRESHOLD}%)</b><br>
            {legend_rows}
            <span style="color:{self.MIXED_CONTRIBUTION_COLOR};">●</span> Нет преобладающего показателя<br>
            <i>Размер точки зависит от ИЗВ</i>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        cluster_rows = ''.join(
            f"<tr><td>{cluster}</td><td>{int(item['samples'])}</td><td>{item['mean_izv']:.2f}</td>"
            f"<td>{self.CONTRIBUTION_STYLES[item['main_param']]['name']} ({item['main_share']:.0f}%)</td>"
            f"<td>{int(item['dominant_samples'])}</td></tr>"
            for cluster, item in cluster_statistics.iterrows()
        )
        cluster_html = f'''
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000; font-size: 13px;
            background-color: white; padding: 10px; border-radius: 5px; border: 2px solid grey;">
            <b>Статистика по кластерам</b>
            <table cellpadding="4" style="border-collapse: collapse; margin-top: 4px;">
                <tr><th>Кластер</th><th>Проб</th><th>Средний ИЗВ</th>
                    <th>Наибольший средний вклад</th><th>Точек &gt; {self.DOMINANT_SHARE_THRESHOLD}%</th></tr>
                {cluster_rows}
            </table>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(cluster_html))
        folium.LayerControl().add_to(m)

        html_path = self.output_dir + output_file
        m.save(html_path)
        print(f"\nКарта вкладов сохранена: {html_path}")
        webbrowser.open('file://' + os.path.realpath(html_path))

    def plot_pollutant_histograms(self, output_file=None, lang='ru', show=True):
        """Строит гистограммы концентраций загрязняющих веществ на выбранном языке и сохраняет их в файл."""
        t, _ = self.get_translations(lang)
        output_file = output_file or self.get_figure_file('pollutants_distribution', lang)
        samples_gdf = self.samples_gdf

        # Проверяем, какие параметры есть в данных
        available_params = [p for p in self.HISTOGRAM_TITLES if p in samples_gdf.columns]
        print(f"\nДоступные параметры для визуализации: {available_params}")

        # Создаем сетку графиков
        n_params = len(available_params)
        n_rows = (n_params + 2) // 3  # округление вверх
        n_cols = min(3, n_params)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        print("\nПостроение гистограмм...")

        for i, param in enumerate(available_params):
            ax = axes[i]

            # Данные для гистограммы
            data = samples_gdf[param].dropna()

            # Определяем количество бинов (если данных мало, уменьшаем)
            n_bins = min(20, len(np.unique(data)) // 2 + 1)
            n_bins = max(n_bins, 5)

            # Строим гистограмму
            ax.hist(data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)

            # Добавляем линию ПДК (если есть)
            if param in self.pdk:
                pdk_value = self.pdk[param]
                if isinstance(pdk_value, dict):
                    # Для кислорода особый случай
                    if param == 'oxygen':
                        pdk_value = pdk_value.get('min', 6.0)
                if isinstance(pdk_value, (int, float)):
                    ax.axvline(x=pdk_value, color='red', linestyle='--',
                               linewidth=2, label=t['fig_mpc_label'] % pdk_value)

            # Добавляем линию среднего значения
            mean_val = data.mean()
            ax.axvline(x=mean_val, color='orange', linestyle='-',
                       linewidth=1.5, label=t['fig_mean_label'] % f'{mean_val:.3f}')

            # Добавляем линию медианы
            median_val = data.median()
            ax.axvline(x=median_val, color='green', linestyle=':',
                       linewidth=1.5, label=t['fig_median_label'] % f'{median_val:.3f}')

            # Настройки графика
            ax.set_title(t[f'fig_hist_{param}'], fontsize=12, fontweight='bold')
            ax.set_xlabel(t['fig_concentration_mgl'], fontsize=10)
            ax.set_ylabel(t['sample_count'], fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Добавляем статистику на график
            stats_text = f'n = {len(data)}\nmax = {data.max():.3f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Скрываем лишние оси
        for i in range(len(available_params), len(axes)):
            axes[i].axis('off')

        plt.suptitle(t['fig_hist_title'], fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.finish_figure(fig, output_file, show)
        print(f"Гистограммы сохранены: {output_file}")

    def calculate_chlorophyll(self):
        """Читает снимок Sentinel-2, рассчитывает NDVI, NDWI и оценку хлорофилла-а."""
        with rasterio.open(self.s2_path) as src:
            # Читаем необходимые каналы
            # Sentinel-2: B2(Blue), B3(Green), B4(Red), B5(Red Edge), B8(NIR)
            blue = src.read(3).astype(np.float32)      # B2 - синий
            green = src.read(2).astype(np.float32)     # B3 - зеленый
            red = src.read(1).astype(np.float32)       # B4 - красный
            nir = src.read(4).astype(np.float32)       # B8 - ближний инфракрасный
            red_edge = src.read(6).astype(np.float32)  # B5 - красный край

            print(f"  Размер снимка: {src.width} x {src.height}")
            print(f"  Каналы: B2(Blue), B3(Green), B4(Red), B5(Red Edge), B8(NIR)")

        # Нормализация RGB для отображения (растяжка 2-98 процентили)
        def stretch_rgb(band):
            p2, p98 = np.percentile(band, (2, 98))
            return np.clip((band - p2) / (p98 - p2), 0, 1)

        self.rgb = np.stack([stretch_rgb(red), stretch_rgb(green), stretch_rgb(blue)], axis=-1)

        # Расчет индексов для оценки эвтрофикации
        # NDVI (вегетационный индекс)
        self.ndvi = (nir - red) / (nir + red + 1e-10)

        # NDWI (водный индекс)
        self.ndwi = (green - nir) / (green + nir + 1e-10)

        # Оценка хлорофилла-а по эмпирической формуле с условными коэффициентами
        # Chl-a = 10^(a + b*log10(R) + c*(log10(R))^2), где R = B3/B5
        # Пиксели с нулем в B3 или B5 остаются NaN, так как для них log10(R) не определен
        valid = (green > 0) & (red_edge > 0)
        log_r = np.full(green.shape, np.nan, dtype=np.float32)
        log_r[valid] = np.log10(green[valid] / red_edge[valid])

        a, b, c = self.CHL_A_COEFFS['a'], self.CHL_A_COEFFS['b'], self.CHL_A_COEFFS['c']
        chl_a_estimate = 10 ** (a + b * log_r + c * log_r ** 2)
        self.chl_a_estimate = np.clip(chl_a_estimate, 0, 30)

        # Оставляем только водные объекты (где NDWI > 0.2)
        water_mask = self.ndwi > 0.2
        self.chl_a_water = self.chl_a_estimate.copy()
        self.chl_a_water[~water_mask] = np.nan  # маскируем сушу

        print(f"\nСтатистика NDWI:")
        print(f"  Минимум: {self.ndwi.min():.3f}")
        print(f"  Максимум: {self.ndwi.max():.3f}")
        print(f"  Водная поверхность: {water_mask.sum() / water_mask.size * 100:.1f}%")

    def plot_chlorophyll_analysis(self, output_file=None, lang='ru', show=True):
        """Строит RGB, NDVI, NDWI, карты и гистограмму хлорофилла-а на выбранном языке, сохраняет рисунок в файл."""
        t, _ = self.get_translations(lang)
        output_file = output_file or self.get_figure_file('chlorophyll_analysis', lang)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. RGB снимок
        axes[0, 0].imshow(self.rgb)
        axes[0, 0].set_title(t['fig_rgb'], fontsize=12)
        axes[0, 0].axis('off')

        # 2. NDVI (вегетация)
        im1 = axes[0, 1].imshow(self.ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0, 1].set_title(t['fig_ndvi'], fontsize=12)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        axes[0, 1].axis('off')

        # 3. NDWI (вода)
        im2 = axes[0, 2].imshow(self.ndwi, cmap='Blues', vmin=-0.5, vmax=0.5)
        axes[0, 2].set_title(t['fig_ndwi'], fontsize=12)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        axes[0, 2].axis('off')

        # 4. Хлорофилл-а (оценка)
        im3 = axes[1, 0].imshow(self.chl_a_estimate, cmap='viridis', vmin=0, vmax=15)
        axes[1, 0].set_title(t['fig_chl_estimate'], fontsize=12)
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        axes[1, 0].axis('off')

        # 5. Хлорофилл-а только на воде
        im4 = axes[1, 1].imshow(self.chl_a_water, cmap='viridis', vmin=0, vmax=15)
        axes[1, 1].set_title(t['fig_chl_water'], fontsize=12)
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].axis('off')

        # 6. Гистограмма хлорофилла на воде
        chl_values = self.chl_a_water[~np.isnan(self.chl_a_water)]
        axes[1, 2].hist(chl_values, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 2].set_title(t['fig_chl_hist'], fontsize=12)
        axes[1, 2].set_xlabel(t['fig_concentration_mgm3'])
        axes[1, 2].set_ylabel(t['fig_pixel_count'])

        # Добавляем линии трофических статусов
        axes[1, 2].axvline(x=1, color='blue', linestyle='--', linewidth=2, label=t['fig_oligotrophic'])
        axes[1, 2].axvline(x=3, color='orange', linestyle='--', linewidth=2, label=t['fig_mesotrophic'])
        axes[1, 2].axvline(x=10, color='red', linestyle='--', linewidth=2, label=t['fig_eutrophic'])
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        self.finish_figure(fig, output_file, show)
        print(f"Анализ хлорофилла сохранен: {output_file}")

    def show_chlorophyll_statistics(self):
        """Выводит статистику хлорофилла-а на водных объектах."""
        chl_a_water = self.chl_a_water

        mean_chl = np.nanmean(chl_a_water)
        median_chl = np.nanmedian(chl_a_water)
        std_chl = np.nanstd(chl_a_water)

        print(f"\nХлорофилл-а (только водные объекты):")
        print(f"  Среднее: {mean_chl:.2f} ± {std_chl:.2f} мг/м³")
        print(f"  Медиана: {median_chl:.2f} мг/м³")
        print(f"  Минимум: {np.nanmin(chl_a_water):.2f} мг/м³")
        print(f"  Максимум: {np.nanmax(chl_a_water):.2f} мг/м³")

    def get_trophic_status(self, chl):
        """Определяет трофический статус по концентрации хлорофилла-а."""
        if chl < 1:
            return "Олиготрофный (чистая вода)"
        elif chl < 3:
            return "Мезотрофный (умеренная продуктивность)"
        elif chl < 10:
            return "Эвтрофный (высокая продуктивность, риск цветения)"
        return "Гиперэвтрофный (цветение воды, критическое состояние)"

    def show_trophic_status(self):
        """Выводит трофический статус по среднему и распределение воды по статусам."""
        chl_values = self.chl_a_water[~np.isnan(self.chl_a_water)]

        trophic = self.get_trophic_status(chl_values.mean())
        print(f"\nТрофический статус (по среднему): {trophic}")

        # Распределение по статусам
        print(f"\nРаспределение водной поверхности по трофическим статусам:")
        oligo_pct = np.sum(chl_values < 1) / len(chl_values) * 100
        meso_pct = np.sum((chl_values >= 1) & (chl_values < 3)) / len(chl_values) * 100
        eu_pct = np.sum((chl_values >= 3) & (chl_values < 10)) / len(chl_values) * 100
        hyper_pct = np.sum(chl_values >= 10) / len(chl_values) * 100

        print(f"  Олиготрофный (< 1 мг/м³): {oligo_pct:.1f}%")
        print(f"  Мезотрофный (1-3 мг/м³): {meso_pct:.1f}%")
        print(f"  Эвтрофный (3-10 мг/м³): {eu_pct:.1f}%")
        print(f"  Гиперэвтрофный (> 10 мг/м³): {hyper_pct:.1f}%")

    def collect_chlorophyll_data(self):
        """Собирает данные эвтрофикации для отчета. Возвращает None, если хлорофилл не рассчитан."""
        if self.chl_a_water is None or np.all(np.isnan(self.chl_a_water)):
            return None

        chl_v = self.chl_a_water[~np.isnan(self.chl_a_water)]
        mean_chl = float(np.mean(chl_v))
        chlorophyll_data = {
            'mean_chl': mean_chl,
            'oligo_pct': (chl_v < 1).sum() / len(chl_v) * 100,
            'meso_pct': ((chl_v >= 1) & (chl_v < 3)).sum() / len(chl_v) * 100,
            'eu_pct': ((chl_v >= 3) & (chl_v < 10)).sum() / len(chl_v) * 100,
            'hyper_pct': (chl_v >= 10).sum() / len(chl_v) * 100,
            'trophic_status': self.get_trophic_status(mean_chl)
        }
        print(f"Эвтрофикация: {chlorophyll_data['trophic_status']} ({mean_chl:.2f} мг/м³)")
        return chlorophyll_data

    def get_exceeded_params(self, row):
        """Возвращает вещества пробы, концентрация которых превышает ПДК более чем в EXCEEDANCE_FACTOR раз."""
        exceeded = []
        for param, name in self.POLLUTANT_NAMES.items():
            if param in row and row[param] > self.pdk[param] * self.EXCEEDANCE_FACTOR:
                exceeded.append({'name': name, 'ratio': float(row[param] / self.pdk[param])})
        return exceeded

    def collect_attachments(self, lang='ru'):
        """Возвращает приложения к отчету, для которых есть данные и созданные файлы."""
        attachments = [
            ('attachment_geojson', 'reports/water_quality.geojson'),
            ('attachment_csv', 'reports/water_quality.csv'),
            ('attachment_izv_map', self.output_dir + 'izv_map.html'),
            ('attachment_contribution_map', self.output_dir + 'izv_contribution_map.html')
        ]
        if self.report_data['chlorophyll_analysis']:
            attachments.append(('attachment_chlorophyll',
                                self.output_dir + self.get_figure_file('chlorophyll_analysis', lang)))

        return [{'key': key, 'path': path} for key, path in attachments if os.path.exists(path)]

    def prepare_report_data(self, chlorophyll_data=None):
        """
        Подготовка данных для шаблона отчета.

        Параметры:
        - chlorophyll_data: словарь с данными эвтрофикации (опционально)

        Возвращает:
        - report_data: словарь с подготовленными данными для шаблона
        """
        samples_gdf = self.samples_gdf
        # Координаты для отчета в градусах, так как исходные данные в UTM
        samples_wgs84 = samples_gdf.to_crs('EPSG:4326')

        print("\nРасчет статистики для отчета...")

        def format_date(row):
            return row.date.strftime('%Y-%m-%d') if hasattr(row.date, 'strftime') else str(row.date)

        # Основная статистика
        total_points = samples_gdf['id'].nunique()
        total_samples = len(samples_gdf)
        print(f"  - Всего точек: {total_points}")
        print(f"  - Всего проб: {total_samples}")

        # Статистика по классам качества
        class_stats = {}
        for class_name, group in samples_gdf.groupby('quality_class'):
            class_stats[class_name] = {
                'count': len(group),
                'avg_izv': group['IZV'].mean()
            }
        print(f"  - Классов качества: {len(class_stats)}")

        # Пробы с высоким риском
        high_risk = samples_gdf[samples_gdf['IZV'] > 4.0]
        high_risk_samples = []
        for _, row in high_risk.iterrows():
            high_risk_samples.append({
                'id': int(row.id),
                'date': format_date(row),
                'IZV': float(row.IZV)
            })
        print(f"  - Проб с высоким риском (ИЗВ > 4): {len(high_risk_samples)}")

        # Пробы с повышенным загрязнением
        elevated_risk = samples_gdf[(samples_gdf['IZV'] > 2.5) & (samples_gdf['IZV'] <= 4.0)]
        elevated_risk_samples = []
        for _, row in elevated_risk.iterrows():
            elevated_risk_samples.append({
                'id': int(row.id),
                'date': format_date(row),
                'IZV': float(row.IZV)
            })
        print(f"  - Проб с повышенным загрязнением (2.5 < ИЗВ ≤ 4): {len(elevated_risk_samples)}")

        # Преобладающий класс
        dominant_class = samples_gdf['quality_class'].mode()[0] if len(
            samples_gdf['quality_class'].mode()) > 0 else "III (умеренно загрязненная)"
        print(f"  - Преобладающий класс: {dominant_class}")

        # Определение основных загрязнителей
        main_pollutants_list = []
        for param, name in self.POLLUTANT_NAMES.items():
            if param in samples_gdf.columns and samples_gdf[param].mean() > self.pdk[param]:
                main_pollutants_list.append(name)
        main_pollutants_str = ', '.join(main_pollutants_list) if main_pollutants_list else 'превышений ПДК не выявлено'
        print(f"  - Основные загрязнители: {main_pollutants_str}")

        # Доля превышений ПДК
        exceedance_count = 0
        total_checks = 0
        for param in ['NH4', 'NO2', 'NO3', 'PO4']:
            if param in samples_gdf.columns:
                exceedance_count += (samples_gdf[param] > self.pdk[param]).sum()
                total_checks += len(samples_gdf)
        exceedance_percent = (exceedance_count / total_checks * 100) if total_checks > 0 else 0
        print(f"  - Доля превышений ПДК: {exceedance_percent:.1f}%")

        # Данные для таблицы (первые 20 проб для отчета)
        samples_list = []
        for _, row in samples_wgs84.head(20).iterrows():
            samples_list.append({
                'id': int(row.id),
                'date': format_date(row),
                'lat': float(row.geometry.y),
                'lon': float(row.geometry.x),
                'NH4': float(row.NH4),
                'NO2': float(row.NO2),
                'IZV': float(row.IZV),
                'quality_class': row.quality_class,
                'exceeded_params': self.get_exceeded_params(row)
            })

        # Статистика ИЗВ
        izv_stats = {
            'min': float(samples_gdf['IZV'].min()),
            'max': float(samples_gdf['IZV'].max()),
            'mean': float(samples_gdf['IZV'].mean()),
            'median': float(samples_gdf['IZV'].median()),
            'std': float(samples_gdf['IZV'].std())
        }

        # Данные эвтрофикации (если есть)
        chlorophyll_analysis = chlorophyll_data is not None
        if chlorophyll_analysis:
            mean_chl = chlorophyll_data.get('mean_chl', 0)
            trophic_status = chlorophyll_data.get('trophic_status', 'нет данных')
            oligo_pct = chlorophyll_data.get('oligo_pct', 0)
            meso_pct = chlorophyll_data.get('meso_pct', 0)
            eu_pct = chlorophyll_data.get('eu_pct', 0)
            hyper_pct = chlorophyll_data.get('hyper_pct', 0)
        else:
            mean_chl = 0
            trophic_status = 'нет данных'
            oligo_pct = meso_pct = eu_pct = hyper_pct = 0

        # Сборка всех данных
        self.report_data = {
            # Общая информация
            'report_id': f"ЭКО-{datetime.now().strftime('%Y%m%d')}-001",
            'report_date': datetime.now().strftime('%d.%m.%Y'),
            'object_name': 'Водные объекты дельты Волги',
            'start_date': samples_gdf['date'].min().strftime('%d.%m.%Y') if hasattr(samples_gdf['date'].min(), 'strftime') else '01.03.2026',
            'end_date': samples_gdf['date'].max().strftime('%d.%m.%Y') if hasattr(samples_gdf['date'].max(), 'strftime') else '31.03.2026',
            'executor': 'СПбГЭТУ «ЛЭТИ», кафедра МОЭВМ',

            # Статистика
            'total_points': total_points,
            'total_samples': total_samples,
            'avg_izv': float(samples_gdf['IZV'].mean()),
            'dominant_class': dominant_class,
            'izv_stats': izv_stats,

            # Загрязнители
            'main_pollutants': main_pollutants_str,
            'main_pollutants_list': main_pollutants_list,
            'exceedance_percent': exceedance_percent,
            'exceedance_factor': self.EXCEEDANCE_FACTOR,

            # Пробы
            'samples': samples_list,
            'class_stats': class_stats,
            'high_risk_samples': high_risk_samples,
            'elevated_risk_samples': elevated_risk_samples,
            'high_risk_count': len(high_risk_samples),

            # Эвтрофикация
            'chlorophyll_analysis': chlorophyll_analysis,
            'mean_chl': mean_chl,
            'trophic_status': trophic_status,
            'oligo_pct': oligo_pct,
            'meso_pct': meso_pct,
            'eu_pct': eu_pct,
            'hyper_pct': hyper_pct,

            # Данные для карты (координаты)
            'bounds': {
                'min_lat': float(samples_wgs84.geometry.y.min()),
                'max_lat': float(samples_wgs84.geometry.y.max()),
                'min_lon': float(samples_wgs84.geometry.x.min()),
                'max_lon': float(samples_wgs84.geometry.x.max())
            }
        }
        return self.report_data

    def show_report_summary(self):
        """Выводит краткий итог подготовки данных отчета."""
        report_data = self.report_data
        print(f"""
            Итоги:
            - Проб: {report_data['total_samples']} (ИЗВ средний: {report_data['avg_izv']:.2f})
            - Класс: {report_data['dominant_class']}
            - Загрязнители: {report_data['main_pollutants']}
            - Высокий риск (ИЗВ > 4): {len(report_data['high_risk_samples'])}
            - Спутниковый мониторинг: {'Доступен' if report_data['chlorophyll_analysis'] else 'нет данных'}
        """)
        print("Все данные собраны в объект 'report_data'.")

    def get_report_path(self, extension, lang='ru'):
        """Возвращает путь к файлу отчета с суффиксом языка: report_ru.pdf, report_en.pdf."""
        return f'reports/report_{lang}.{extension}'

    def format_report_date(self, value, lang='ru'):
        """Переводит дату из report_data (ДД.ММ.ГГГГ или ГГГГ-ММ-ДД) в формат выбранного языка."""
        for source_format in ('%d.%m.%Y', '%Y-%m-%d'):
            try:
                date = datetime.strptime(str(value), source_format)
                break
            except ValueError:
                continue
        else:
            return value

        date_format = DATE_FORMATS[lang]
        months = date_format.get('months', [])
        return date_format['format'].format(
            day=date.day,
            month_num=date.month,
            month=months[date.month - 1] if months else date.month,
            year=date.year
        )

    def generate_report_figures(self, lang='ru'):
        """Строит рисунки отчета на выбранном языке без показа на экране."""
        if self.samples_gdf_wgs84 is None:
            self.convert_to_wgs84()
        self.plot_pollutant_histograms(lang=lang, show=False)
        self.create_static_izv_map(lang=lang, show=False)
        # Рисунок хлорофилла строится, только если calculate_chlorophyll() уже выполнен
        if self.chl_a_water is not None:
            self.plot_chlorophyll_analysis(lang=lang, show=False)

    def generate_html_report(self, template_dir='templates',
                             template_name='report_template.html',
                             output_file=None,
                             lang='ru'):
        """Заполняет HTML-шаблон данными report_data на выбранном языке и сохраняет готовый отчет."""
        t, value_translations = self.get_translations(lang)

        output_file = output_file or self.get_report_path('html', lang)
        print(f"\nГенерация HTML-отчета ({lang})...")

        # Загружаем шаблон
        env = Environment(loader=FileSystemLoader(template_dir))
        env.filters['translate'] = lambda value: value_translations.get(value, value)
        env.filters['format_date'] = lambda value: self.format_report_date(value, lang)
        template = env.get_template(template_name)

        # Рендерим HTML
        # Приложения определяются здесь, так как файлы экспорта могут быть созданы после подготовки данных
        self.html_content = template.render(
            **self.report_data,
            attachments=self.collect_attachments(lang),
            t=t,
            lang=lang
        )

        # Сохраняем HTML для проверки
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(self.html_content)

        print(f"HTML-отчет сохранен: {output_file}")

    def generate_pdf_report(self, html_file=None, output_file=None, lang='ru'):
        """Создает PDF-отчет из HTML-файла, сохраненного в generate_html_report()."""
        html_file = html_file or self.get_report_path('html', lang)
        output_file = output_file or self.get_report_path('pdf', lang)

        from weasyprint import HTML
        HTML(filename=html_file).write_pdf(output_file)
        print(f"PDF-отчет сохранен: {output_file}")



    def generate_docx_report(self, output_file='reports/report.docx'):
        """Создает отчет в формате DOCX из данных report_data."""
        data = self.report_data
        doc = Document()

        # Заголовок
        title = doc.add_heading('ОТЧЕТ ПО ЭКОЛОГИЧЕСКОМУ МОНИТОРИНГУ', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"№ {data['report_id']} от {data['report_date']}")

        # Раздел 1
        doc.add_heading('1. ОБЩИЕ СВЕДЕНИЯ', level=2)
        doc.add_paragraph(f"Объект: {data['object_name']}")
        doc.add_paragraph(f"Период: {data['start_date']} - {data['end_date']}")
        doc.add_paragraph(f"Всего проб: {data['total_samples']}")
        doc.add_paragraph(f"Средний ИЗВ: {data['avg_izv']:.2f}")
        doc.add_paragraph(f"Преобладающий класс: {data['dominant_class']}")

        # Раздел 2
        doc.add_heading('2. РЕЗУЛЬТАТЫ АНАЛИЗА', level=2)

        # Таблица
        table = doc.add_table(rows=1, cols=6)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'ID'
        hdr_cells[1].text = 'Дата'
        hdr_cells[2].text = 'NH₄ (мг/л)'
        hdr_cells[3].text = 'NO₂ (мг/л)'
        hdr_cells[4].text = 'ИЗВ'
        hdr_cells[5].text = 'Класс'

        # Добавляем данные (первые 20 проб)
        for s in data['samples'][:20]:
            row_cells = table.add_row().cells
            row_cells[0].text = str(s['id'])
            row_cells[1].text = s['date']
            row_cells[2].text = f"{s['NH4']:.3f}"
            row_cells[3].text = f"{s['NO2']:.3f}"
            row_cells[4].text = f"{s['IZV']:.2f}"
            row_cells[5].text = s['quality_class']

        # Раздел 3: Статистика
        doc.add_heading('3. СТАТИСТИКА ПО КЛАССАМ КАЧЕСТВА', level=2)
        for class_name, stats in data['class_stats'].items():
            doc.add_paragraph(f"{class_name}: {stats['count']} проб (средний ИЗВ = {stats['avg_izv']:.2f})")

        # Раздел 4: Загрязнители
        doc.add_heading('4. ОСНОВНЫЕ ЗАГРЯЗНИТЕЛИ', level=2)
        doc.add_paragraph(f"Основные загрязнители: {data['main_pollutants']}")
        doc.add_paragraph(f"Доля превышений ПДК: {data['exceedance_percent']:.1f}%")

        # Раздел 5: Риски
        if data['high_risk_samples']:
            doc.add_heading('5. ПРОБЫ С ВЫСОКИМ РИСКОМ (ИЗВ > 4)', level=2)
            for s in data['high_risk_samples']:
                doc.add_paragraph(f"Проба №{s['id']} от {s['date']}: ИЗВ = {s['IZV']:.2f}")

        # Раздел 6: Эвтрофикация
        if data['chlorophyll_analysis']:
            doc.add_heading('6. АНАЛИЗ ЭВТРОФИКАЦИИ', level=2)
            doc.add_paragraph(f"Средняя концентрация хлорофилла-а: {data['mean_chl']:.2f} мг/м³")
            doc.add_paragraph(f"Трофический статус: {data['trophic_status']}")
            doc.add_paragraph(
                f"Распределение: олиготрофный: {data['oligo_pct']:.1f}%, "
                f"мезотрофный: {data['meso_pct']:.1f}%, "
                f"эвтрофный: {data['eu_pct']:.1f}%, "
                f"гиперэвтрофный: {data['hyper_pct']:.1f}%"
            )

        # Раздел 7: Выводы
        doc.add_heading('7. ВЫВОДЫ И РЕКОМЕНДАЦИИ', level=2)
        doc.add_paragraph(f"1. По результатам мониторинга качество воды соответствует классу {data['dominant_class']}.")
        doc.add_paragraph(f"2. Основными загрязнителями являются: {data['main_pollutants']}.")
        if data['high_risk_samples']:
            doc.add_paragraph(
                f"3. Выявлено {len(data['high_risk_samples'])} проб с высоким уровнем загрязнения (ИЗВ > 4)."
            )
        if data['chlorophyll_analysis']:
            doc.add_paragraph(
                f"4. Трофический статус водных объектов по спутниковым данным: {data['trophic_status']}."
            )
        doc.add_paragraph("5. Рекомендуется провести дополнительные исследования в точках с повышенным ИЗВ.")

        # Подпись
        doc.add_paragraph()
        doc.add_paragraph("Начальник отдела экологического мониторинга")
        doc.add_paragraph("____________________ / И.И. Иванов/")
        doc.add_paragraph(f"Дата: {data['report_date']}")

        # Сохраняем
        doc.save(output_file)
        print(f"DOCX-отчет сохранен: {output_file}")

    def prepare_gis_export(self):
        """Готовит данные для ГИС: координаты WGS84, цвет, категория ИЗВ и уровень риска."""
        # Создаем копию для экспорта и переводим в WGS84 (градусы)
        export_gdf = self.samples_gdf.to_crs('EPSG:4326').copy()

        # Добавляем цветовую кодировку для отображения в ГИС
        if 'quality_class' in export_gdf.columns:
            export_gdf['color'] = export_gdf['quality_class'].map(self.CLASS_COLORS).fillna('#808080')

        print(f"Добавлена цветовая кодировка для {len(export_gdf)} объектов")

        # Добавляем дополнительные поля для ГИС-анализа
        # right=False: границы интервалов те же, что в get_quality_class()
        export_gdf['izv_category'] = pd.cut(
            export_gdf['IZV'],
            bins=[0, 0.3, 1.0, 2.5, 4.0, 6.0, 10.0, float('inf')],
            labels=['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'],
            right=False
        ).astype(str)

        export_gdf['risk_level'] = 'низкий'
        export_gdf.loc[export_gdf['IZV'] > 2.5, 'risk_level'] = 'повышенный'
        export_gdf.loc[export_gdf['IZV'] > 4.0, 'risk_level'] = 'высокий'

        print(f"Категории риска: {export_gdf['risk_level'].value_counts().to_dict()}")

        self.export_gdf = export_gdf

    def export_geojson(self, output_file='reports/water_quality.geojson',
                       simple_output_file='reports/water_quality_simple.geojson'):
        """Сохраняет полную и упрощенную версии данных в формате GeoJSON."""
        # Сохраняем GeoJSON
        self.export_gdf.to_file(output_file, driver='GeoJSON')
        print(f"GeoJSON сохранен: {output_file}")

        # Дополнительно: создаем упрощенную версию с основными полями для быстрого просмотра
        export_gdf_simple = self.export_gdf[['id', 'IZV', 'quality_class', 'risk_level', 'color', 'geometry']].copy()
        export_gdf_simple.to_file(simple_output_file, driver='GeoJSON')
        print(f"Упрощенный GeoJSON сохранен: {simple_output_file}")

    def export_csv(self, output_file='reports/water_quality.csv'):
        """Сохраняет данные проб в CSV с координатами в градусах и русскими названиями колонок."""
        # Создаем DataFrame без геометрии
        export_df = self.samples_gdf.drop(columns='geometry').copy()

        # Добавляем координаты в градусах (исходные данные в UTM, поэтому сначала переводим в WGS84)
        samples_wgs84 = self.samples_gdf.to_crs('EPSG:4326')
        export_df['longitude'] = samples_wgs84.geometry.x
        export_df['latitude'] = samples_wgs84.geometry.y

        # Форматируем дату
        if 'date' in export_df.columns:
            export_df['date_str'] = pd.to_datetime(export_df['date']).dt.strftime('%Y-%m-%d')
            export_df = export_df.drop(columns=['date'])

        # Переименовываем колонки для удобства
        column_names = {
            'id': 'ID_пробы',
            'NH4': 'Аммоний_NH4_мг_л',
            'NO2': 'Нитриты_NO2_мг_л',
            'NO3': 'Нитраты_NO3_мг_л',
            'PO4': 'Фосфаты_PO4_мг_л',
            'BOD5': 'БПК5_мг_л',
            'oil': 'Нефтепродукты_мг_л',
            'oxygen': 'Растворенный_кислород_мг_л',
            'location': 'Локация',
            'IZV': 'ИЗВ',
            'quality_class': 'Класс_качества',
            'longitude': 'Долгота',
            'latitude': 'Широта',
            'date_str': 'Дата_отбора'
        }
        export_df = export_df.rename(columns={k: v for k, v in column_names.items() if k in export_df.columns})

        # Добавляем категорию риска
        export_df['Категория_риска'] = 'низкий'
        export_df.loc[export_df['ИЗВ'] > 2.5, 'Категория_риска'] = 'повышенный'
        export_df.loc[export_df['ИЗВ'] > 4.0, 'Категория_риска'] = 'высокий'

        # Сохраняем CSV
        export_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"CSV-файл сохранен: {output_file}")
        print(f"Размер: {len(export_df)} строк, {len(export_df.columns)} столбцов")

    def generate_report(self, output_formats=['pdf', 'geojson', 'csv'], lang='ru'):
        """
        Генерация полного отчета

        Параметры:
        - output_formats: список форматов из 'pdf', 'geojson', 'csv'
        - lang: язык отчета, 'ru' или 'en'

        Возвращает:
        - data: словарь с подготовленными данными отчета
        """
        if self.samples_gdf is None:
            self.load_data()

        if 'IZV' not in self.samples_gdf.columns:
            self.calculate_izv_all()

        os.makedirs('reports', exist_ok=True)

        # Данные о хлорофилле попадут в отчет, если calculate_chlorophyll() уже выполнен
        data = self.prepare_report_data(self.collect_chlorophyll_data())

        # Генерация в разных форматах
        if 'geojson' in output_formats:
            # Экспорт в GeoJSON (сохраняем геометрию)
            self.prepare_gis_export()
            self.export_geojson()

        if 'csv' in output_formats:
            # Экспорт в CSV (без геометрии)
            self.export_csv()

        if 'pdf' in output_formats:
            # PDF создается из HTML-версии отчета
            self.generate_report_figures(lang)
            self.generate_html_report(lang=lang)
            self.generate_pdf_report(lang=lang)

        return data
