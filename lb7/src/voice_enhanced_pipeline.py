import os
import re
from datetime import datetime
import geopandas as gpd
import librosa
import pandas as pd
from water_quality_pipeline import WaterQualityPipeline


def parse_coordinates(text):
    normalized = text.replace(',', '.')
    numbers = re.findall(r'\d{2,3}\.\d{1,4}', normalized)
    if not numbers:
        numbers = re.findall(r'\d{1,3}', normalized)
    print(f"  Найденные числа: {numbers}")
    if not numbers:
        return None
    if '.' in numbers[0]:
        if len(numbers) < 2:
            return None
        lat, lon = float(numbers[0]), float(numbers[1])
    else:
        if len(numbers) < 4:
            return None
        lat = float(f'{numbers[0]}.{numbers[1]}')
        lon = float(f'{numbers[2]}.{numbers[3]}')
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {'lat': lat, 'lon': lon}


def parse_voice_report(text):
    result = {
        'timestamp': datetime.now().isoformat(),
        'coordinates': None,
        'measurements': {},
        'material': None,
        'damage': None,
        'problems': [],
        'raw_text': text
    }
    print(f"\nРазбор текста: {text[:80]}...")
    result['coordinates'] = parse_coordinates(text)
    if result['coordinates']:
        print(f"  Координаты: {result['coordinates']['lat']}, {result['coordinates']['lon']}")
    patterns = {
        'NH4': r'(?:аммоний|аммония|аммунии|аммони|NH4)\s*[.:\s]*(\d{1,2}[.,]?\d*)',
        'PO4': r'(?:фосфаты|фосфатов|фасфаты|PO4)\s*[.:\s]*(\d{1,2}[.,]?\d*)',
        'pH': r'(?:pH|пэ-аш|пх)\s*[.:\s]*(\d{1,2}[.,]?\d*)'
    }
    for param, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(',', '.'))
                result['measurements'][param] = value
                print(f"  {param}: {value}")
            except:
                pass
    materials = {
        'асбест': 'asbestos',
        'сбестовая': 'asbestos',
        'сбест': 'asbestos',
        'пнд': 'hdp',
        'бетон': 'concrete',
        'керамика': 'ceramics',
        'металл': 'metal'
    }
    for material_ru, material_en in materials.items():
        if material_ru in text.lower():
            result['material'] = material_en
            print(f"  Материал: {material_en}")
            break
    damage_match = re.search(r'(\d+)\s*(?:%|процент)', text, re.IGNORECASE)
    if damage_match:
        result['damage'] = int(damage_match.group(1))
        print(f"  Повреждение: {result['damage']}%")
    return result


class VoiceEnhancedPipeline(WaterQualityPipeline):
    def __init__(self, name="Voice-Enhanced Pipeline"):
        super().__init__(name)
        self.asr_model = None
        self.voice_samples = []

    def init_voice_module(self):
        print("Инициализация речевого модуля...")
        try:
            import whisper
            self.asr_model = whisper.load_model("small")
            print("  Whisper загружен")
        except ImportError:
            print("  Whisper не установлен")
            self.asr_model = None
        print("Речевой модуль готов")

    def add_sample_by_voice(self, audio_path):
        if self.asr_model is None:
            print("Речевой модуль не инициализирован")
            return None
        if not os.path.exists(audio_path):
            print(f"Файл {audio_path} не найден")
            return None
        print(f"\nРаспознавание: {os.path.basename(audio_path)}")
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            result = self.asr_model.transcribe(audio, language='ru')
            text = result['text']
            print(f"  Текст: {text}")
        except Exception as e:
            print(f"  Ошибка: {e}")
            return None
        data = parse_voice_report(text)
        if data['coordinates']:
            new_id = len(self.samples) + 1 if self.samples is not None else 1
            new_sample = pd.DataFrame({
                'id': [new_id],
                'lat': [data['coordinates']['lat']],
                'lon': [data['coordinates']['lon']],
                'name': [f'Голосовая_{new_id}'],
                'NH4_mgL': [data['measurements'].get('NH4', 0.5)],
                'PO4_mgL': [data['measurements'].get('PO4', 0.3)],
                'pH': [data['measurements'].get('pH', 7.0)],
                'source': ['voice'],
                'original_text': [text]
            })
            geometry = gpd.points_from_xy(
                [data['coordinates']['lon']],
                [data['coordinates']['lat']]
            )
            new_gdf = gpd.GeoDataFrame(new_sample, geometry=geometry, crs='EPSG:4326')
            if self.samples is None:
                self.samples = new_gdf
            else:
                self.samples = pd.concat([self.samples, new_gdf], ignore_index=True)
            print(f"  Добавлена проба #{new_id} ({data['coordinates']['lat']}, {data['coordinates']['lon']})")
            self.voice_samples.append({
                'id': new_id,
                'audio': audio_path,
                'text': text,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            print(f"  Координаты не найдены")
        return data

    def get_voice_stats(self):
        return {
            'total': len(self.voice_samples),
            'with_coords': sum(1 for s in self.voice_samples if s['data']['coordinates']),
            'samples': self.voice_samples
        }
