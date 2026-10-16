import ee
import geopandas as gpd

ee.Authenticate()
ee.Initialize(project='ecology-511714')
#      широта     долгота
pt1 = (45.807538, 48.564195)
pt2 = (45.717241, 48.743566)
pt3 = (45.660140, 48.549928)

data = {
    'name': ['Point 1', 'Point 2', 'Point 3'],
    'lon': [pt1[1], pt2[1], pt3[1]], # долгота
    'lat': [pt1[0], pt2[0], pt3[0]]  # широта
}


gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['lon'], data['lat']),
    crs="EPSG:4326"
)

min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2025-06-01', '2025-08-01')
    .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', 30))
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
)

size = collection.size().getInfo()
print(f"Найдено снимков: {size}")

if size > 0:
    image = collection.first()
    cloudiness = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"Облачность лучшего снимка: {cloudiness}%")

    selected_bands = image.select(['B4', 'B3', 'B2', 'B8', 'B11'])

    task = ee.batch.Export.image.toDrive(
        image=selected_bands,
        description='sentinel2_spb',
        folder='gee_exports',
        fileNamePrefix='sentinel2_16bit_data',
        region=roi,
        scale=10,
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()
    print("Задача экспорта запущена!")
    print("После завершения скачайте снимок в папку data/")
