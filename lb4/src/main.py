from inference_benchmark import run_inference_benchmark
from mobile_pipeline import create_near_water_geojson, create_test_geojson, MobilePipeline
from save_result import save_pract5_results
import os


def task_2_3():
    run_inference_benchmark()


def task_5_3(mobile_pipeline: MobilePipeline):
    mobile_pipeline.import_mobile_data(create_near_water_geojson())
    mobile_pipeline.load_water_bodies()
    mobile_pipeline.integrate_with_segmentation()
    mobile_pipeline.visualize_map()


def main():
    geojson_file = 'inspections.geojson'
    if not os.path.exists(geojson_file):
        create_test_geojson(geojson_file)

    mobile_pipeline = MobilePipeline()

    imported_data = mobile_pipeline.import_mobile_data(geojson_file)

    if imported_data is not None:
        mobile_pipeline.analyze_mobile_data()
    else:
        print("Не удалось импортировать данные. Проверьте путь к файлу.")

    task_5_3(mobile_pipeline)

    task_2_3()

    save_pract5_results()


if __name__ == '__main__':
    main()
