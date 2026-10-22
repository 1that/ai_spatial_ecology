from water_quality_pipeline_extended import WaterQualityPipelineExtended


def get_data_set(pipeline: WaterQualityPipelineExtended):
    return pipeline.create_dataset_from_pract2(stride=64)


def main():
    pipeline = WaterQualityPipelineExtended()
    get_data_set(pipeline)
    pipeline.visualize_dataset(n_samples=4)

if __name__ == '__main__':
    main()
