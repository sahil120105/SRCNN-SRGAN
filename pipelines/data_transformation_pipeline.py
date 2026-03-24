from config.configuration import ConfigurationManager
from components.data_transformation import DataTransformation
from custom_logger import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        
        logger.info("Starting Data Transformation for SRCNN...")
        data_transformation.create_srcnn_data(split='train')
        data_transformation.create_srcnn_data(split='valid')
        logger.info("Data Transformation complete.")

if __name__ == "__main__":
    STAGE_NAME = "Data Transformation Stage"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e