import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.configuration import ConfigurationManager
from components.data_ingestion import DataIngestion
from custom_logger import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initializes the Data Ingestion Pipeline.
        """
        pass

    def main(self):
        """
        Orchestrates the data ingestion process:
        1. Load Configuration
        2. Initialize Component
        3. Execute Download/Extraction
        """
        try:
            # 1. Configuration Management
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            # 2. Component Initialization
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            # 3. Execution
            logger.info("Starting Data Ingestion stage...")
            data_ingestion.download_file()
            logger.info("Data Ingestion stage completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in Data Ingestion Pipeline: {str(e)}")
            raise e

if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e