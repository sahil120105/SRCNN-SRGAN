from config.configuration import ConfigurationManager
from components.model_training import ModelTraining
from custom_logger import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Executes the Model Training lifecycle:
        1. Load specific model config (SRCNN or SRGAN)
        2. Initialize Training Engine
        3. Start Training Loop
        """
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_training = ModelTraining(config=training_config)
        
        logger.info(f"--- Stage: {training_config.model_type.upper()} Training Initialized ---")
        model_training.train()
        logger.info(f"--- Stage: {training_config.model_type.upper()} Training Completed ---")

if __name__ == "__main__":
    STAGE_NAME = "Model Training"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e