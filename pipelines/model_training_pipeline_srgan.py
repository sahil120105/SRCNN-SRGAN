from config.configuration import ConfigurationManager
from components.model_training_srgan import SRGANTraining
from custom_logger import logger

class SRGANTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Executes the SRGAN Model Training lifecycle:
        1. Load SRGAN specific config
        2. Initialize Training Engine
        3. Start Training Loop
        """
        config_manager = ConfigurationManager()
        training_config = config_manager.get_srgan_training_config()
        model_training = SRGANTraining(config=training_config)
        
        logger.info(f"--- Stage: SRGAN Training Initialized ---")
        model_training.train()
        logger.info(f"--- Stage: SRGAN Training Completed ---")

if __name__ == "__main__":
    STAGE_NAME = "SRGAN Model Training"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = SRGANTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
