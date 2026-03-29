from utils.common import read_yaml, create_directories
from pathlib import Path
from custom_logger import logger
from entity import DataIngestionConfig, DataTransformationConfig, ModelTrainingConfig, SRGANTrainingConfig

CONFIG_FILE_PATH = Path("config/config.yaml")   

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

        create_directories([self.config.data_root_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Create the root directory for data ingestion artifacts
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir), # data/raw
            dataset_name=config.dataset.name,
            dataset_root=Path(config.dataset.root_dir),
            train_hr_dir=Path(config.dataset.train.hr_dir),
            train_lr_dir=Path(config.dataset.train.lr_dir),
            val_hr_dir=Path(config.dataset.val.hr_dir),
            val_lr_dir=Path(config.dataset.val.lr_dir),
            # --- Set5 Mapping ---
            test_dataset_name=config.test_dataset.name,
            test_dataset_root=Path(config.test_dataset.root_dir),
            test_hr_dir=Path(config.test_dataset.hr_dir),
            test_lr_dir=Path(config.test_dataset.lr_dir),
            # --------------------
            download_enabled=config.download.enabled,
            download_urls=config.download.urls,
            processing_scale=config.processing.scale,
            processing_patch_size=config.processing.patch_size,
            processing_batch_size=config.processing.batch_size,
            processing_num_workers=config.processing.num_workers
        )
        logger.info(f"Data Ingestion config object created")
        
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            srcnn_dir = Path(config.srcnn_dir),
            srgan_dir = Path(config.srgan_dir),
            patch_size = config.params.patch_size,
            stride = config.params.stride,
            scale = config.params.scale
        )

        logger.info(f"Data Transformation config object created")
        
        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config.model_training
        model_type = training_config.trained_model_choice
        model_params = training_config[model_type]
        
        create_directories([training_config.root_dir])

        return ModelTrainingConfig(
            root_dir=Path(training_config.root_dir),
            train_data_path=Path(model_params.train_data),
            valid_data_path=Path(model_params.valid_data),
            model_path=Path(model_params.model_path),
            model_type=model_type,
            epochs=model_params.params.epochs,
            batch_size=model_params.params.batch_size,
            learning_rate=model_params.params.lr,
            normalization=model_params.params.normalization,
            # --- Updated mappings below ---
            device=training_config.device,
            log_step=training_config.log_step,
            patience=training_config.patience
        )

    def get_srgan_training_config(self) -> SRGANTrainingConfig:
        training_config = self.config.model_training
        model_params = training_config.srgan
        
        create_directories([training_config.root_dir])

        return SRGANTrainingConfig(
            root_dir=Path(training_config.root_dir),
            train_data_path=Path(model_params.train_data),
            valid_data_path=Path(model_params.valid_data),
            model_path_g=Path(model_params.model_path_g),
            model_path_d=Path(model_params.model_path_d),
            pretrain_epochs=model_params.params.pretrain_epochs,
            epochs=model_params.params.epochs,
            batch_size=model_params.params.batch_size,
            learning_rate_g=model_params.params.lr_g,
            learning_rate_d=model_params.params.lr_d,
            normalization=model_params.params.normalization,
            device=training_config.device,
            log_step=training_config.log_step,
            patience=training_config.patience
        )