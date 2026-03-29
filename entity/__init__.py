from dataclasses import dataclass
from pathlib import Path

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_root: Path
    train_hr_dir: Path
    train_lr_dir: Path
    val_hr_dir: Path
    val_lr_dir: Path
    # --- New Set5 Fields ---
    test_dataset_name: str
    test_dataset_root: Path
    test_hr_dir: Path
    test_lr_dir: Path
    # -----------------------
    download_enabled: bool
    download_urls: dict
    processing_scale: int
    processing_patch_size: int
    processing_batch_size: int
    processing_num_workers: int


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    srcnn_dir: Path
    srgan_dir: Path
    patch_size: int
    stride: int
    scale: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    valid_data_path: Path
    model_path: Path
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    normalization: str
    device: str      # Moved in YAML
    log_step: int    # New
    patience: int    # New


@dataclass(frozen=True)
class SRGANTrainingConfig:
    root_dir: Path
    train_data_path: Path
    valid_data_path: Path
    model_path_g: Path
    model_path_d: Path
    pretrain_epochs: int
    epochs: int
    batch_size: int
    learning_rate_g: float
    learning_rate_d: float
    normalization: str
    device: str
    log_step: int
    patience: int