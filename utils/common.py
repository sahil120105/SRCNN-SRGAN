import os
from box.exceptions import BoxValueError
import yaml
from custom_logger import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



@ensure_annotations         # Ensures function arguments and return types follow the declared type hints at runtime
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns

    Args:
        path_to_yaml (Path): Path like input

    Returns:
        ConfigBox: ConfigBox type object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")

            # Convert dictionary into ConfigBox for dot-notation access (config.key instead of config["key"])
            return ConfigBox(content)

    except BoxValueError:
        # Raised by ConfigBox if YAML is empty or invalid (e.g., safe_load returns None)
        raise ValueError("yaml file is empty")

    except Exception as e:
        logger.exception(f"Error occured while reading yaml file: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories

    Args:
        path_to_directories (list): List of path of directories
        verbose (bool, optional): _description_. Defaults to True.
    """
    for path_to_directory in path_to_directories:
        os.makedirs(path_to_directory, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path_to_directory}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Get size of file

    Args:
        path (Path): Path like input

    Returns:
        str: size in kb
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"



class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0