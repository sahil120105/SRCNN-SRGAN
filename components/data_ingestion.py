import os
import urllib.request
import zipfile
import time
from pathlib import Path
from tqdm import tqdm
from custom_logger import logger
from entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if self._data_exists():
            logger.info("All required dataset directories exist and are populated. Skipping download phase.")
            return
            
        start_time = time.time()
        
        for name, url in self.config.download_urls.items():
            zip_path = Path(self.config.root_dir) / f"{name}.zip"
            extract_to = self.config.test_dataset_root if name == "set5" else self.config.dataset_root
            
            # Individual Skip Check (same as before)
            check_dir = self.config.test_hr_dir if name == "set5" else self.config.train_hr_dir
            if check_dir.exists() and any(check_dir.iterdir()):
                logger.info(f"Skipping {name}: Data already exists.")
                continue

            try:
                os.makedirs(extract_to, exist_ok=True)
                self._download_with_progress(url, zip_path, name)
                
                # New Recursive Extraction Logic
                self._perform_extraction(zip_path, extract_to)
                
                logger.info(f"Successfully processed {name}")
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                if zip_path.exists(): os.remove(zip_path)

        logger.info(f"Ingestion finished in {time.time() - start_time:.2f}s")

    def _perform_extraction(self, zip_path: Path, extract_to: Path):
        """Extracts a zip and checks if it contained more zips to extract."""
        logger.info(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Cleanup the initial zip
        if zip_path.exists():
            os.remove(zip_path)

        # RECURSIVE CHECK: Look for any .zip or .url files created inside the folder
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                file_path = Path(root) / file
                if file.endswith('.zip'):
                    logger.info(f"Found nested zip: {file}. Extracting...")
                    with zipfile.ZipFile(file_path, 'r') as nested_ref:
                        nested_ref.extractall(root)
                    os.remove(file_path) # Delete the nested zip
                
                elif file.endswith('.url') or 'Startcrack' in file:
                    logger.info(f"Removing junk file: {file}")
                    os.remove(file_path)

    def _download_with_progress(self, url: str, filepath: Path, name: str):
        """Downloads a file with a visual progress bar and custom headers."""
        
        # Define a browser-like User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        request = urllib.request.Request(url, headers=headers)

        def progress_hook(t):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b
            return update_to

        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {name}") as t:
            # We use urlopen instead of urlretrieve to support headers more easily
            with urllib.request.urlopen(request) as response, open(filepath, 'wb') as out_file:
                tsize = int(response.info().get('Content-Length', 0))
                t.total = tsize
                
                bsize = 1024 * 8
                block_count = 0
                while True:
                    block = response.read(bsize)
                    if not block:
                        break
                    out_file.write(block)
                    block_count += 1
                    t.update(len(block))

    def _data_exists(self) -> bool:
        required_paths = [
            self.config.train_hr_dir, 
            self.config.val_hr_dir, 
            self.config.test_hr_dir
        ]
        
        status = True
        for p in required_paths:
            abs_p = p.resolve()
            if not p.exists():
                logger.warning(f"MISSING: {abs_p}")
                status = False
            elif not any(p.iterdir()):
                logger.warning(f"EMPTY: {abs_p}")
                status = False
            else:
                # Observability: Count files found
                count = len(list(p.glob('*')))
                logger.info(f"VERIFIED: {p.name} contains {count} files.")
        
        return status