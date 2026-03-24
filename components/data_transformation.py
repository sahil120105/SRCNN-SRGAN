import os
import cv2
import h5py
import numpy as np
from pathlib import Path
from custom_logger import logger
from entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def create_srcnn_data(self, split='train'):
        hr_path = os.path.join(self.config.data_path, f"DIV2K_{split}_HR")
        lr_path = os.path.join(self.config.data_path, f"DIV2K_{split}_LR_bicubic", "X4")
        
        save_path = Path(self.config.srcnn_dir) / f"srcnn_{split}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Create the HDF5 file with resizable datasets
        with h5py.File(save_path, 'w') as h5_file:
            # maxshape=(None, ...) allows the dataset to grow indefinitely
            lr_ds = h5_file.create_dataset("lr", shape=(0, self.config.patch_size, self.config.patch_size, 3), 
                                          maxshape=(None, self.config.patch_size, self.config.patch_size, 3),
                                          dtype='uint8', chunks=True)
            hr_ds = h5_file.create_dataset("hr", shape=(0, self.config.patch_size, self.config.patch_size, 3), 
                                          maxshape=(None, self.config.patch_size, self.config.patch_size, 3),
                                          dtype='uint8', chunks=True)

            img_list = os.listdir(hr_path)
            total_patches = 0

            for i, img_name in enumerate(img_list):
                hr_img = cv2.imread(os.path.join(hr_path, img_name))
                lr_img = cv2.imread(os.path.join(lr_path, img_name.replace(".png", "x4.png")))

                # SRCNN Step: Pre-upscale LR to match HR size [cite: 27, 104]
                lr_upscaled = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), 
                                       interpolation=cv2.INTER_CUBIC)

                current_img_patches_lr = []
                current_img_patches_hr = []

                # Extract Patches for this specific image
                for y in range(0, hr_img.shape[0] - self.config.patch_size + 1, self.config.stride):
                    for x in range(0, hr_img.shape[1] - self.config.patch_size + 1, self.config.stride):
                        hr_patch = hr_img[y:y+self.config.patch_size, x:x+self.config.patch_size]
                        lr_patch = lr_upscaled[y:y+self.config.patch_size, x:x+self.config.patch_size]
                        
                        current_img_patches_hr.append(hr_patch)
                        current_img_patches_lr.append(lr_patch)

                # 2. Append patches from this image to the HDF5 file
                if current_img_patches_hr:
                    num_new_patches = len(current_img_patches_hr)
                    
                    # Resize datasets to accommodate new data
                    lr_ds.resize(total_patches + num_new_patches, axis=0)
                    hr_ds.resize(total_patches + num_new_patches, axis=0)

                    # Write new patches to the newly created space
                    lr_ds[total_patches:] = np.array(current_img_patches_lr)
                    hr_ds[total_patches:] = np.array(current_img_patches_hr)

                    total_patches += num_new_patches

                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(img_list)} images. Total patches: {total_patches}")

        logger.info(f"Final dataset saved. Total patches: {total_patches}")