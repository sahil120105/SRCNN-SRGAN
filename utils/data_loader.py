import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class HDF5Dataset(Dataset):
    """
    Industry-grade HDF5 loader that handles dynamic normalization
    based on the model's architectural requirements.
    """
    def __init__(self, h5_file, normalization='zero_to_one'):
        super(HDF5Dataset, self).__init__()
        self.h5_file = h5_file
        self.norm = normalization

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            # uint8 -> float32 conversion
            lr = np.array(f['lr'][index]).astype(np.float32)
            hr = np.array(f['hr'][index]).astype(np.float32)
            
            # SRCNN uses [0, 1] for MSE Loss consistency
            if self.norm == "zero_to_one":
                lr /= 255.0
                hr /= 255.0
            # SRGAN Generator uses Tanh, requiring [-1, 1]
            else:
                lr = (lr / 127.5) - 1.0
                hr = (hr / 127.5) - 1.0
                
            # Change shape from (H, W, C) to PyTorch (C, H, W)
            lr_tensor = torch.from_numpy(lr).permute(2, 0, 1)
            hr_tensor = torch.from_numpy(hr).permute(2, 0, 1)
            
            return lr_tensor, hr_tensor

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])