import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        
        # Layer 1: Patch extraction and representation
        # Filters: 64, Kernel: 9x9
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Non-linear mapping
        # Filters: 32, Kernel: 5x5
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Reconstruction
        # Filters: num_channels (3), Kernel: 5x5
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x