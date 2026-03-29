import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Applied to each channel independently (RGB = 3 channels)
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # pred and target are (B, C, H, W)
        pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        # Edge maps (magnitude)
        pred_mag = torch.sqrt(pred_x**2 + pred_y**2 + 1e-6)
        target_mag = torch.sqrt(target_x**2 + target_y**2 + 1e-6)

        return self.l1_loss(pred_mag, target_mag)