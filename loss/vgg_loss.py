import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # VGG expects normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target):
        # input and target are expected to be in [-1, 1]
        input = (input + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.loss(vgg_input, vgg_target)
