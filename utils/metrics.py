import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import lpips

# Initialize LPIPS model (VGG-based) once to save memory
loss_fn_alex = lpips.LPIPS(net='alex') 

def calculate_psnr(img1, img2):
    """
    Calculates Peak Signal-to-Noise Ratio.
    Higher is better for signal fidelity.
    """
    # Expects images in range [0, 255] or [0, 1]
    return psnr_func(img1, img2, data_range=img2.max() - img2.min())

def calculate_ssim(img1, img2):
    """
    Calculates Structural Similarity Index.
    Measures luminance, contrast, and structure.
    """
    return ssim_func(img1, img2, channel_axis=2, data_range=img2.max() - img2.min())

def calculate_lpips(img1_tensor, img2_tensor):
    """
    Calculates Learned Perceptual Image Patch Similarity.
    Lower is better (closer to human perception).
    Expects Tensors in range [-1, 1].
    """
    with torch.no_grad():
        # LPIPS expects batch dimension, usually (1, C, H, W) or (B, C, H, W)
        if len(img1_tensor.shape) == 3:
            img1_tensor = img1_tensor.unsqueeze(0)
            img2_tensor = img2_tensor.unsqueeze(0)
        dist = loss_fn_alex(img1_tensor.cpu(), img2_tensor.cpu())
    return dist.mean().item()

def calculate_edge_fidelity(img1, img2):
    """
    Novelty Metric: Measures how well sharp edges were preserved 
    using Gradient Magnitude Similarity (GMS).
    """
    import cv2
    # Compute gradients using Sobel
    grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
    
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    # Simple MSE between gradient magnitudes
    return np.mean((mag1 - mag2)**2)