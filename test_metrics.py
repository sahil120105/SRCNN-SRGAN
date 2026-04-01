import os
import sys
import time
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from custom_logger import logger 

# ── Model definitions (keys match checkpoints exactly) ────────────────────────
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.patch_extraction = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.nonlinear_map    = nn.Conv2d(64,           32, kernel_size=5, padding=2)
        self.recon            = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu             = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.patch_extraction(x))
        x = self.relu(self.nonlinear_map(x))
        return self.recon(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class SRGANGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        body_fea = self.conv_body(self.body(fea))
        fea = fea + body_fea
        
        # Upsampling x4 (Two 2x stages)
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# ── Constants ─────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(__file__)
SRCNN_PATH = os.path.join(script_dir, "artifacts", "model_training", "srcnn2.pt")
SRGAN_PATH = os.path.join(script_dir, "artifacts", "model_training", "RealESRGAN_x4.pth")
SCALE = 4

# ── Inference helpers ─────────────────────────────────────────────────────────
def pil_to_np(img):
    return np.array(img.convert("RGB"), dtype=np.uint8)

def metrics(pred_np, ref_np):
    p = calc_psnr(ref_np, pred_np, data_range=255)
    s = calc_ssim(ref_np, pred_np, data_range=255, channel_axis=2)
    return p, s

def run_bicubic(lr: Image.Image, size: tuple):
    t0  = time.perf_counter()
    out = lr.resize(size, Image.BICUBIC)
    ms  = (time.perf_counter() - t0) * 1000
    return out, ms

def run_srcnn(lr: Image.Image, size: tuple, model, device):
    up  = lr.resize(size, Image.BICUBIC)
    arr = pil_to_np(up).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    t0  = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    if device.type == "cuda": torch.cuda.synchronize()
    ms  = (time.perf_counter() - t0) * 1000
    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = (np.clip(out_np, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out_np), ms

def run_srgan(lr: Image.Image, hr_size: tuple, model, device):
    arr = pil_to_np(lr).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    if device.type == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    
    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = (np.clip(out_np, 0, 1) * 255).astype(np.uint8)
    out_img = Image.fromarray(out_np)

    if out_img.size != hr_size:
        out_img = out_img.resize(hr_size, Image.BICUBIC)
    return out_img, ms


def load_images(img_dir, num_images=20):
    images = []
    # Try multiple standard extensions
    for ext in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
        images.extend(glob.glob(os.path.join(img_dir, f"*.{ext}")))
    
    if not images:
        return []
    
    # Shuffle and pick the requested number
    random.shuffle(images)
    selected = images[:num_images]
    logger.info(f"Loaded {len(selected)} random testing images from {img_dir}")
    return selected


def main(image_dir=None, num_images=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize models
    srcnn_model = SRCNN().to(device)
    srcnn_state = torch.load(SRCNN_PATH, map_location=device, weights_only=False)
    srcnn_model.load_state_dict(srcnn_state)
    srcnn_model.eval()

    srgan_model = SRGANGenerator(nf=64, nb=23, gc=32).to(device)
    srgan_data = torch.load(SRGAN_PATH, map_location=device, weights_only=False)
    state_dict = srgan_data.get('params_ema', srgan_data.get('params', srgan_data))
    srgan_model.load_state_dict(state_dict, strict=True)
    srgan_model.eval()

    logger.info("Models loaded successfully.")

    # Find the image directory if None provided
    if not image_dir:
        potential_dirs = [
            os.path.join(script_dir, "data", "raw", "Set5", "HR"),
            os.path.join(script_dir, "data", "raw", "DIV2K", "DIV2K_train_HR"),
            os.path.join(script_dir, "data", "raw", "DIV2K", "DIV2K_valid_HR")
        ]
        for d in potential_dirs:
            if os.path.exists(d) and len(os.listdir(d)) > 0:
                image_dir = d
                break
    
    if not image_dir or not os.path.exists(image_dir):
        logger.error("No valid image directory found. Please specify an image directory.")
        return

    images = load_images(image_dir, num_images)
    if not images:
        logger.error(f"No images found in {image_dir}.")
        return

    total_metrics = {
        "bicubic": {"psnr": 0, "ssim": 0, "latency": 0},
        "srcnn": {"psnr": 0, "ssim": 0, "latency": 0},
        "srgan": {"psnr": 0, "ssim": 0, "latency": 0}
    }

    count = 0
    logger.info(f"Evaluating metrics over {len(images)} images...")

    for img_path in images:
        try:
            hr_orig = Image.open(img_path).convert("RGB")
            W, H = hr_orig.size
            
            # Filter extremely large images to avoid OOM or slow evaluation 
            if W * H > 2000 * 2000:
                # Optionally downsize very large images
                pass

            W_crop = (W // SCALE) * SCALE
            H_crop = (H // SCALE) * SCALE
            hr = hr_orig.crop((0, 0, W_crop, H_crop))
            lr = hr.resize((W_crop // SCALE, H_crop // SCALE), Image.BICUBIC)
            size = (W_crop, H_crop)
            hr_np = pil_to_np(hr)

            # Bicubic
            bic_img, bic_ms = run_bicubic(lr, size)
            bic_psnr, bic_ssim = metrics(pil_to_np(bic_img), hr_np)

            # SRCNN
            srcnn_img, srcnn_ms = run_srcnn(lr, size, srcnn_model, device)
            srcnn_psnr, srcnn_ssim = metrics(pil_to_np(srcnn_img), hr_np)

            # Real-ESRGAN
            srgan_img, srgan_ms = run_srgan(lr, size, srgan_model, device)
            srgan_psnr, srgan_ssim = metrics(pil_to_np(srgan_img), hr_np)

            total_metrics["bicubic"]["psnr"] += bic_psnr
            total_metrics["bicubic"]["ssim"] += bic_ssim
            total_metrics["bicubic"]["latency"] += bic_ms

            total_metrics["srcnn"]["psnr"] += srcnn_psnr
            total_metrics["srcnn"]["ssim"] += srcnn_ssim
            total_metrics["srcnn"]["latency"] += srcnn_ms

            total_metrics["srgan"]["psnr"] += srgan_psnr
            total_metrics["srgan"]["ssim"] += srgan_ssim
            total_metrics["srgan"]["latency"] += srgan_ms

            count += 1
            logger.info(f"Idx {count}: {os.path.basename(img_path)} evaluated.")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    if count == 0:
        logger.error("No images were successfully processed.")
        return

    logger.info("\n" + "="*50)
    logger.info(f"FINAL MEAN METRICS FOR {count} IMAGES:")
    logger.info("="*50)
    
    methods = ["bicubic", "srcnn", "srgan"]
    for method in methods:
        m = total_metrics[method]
        avg_psnr = m["psnr"] / count
        avg_ssim = m["ssim"] / count
        avg_lat = m["latency"] / count
        logger.info(f"--- {method.upper()} ---")
        logger.info(f"Mean PSNR:    {avg_psnr:.2f} dB")
        logger.info(f"Mean SSIM:    {avg_ssim:.4f}")
        logger.info(f"Mean Latency: {avg_lat:.1f} ms")
        logger.info("-" * 25)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Test AI Models")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory with HR images")
    parser.add_argument("--num", type=int, default=20, help="Number of random images to test")
    args = parser.parse_args()
    
    main(image_dir=args.img_dir, num_images=args.num)
