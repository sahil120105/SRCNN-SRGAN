import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.srgan import Generator, Discriminator
from loss.vgg_loss import VGGLoss
from loss.edge_loss import EdgeLoss
from entity import SRGANTrainingConfig
from custom_logger import logger
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_edge_fidelity
from utils.data_loader import HDF5Dataset
import torchvision.utils as vutils

class SRGANTraining:
    def __init__(self, config: SRGANTrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        log_dir = self.config.root_dir / "logs_srgan"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        self.scaler_g = torch.amp.GradScaler('cuda')
        self.scaler_d = torch.amp.GradScaler('cuda')

    def train(self):
        train_loader = self._get_dataloader(self.config.train_data_path)
        valid_loader = self._get_dataloader(self.config.valid_data_path)

        netG = Generator().to(self.device)
        netD = Discriminator().to(self.device)
        
        vgg_loss = VGGLoss(self.device)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        edge_loss = EdgeLoss(self.device).to(self.device)
        
        optimizer_g = optim.Adam(netG.parameters(), lr=self.config.learning_rate_g)
        optimizer_d = optim.Adam(netD.parameters(), lr=self.config.learning_rate_d)
        
        best_psnr = 0.0
        global_step = 0

        # Resume logic
        if os.path.exists(self.config.model_path_g):
            logger.info(f"Loading existing Generator weights from {self.config.model_path_g}")
            netG.load_state_dict(torch.load(self.config.model_path_g, map_location=self.device, weights_only=True))
        if os.path.exists(self.config.model_path_d):
            logger.info(f"Loading existing Discriminator weights from {self.config.model_path_d}")
            netD.load_state_dict(torch.load(self.config.model_path_d, map_location=self.device, weights_only=True))
            
            
        # --- Pre-Train Generator logic ---
        if not os.path.exists(self.config.model_path_g) and self.config.pretrain_epochs > 0:
            logger.info(f"Starting Generator Pre-training ({self.config.pretrain_epochs} epochs) with MSE Loss only on {self.device}")
            for pretrain_epoch in range(self.config.pretrain_epochs):
                netG.train()
                epoch_loss_g = 0.0
                progress_bar = tqdm(train_loader, desc=f"Pre-Train Epoch {pretrain_epoch+1}/{self.config.pretrain_epochs}")
                
                for batch_idx, (lr, hr) in enumerate(progress_bar):
                    lr, hr = lr.to(self.device), hr.to(self.device)
                    optimizer_g.zero_grad()
                    
                    with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.autocast("cpu", enabled=False):
                        fake_hr = netG(lr)
                        g_loss = mse_loss(fake_hr, hr)
                        
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(optimizer_g)
                    self.scaler_g.update()
                    
                    epoch_loss_g += g_loss.item()
                    global_step += 1
                    
                    if global_step % self.config.log_step == 0:
                        avg_g_loss = epoch_loss_g / (batch_idx + 1)
                        self.writer.add_scalar("Loss/PreTrain_G_Step", avg_g_loss, global_step)
                        progress_bar.set_postfix({"G_MSE": f"{avg_g_loss:.4f}"})
               
                # Validation
                val_results = self._validate(netG, valid_loader, mse_loss, epoch=None) # Pass None to skip image logging or overlap
                avg_val_mse, avg_psnr, avg_ssim, avg_lpips, avg_gms = val_results
                
                self.writer.add_scalar("Metrics/PreTrain_PSNR_dB", avg_psnr, pretrain_epoch)
                logger.info(f"Pre-Train Epoch {pretrain_epoch+1}: PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f}")
                
            # Save pre-trained weights
            torch.save(netG.state_dict(), self.config.model_path_g)
            logger.info(f"Saved Pre-trained Generator at {self.config.model_path_g}")
            global_step = 0 # Reset for GAN train

        logger.info(f"Starting Main SRGAN (GAN) Training on {self.device}")

        for epoch in range(self.config.epochs):
            netG.train()
            netD.train()
            
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            
            for batch_idx, (lr, hr) in enumerate(progress_bar):
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                # --- Train Discriminator ---
                optimizer_d.zero_grad()
                with torch.amp.autocast('cuda'):
                    fake_hr = netG(lr)
                    
                    real_pred = netD(hr)
                    fake_pred = netD(fake_hr.detach())
                    
                    real_labels = torch.ones_like(real_pred)
                    fake_labels = torch.zeros_like(fake_pred)
                    
                    d_loss_real = bce_loss(real_pred, real_labels)
                    d_loss_fake = bce_loss(fake_pred, fake_labels)
                    d_loss = d_loss_real + d_loss_fake
                
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.step(optimizer_d)
                self.scaler_d.update()
                
                # --- Train Generator ---
                optimizer_g.zero_grad()
                with torch.amp.autocast('cuda'):
                    fake_hr = netG(lr) # Recompute since fake_hr was detached above, actually we didn't detach fake_hr, we detached fake_hr when passing to netD so fake_hr still has grad history from netG, but doing it again is safer or we can just not detach. But let's follow standard PyTorch tutorial which does it this way or just pass fake_hr directly. Oh wait, my code detach fake_hr: `fake_pred = netD(fake_hr.detach())`. So fake_hr is still attached. Wait, we can reuse it! Since we zero_grad on G, we just pass fake_hr to netD.
                    fake_pred = netD(fake_hr)
                    
                    real_labels = torch.ones_like(fake_pred)
                    g_adv_loss = bce_loss(fake_pred, real_labels)
                    
                    g_content_loss = vgg_loss(fake_hr, hr)
                    g_edge_loss = edge_loss(fake_hr, hr)
                    g_mse_loss = mse_loss(fake_hr, hr) # Kept for base convergence
                    
                    # Composite Generator Loss = Content (VGG + MSE) + Adversarial + Sobel Edge
                    g_loss = g_mse_loss + 0.006 * g_content_loss + 1e-3 * g_adv_loss + 0.1 * g_edge_loss
                
                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.step(optimizer_g)
                self.scaler_g.update()
                
                epoch_loss_d += d_loss.item()
                epoch_loss_g += g_loss.item()
                global_step += 1
                
                if global_step % self.config.log_step == 0:
                    avg_d_loss = epoch_loss_d / (batch_idx + 1)
                    avg_g_loss = epoch_loss_g / (batch_idx + 1)
                    self.writer.add_scalar("Loss/D_Step", avg_d_loss, global_step)
                    self.writer.add_scalar("Loss/G_Step", avg_g_loss, global_step)
                    progress_bar.set_postfix({"D_Loss": f"{avg_d_loss:.4f}", "G_Loss": f"{avg_g_loss:.4f}"})
            
            # Validation at end of epoch
            val_results = self._validate(netG, valid_loader, mse_loss, epoch)
            avg_val_mse, avg_psnr, avg_ssim, avg_lpips, avg_gms = val_results
            
            self.writer.add_scalar("Loss/Validation_MSE", avg_val_mse, epoch)
            self.writer.add_scalar("Metrics/PSNR_dB", avg_psnr, epoch)
            self.writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
            self.writer.add_scalar("Metrics/LPIPS", avg_lpips, epoch)
            self.writer.add_scalar("Metrics/GMS", avg_gms, epoch)
            
            logger.info(f"Epoch {epoch+1}: Val MSE={avg_val_mse:.6f} | PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f} | LPIPS={avg_lpips:.4f} | GMS={avg_gms:.4f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(netG.state_dict(), self.config.model_path_g)
                torch.save(netD.state_dict(), self.config.model_path_d)
                logger.info(f"New Best Model saved at {best_psnr:.2f}dB PSNR")
        
        self.writer.close()
        logger.info("SRGAN Training finalized.")

    def _validate(self, netG, loader, criterion, epoch=None):
        netG.eval()
        val_loss = 0.0
        psnr_values, ssim_values, lpips_values, gms_values = [], [], [], []
        
        with torch.no_grad():
            for batch_idx, (lr, hr) in enumerate(tqdm(loader, desc="Validation")):
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.autocast("cpu", enabled=False):
                    fake_hr = netG(lr)
                    loss = criterion(fake_hr, hr)
                
                val_loss += loss.item()
                
                # Inference Visibility (Save visualization array per epoch)
                if batch_idx == 0 and epoch is not None:
                    # Upscale LR explicitly to compare spatially with HR in TensorBoard
                    lr_up = torch.nn.functional.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
                    vis_lr = (lr_up[:4] + 1) / 2.0
                    vis_hr = (hr[:4] + 1) / 2.0
                    vis_fake = (fake_hr[:4] + 1) / 2.0
                    
                    # Stack them properly LR | Fake | Real
                    grid = vutils.make_grid(torch.cat([vis_lr, vis_fake, vis_hr], dim=0), nrow=4, normalize=False)
                    self.writer.add_image("Inference_Visibility/LR_FakeHR_RealHR", grid, epoch)
                
                # Convert for skimage metrics
                out_np = (fake_hr.cpu().numpy() + 1.0) / 2.0 * 255.0
                hr_np = (hr.cpu().numpy() + 1.0) / 2.0 * 255.0
                out_np_uint = np.clip(out_np, 0, 255).astype(np.uint8)
                hr_np_uint = np.clip(hr_np, 0, 255).astype(np.uint8)
                
                for i in range(out_np.shape[0]):
                    img_out = np.transpose(out_np_uint[i], (1, 2, 0))
                    img_hr = np.transpose(hr_np_uint[i], (1, 2, 0))
                    
                    psnr_values.append(calculate_psnr(img_out, img_hr))
                    ssim_values.append(calculate_ssim(img_out, img_hr))
                    # LPIPS expects tensor in [-1, 1], shape (1, C, H, W)
                    lpips_values.append(calculate_lpips(fake_hr[i:i+1], hr[i:i+1]))
                    
                    # Convert to grayscale for gradient fidelity
                    import cv2
                    img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)
                    img_hr_gray = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
                    gms_values.append(calculate_edge_fidelity(img_out_gray, img_hr_gray))

        return (val_loss / len(loader), np.mean(psnr_values), np.mean(ssim_values), 
                np.mean(lpips_values), np.mean(gms_values))

    def _get_dataloader(self, path):
        dataset = HDF5Dataset(path, normalization=self.config.normalization)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False
        )
