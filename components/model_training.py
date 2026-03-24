import torch
import h5py
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.srcnn import SRCNN
from entity import ModelTrainingConfig
from custom_logger import logger
from utils.metrics import calculate_psnr
from utils.common import EarlyStopping
from utils.data_loader import HDF5Dataset
from tqdm import tqdm
import os

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # TensorBoard setup for real-time visibility
        log_dir = self.config.root_dir / "logs"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Modern Mixed Precision Scaler (2026 PyTorch API)
        self.scaler = torch.amp.GradScaler('cuda') 

    def train(self):
        # 1. Initialize Loaders
        train_loader = self._get_dataloader(self.config.train_data_path)
        valid_loader = self._get_dataloader(self.config.valid_data_path)

        # 2. Setup Architecture & Optimization
        model = SRCNN().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Early Stopping & Best Metric Tracking
        early_stopping = EarlyStopping(patience=self.config.patience)
        best_psnr = 0.0
        global_step = 0

        # --- RESUME LOGIC ---
        if os.path.exists(self.config.model_path):
            logger.info(f"Loading existing weights from {self.config.model_path}...")
            model.load_state_dict(torch.load(self.config.model_path, weights_only=True))

        logger.info(f"Training {self.config.model_type.upper()} on {self.device} with Step Logging every {self.config.log_step} batches.")

        # 3. Training Loop
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            
            for batch_idx, (lr, hr) in enumerate(progress_bar):
                lr, hr = lr.to(self.device), hr.to(self.device)
                optimizer.zero_grad()

                # --- High-Speed Math (AMP) ---
                with torch.amp.autocast('cuda'):
                    outputs = model(lr)
                    loss = criterion(outputs, hr)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                global_step += 1

                # --- STEP-LEVEL LOGGING ---
                if global_step % self.config.log_step == 0:
                    avg_step_loss = epoch_loss / (batch_idx + 1)
                    self.writer.add_scalar("Loss/Train_Step", avg_step_loss, global_step)
                    progress_bar.set_postfix({"loss": f"{avg_step_loss:.5f}"})

            # 4. VALIDATION & QUALITY ASSURANCE (End of Epoch)
            avg_val_loss, avg_psnr = self._validate(model, valid_loader, criterion)
            
            self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            self.writer.add_scalar("Metrics/PSNR_dB", avg_psnr, epoch)

            logger.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.6f} | Val PSNR: {avg_psnr:.2f}dB")

            # --- BEST MODEL CHECKPOINTING ---
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), self.config.model_path)
                logger.info(f"New Best Model saved at {best_psnr:.2f}dB PSNR")

            # --- EARLY STOPPING ---
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                logger.warning(f"Early Stopping triggered at epoch {epoch+1}. Model converged.")
                break

        self.writer.close()
        logger.info("Training process finalized.")

    def _validate(self, model, loader, criterion):
        """
        Runs a full pass over the validation set to calculate MSE and PSNR.
        """
        model.eval()
        val_loss = 0.0
        psnr_values = []
        
        with torch.no_grad():
            for lr, hr in loader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(lr)
                    loss = criterion(outputs, hr)
                
                val_loss += loss.item()
                
                # Metric calculation: Convert back to [0, 255] range for PSNR
                out_np = outputs.cpu().numpy() * 255.0
                hr_np = hr.cpu().numpy() * 255.0
                psnr_values.append(calculate_psnr(out_np, hr_np))

        return val_loss / len(loader), np.mean(psnr_values)

    def _get_dataloader(self, path):
        """
        Creates a high-throughput dataloader using multiprocessing and memory pinning.
        """
        dataset = HDF5Dataset(path, normalization=self.config.normalization)
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=0,
            persistent_workers=False
        )

    def _train_srgan(self, loader):
        """
        Placeholder for Perceptual/Adversarial training logic.
        Will involve Generator, Discriminator, and VGG Perceptual Loss[cite: 35, 36, 106].
        """
        logger.info("SRGAN Training logic not yet implemented. Focus on SRCNN baseline first.")
        pass