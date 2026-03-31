# Super-Resolution Comparison Project Details

This document outlines the technical implementation, architecture, and results of the Super-Resolution (SR) comparison application.

## 1. Project Objective
The goal is to provide a real-time, interactive interface to compare multiple Super-Resolution techniques against a ground truth (High Resolution) reference. The system evaluates performance using both quantitative metrics (PSNR, SSIM, Latency) and qualitative visual comparison.

## 2. Technical Stack
- **Framework**: Streamlit (Python-based Web UI)
- **Deep Learning**: PyTorch
- **Image Processing**: Pillow (PIL), NumPy, Scikit-Image
- **Visual Comparison**: `streamlit-image-comparison` (React-based slider)

---

## 3. Model Architectures

### A. SRCNN (Super-Resolution Convolutional Neural Network)
- **Type**: Shallow CNN (3 layers).
- **Working**:
  1. **Patch Extraction**: Extracts overlapping patches from the bicubic-upsampled input (9x9 kernel, 64 filters).
  2. **Non-linear Mapping**: Maps the high-dimensional vectors to another set of feature maps (5x5 kernel, 32 filters).
  3. **Reconstruction**: Aggregates the patches to produce the final SR image (5x5 kernel, 3 filters).
- **Optimization**: Mean Squared Error (MSE) loss. Focuses on pixel-wise accuracy (high PSNR).

### B. Real-ESRGAN (Enhanced Super-Resolution GAN)
- **Architecture**: `RRDBNet` (Residual-in-Residual Dense Block Network).
- **Complexity**: 23 RRDB blocks.
- **Key Features**:
  - **Dense Connections**: Each Residual Dense Block (RDB) contains 5 densely connected layers for better feature reuse.
  - **Upscaling**: Performs 4x upscaling internally using two stages of nearest-neighbor interpolation followed by refined convolutions.
  - **Perceptual Focus**: Unlike SRCNN, this is trained using a combination of Adversarial Loss (GAN) and Perceptual Loss (VGG feature matching). It produces significantly sharper edges and realistic textures.

---

## 4. Application Pipeline

1. **User Input**: User uploads a High-Resolution (HR) image.
2. **LR Generation**: The system automatically downscales the HR image by a factor of 4 using **Bicubic Interpolation** to create a Low-Resolution (LR) input.
3. **Inference**:
   - **Bicubic**: Upscales the LR image back to HR size (Base Baseline).
   - **SRCNN**: Upscales the LR image (Bicubic) and passes it through the SRCNN refiner.
   - **Real-ESRGAN**: Passes the LR image directly into the RRDBNet for 4x upscaling.
4. **Metric Calculation**:
   - **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level similarity (higher is better).
   - **SSIM (Structural Similarity Index)**: Measures structural and textural similarity (closer to 1.0 is better).
   - **Latency**: Measures the forward-pass execution time in milliseconds.
5. **Visualization**: Displays side-by-side results and interactive "before/after" sliders.

---

## 5. Technical Challenges & Solutions

### Nested Checkpoint Loading
The provided `RealESRGAN_x4.pth` checkpoint used a nested structure (`runner` -> `generator`). We implemented a custom loader in `main.py` that extracts the `state_dict` dynamically and handles common EMA (Exponential Moving Average) weight keys like `params_ema`.

### Module Stubbing
The original `srgan.pth` checkpoint relied on local `utils.config` classes from the training environment. To load it successfully in the standalone Streamlit app, we used `sys.modules` to create dummy stubs of the missing configuration modules, allowing `torch.load` to proceed without `ModuleNotFoundError`.

### Dynamic Normalization
Different models were trained on different scales:
- **SRCNN**: Trained on `[0, 1]` range.
- **Real-ESRGAN**: Operates on `[0, 1]` range in this implementation (aligned with user preferences).

---

## 6. Results Summary

| Model | Visual Quality | PSNR Focus | SSIM Focus | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic** | Blurred edges, pixelated | Baseline | Baseline | < 1ms |
| **SRCNN** | Smoother than Bicubic, less noise | **Highest** | High | ~2-5ms |
| **Real-ESRGAN** | **Sharpest**, realistic textures | Variable | **Highest Perceptual** | ~30-50ms |

> [!NOTE]
> The **Perceptual Tradeoff**: Real-ESRGAN may sometimes have a lower PSNR than SRCNN because it introduces high-frequency details that aren't present in the original image but look better to the human eye.
