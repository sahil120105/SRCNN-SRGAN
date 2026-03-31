"""
Super-Resolution Inference App
Upload a high-resolution image → 4x downscale → compare Bicubic / SRCNN / SRGAN
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import math
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Super Resolution Comparison",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { display: none; }

.block-container { padding: 2rem 3rem; max-width: 1400px; }

h1 { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-bottom: 0.15rem; }
.subtitle { color: #64748b; font-size: 0.875rem; margin-bottom: 2rem; }

.col-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

.metric-block {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    background: #f8fafc;
}
.metric-row-inner {
    display: flex;
    gap: 0.5rem;
}
.metric-item { flex: 1; text-align: center; }
.metric-name {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}
.metric-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0.1rem;
}
.metric-unit { font-size: 0.62rem; color: #94a3b8; }
.delta-good { color: #16a34a; font-size: 0.65rem; font-weight: 600; }
.delta-bad  { color: #dc2626; font-size: 0.65rem; font-weight: 600; }
.delta-neu  { color: #94a3b8; font-size: 0.65rem; }

hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.75rem 0; }

.slider-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model definitions (keys match checkpoints exactly) ────────────────────────

class SRCNN(nn.Module):
    """Matches srcnn2.pt: keys patch_extraction / nonlinear_map / recon"""
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
SRCNN_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_training", "srcnn2.pt")
SRGAN_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_training", "RealESRGAN_x4.pth")
SCALE      = 4


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_srcnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SRCNN().to(device)
    state  = torch.load(SRCNN_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_srgan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRGANGenerator(nf=64, nb=23, gc=32).to(device)
    try:
        data = torch.load(SRGAN_PATH, map_location=device, weights_only=False)
        # RealESRGAN usually saves state in 'params' or 'params_ema'
        state_dict = data.get('params_ema', data.get('params', data))
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        st.error(f"Error loading RealESRGAN: {e}")
    model.eval()
    return model, device


# ── Inference helpers ─────────────────────────────────────────────────────────
def pil_to_np(img):
    return np.array(img.convert("RGB"), dtype=np.uint8)


def run_bicubic(lr: Image.Image, size: tuple):
    t0  = time.perf_counter()
    out = lr.resize(size, Image.BICUBIC)
    ms  = (time.perf_counter() - t0) * 1000
    return out, ms


def run_srcnn(lr: Image.Image, size: tuple, model, device):
    """SRCNN: bicubic upsample first, then refine."""
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
    """
    RealESRGAN Generator (RRDBNet): LR [0, 1] -> 4x output.
    """
    arr = pil_to_np(lr).astype(np.float32) / 255.0     # [0, 1]
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


def metrics(pred_np, ref_np):
    p = calc_psnr(ref_np, pred_np, data_range=255)
    s = calc_ssim(ref_np, pred_np, data_range=255, channel_axis=2)
    return p, s


def to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def metric_card(psnr, ssim, latency, psnr_delta=None, ssim_delta=None):
    def fmt(d, dec):
        if d is None:
            return '<span class="delta-neu">baseline</span>'
        sign = "▲" if d >= 0 else "▼"
        cls  = "delta-good" if d >= 0 else "delta-bad"
        return f'<span class="{cls}">{sign} {abs(d):.{dec}f}</span>'

    return f"""
    <div class="metric-block">
        <div class="metric-row-inner">
            <div class="metric-item">
                <div class="metric-name">PSNR</div>
                <div class="metric-val">{psnr:.2f}<span class="metric-unit"> dB</span></div>
                {fmt(psnr_delta, 2)}
            </div>
            <div class="metric-item">
                <div class="metric-name">SSIM</div>
                <div class="metric-val">{ssim:.4f}</div>
                {fmt(ssim_delta, 4)}
            </div>
            <div class="metric-item">
                <div class="metric-name">Latency</div>
                <div class="metric-val">{latency:.1f}<span class="metric-unit"> ms</span></div>
                <span class="delta-neu">forward pass</span>
            </div>
        </div>
    </div>
    """


# ── Load models ───────────────────────────────────────────────────────────────
try:
    srcnn_model, srcnn_device = load_srcnn()
    srcnn_ok = True
except Exception as e:
    srcnn_ok  = False
    srcnn_err = str(e)

try:
    srgan_model, srgan_device = load_srgan()
    srgan_ok = True
except Exception as e:
    srgan_ok  = False
    srgan_err = str(e)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>Super-Resolution Comparison</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>"
    "Upload a high-resolution image — automatically downscaled 4× — "
    "then compared across <b>Bicubic</b>, <b>SRCNN</b>, and <b>SRGAN</b>."
    "</div>",
    unsafe_allow_html=True,
)

if not srcnn_ok:
    st.error(f"SRCNN failed to load: {srcnn_err}")
if not srgan_ok:
    st.error(f"SRGAN failed to load: {srgan_err}")
if not srcnn_ok or not srgan_ok:
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.875rem; margin-top:0.5rem;'>"
        "Drag and drop or browse an image to get started."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── Preprocess ────────────────────────────────────────────────────────────────
hr_orig = Image.open(uploaded).convert("RGB")
W, H    = hr_orig.size
W_crop  = (W // SCALE) * SCALE
H_crop  = (H // SCALE) * SCALE
hr      = hr_orig.crop((0, 0, W_crop, H_crop))
lr      = hr.resize((W_crop // SCALE, H_crop // SCALE), Image.BICUBIC)
size    = (W_crop, H_crop)


# ── Run all three ─────────────────────────────────────────────────────────────
with st.spinner("Running inference…"):
    bic_img,   bic_ms    = run_bicubic(lr, size)
    srcnn_img, srcnn_ms  = run_srcnn(lr, size, srcnn_model, srcnn_device)
    srgan_img, srgan_ms  = run_srgan(lr, size, srgan_model, srgan_device)

hr_np    = pil_to_np(hr)
bic_np   = pil_to_np(bic_img)
srcnn_np = pil_to_np(srcnn_img)
srgan_np = pil_to_np(srgan_img)

bic_psnr,   bic_ssim   = metrics(bic_np,   hr_np)
srcnn_psnr, srcnn_ssim = metrics(srcnn_np, hr_np)
srgan_psnr, srgan_ssim = metrics(srgan_np, hr_np)


# ── Three-column comparison ───────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

col_bic, col_srcnn, col_srgan = st.columns(3)

with col_bic:
    st.markdown("<div class='col-label'>Bicubic Interpolation</div>", unsafe_allow_html=True)
    st.image(bic_img, use_container_width=True)
    st.markdown(metric_card(bic_psnr, bic_ssim, bic_ms), unsafe_allow_html=True)

with col_srcnn:
    st.markdown("<div class='col-label'>SRCNN</div>", unsafe_allow_html=True)
    st.image(srcnn_img, use_container_width=True)
    st.markdown(
        metric_card(srcnn_psnr, srcnn_ssim, srcnn_ms,
                    srcnn_psnr - bic_psnr, srcnn_ssim - bic_ssim),
        unsafe_allow_html=True,
    )

with col_srgan:
    st.markdown("<div class='col-label'>SRGAN Generator</div>", unsafe_allow_html=True)
    st.image(srgan_img, use_container_width=True)
    st.markdown(
        metric_card(srgan_psnr, srgan_ssim, srgan_ms,
                    srgan_psnr - bic_psnr, srgan_ssim - bic_ssim),
        unsafe_allow_html=True,
    )


# ── Slider comparisons ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

try:
    from streamlit_image_comparison import image_comparison

    st.markdown("<div class='slider-label'>Bicubic vs SRCNN</div>", unsafe_allow_html=True)
    image_comparison(
        img1=bic_img, img2=srcnn_img,
        label1="Bicubic", label2="SRCNN",
        starting_position=50, show_labels=True,
        make_responsive=True, in_memory=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='slider-label'>Bicubic vs SRGAN</div>", unsafe_allow_html=True)
    image_comparison(
        img1=bic_img, img2=srgan_img,
        label1="Bicubic", label2="SRGAN",
        starting_position=50, show_labels=True,
        make_responsive=True, in_memory=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='slider-label'>SRCNN vs SRGAN</div>", unsafe_allow_html=True)
    image_comparison(
        img1=srcnn_img, img2=srgan_img,
        label1="SRCNN", label2="SRGAN",
        starting_position=50, show_labels=True,
        make_responsive=True, in_memory=True,
    )

except ImportError:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(bic_img,   use_container_width=True, caption="Bicubic")
    with c2:
        st.image(srcnn_img, use_container_width=True, caption="SRCNN")
    with c3:
        st.image(srgan_img, use_container_width=True, caption="SRGAN")


# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
d1, d2, d3, _ = st.columns([1, 1, 1, 1])
with d1:
    st.download_button("⬇ Bicubic SR",  to_bytes(bic_img),   "bicubic_sr.png",  "image/png")
with d2:
    st.download_button("⬇ SRCNN SR",    to_bytes(srcnn_img), "srcnn_sr.png",    "image/png")
with d3:
    st.download_button("⬇ SRGAN SR",    to_bytes(srgan_img), "srgan_sr.png",    "image/png")
