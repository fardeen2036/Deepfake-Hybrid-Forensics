import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.cm as cm

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Forensics Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d0f14; color: #e0e0e0; }
  header[data-testid="stHeader"] { background: transparent; }

  .hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1318 100%);
    border: 1px solid #2a3040; border-radius: 16px;
    padding: 36px 40px 28px; margin-bottom: 28px;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #63b3ff, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
  }
  .hero-sub { color: #8892a4; font-size: 1rem; margin: 0; }

  .stat-row { display: flex; gap: 14px; margin-bottom: 28px; flex-wrap: wrap; }
  .stat-card {
    flex: 1; min-width: 120px;
    background: #141820; border: 1px solid #1e2533;
    border-radius: 12px; padding: 18px 20px;
  }
  .stat-label { font-size: 0.72rem; color: #5a6478; text-transform: uppercase; letter-spacing: .08em; }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: #e0e4ef; margin-top: 4px; }

  .verdict-real {
    background: linear-gradient(135deg, #0d2318, #0f2e1a);
    border: 1px solid #1a5c30; border-left: 4px solid #2ecc71;
    border-radius: 12px; padding: 22px 28px; margin: 24px 0;
  }
  .verdict-fake {
    background: linear-gradient(135deg, #2a0d0d, #1f0a0a);
    border: 1px solid #5c1a1a; border-left: 4px solid #e74c3c;
    border-radius: 12px; padding: 22px 28px; margin: 24px 0;
  }
  .verdict-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .1em; color: #8892a4; }
  .verdict-text-real { font-size: 2.2rem; font-weight: 800; color: #2ecc71; }
  .verdict-text-fake { font-size: 2.2rem; font-weight: 800; color: #e74c3c; }
  .verdict-conf { font-size: 0.95rem; color: #8892a4; margin-top: 4px; }

  .risk-bar-bg { background: #1e2533; border-radius: 6px; height: 10px; width: 100%; margin: 12px 0 4px; overflow: hidden; }
  .risk-bar-real { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #1a8a4a, #2ecc71); }
  .risk-bar-fake { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #8a1a1a, #e74c3c); }

  .panel { background: #141820; border: 1px solid #1e2533; border-radius: 14px; padding: 18px; margin-bottom: 16px; }
  .panel-title { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .1em; color: #5a6478; margin-bottom: 10px; }
  .panel-note { font-size: 0.7rem; color: #5a6478; margin-top: 8px; }

  .upload-zone {
    background: #141820; border: 2px dashed #2a3450;
    border-radius: 14px; padding: 48px; text-align: center;
  }

  .stFileUploader > div { background: #141820 !important; border-color: #2a3450 !important; border-radius: 12px !important; }
  div[data-testid="stSidebar"] { background: #0f1318 !important; border-right: 1px solid #1e2533; }
  .stTextInput input { background: #1a1f2e !important; border-color: #2a3450 !important; color: #e0e0e0 !important; border-radius: 8px !important; }
  img { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── MODEL ARCHITECTURE ────────────────────────────────────────
def get_hybrid_model():
    model = models.efficientnet_b0(weights=None)
    orig = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        4, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 1)
    )
    return model


# ── FFT CHANNEL ───────────────────────────────────────────────
def get_fft_channel(img_tensor):
    gray  = 0.299*img_tensor[0] + 0.587*img_tensor[1] + 0.114*img_tensor[2]
    f     = torch.fft.fft2(gray)
    fshift = torch.fft.fftshift(f)
    mag   = torch.log(torch.abs(fshift) + 1)
    mag   = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.unsqueeze(0)


# ── GRAD-CAM ──────────────────────────────────────────────────
def get_gradcam(model, inp_4ch):
    activations, gradients = {}, {}

    def fwd(m, i, o): activations["f"] = o.detach()
    def bwd(m, gi, go): gradients["f"] = go[0].detach()

    target = model.features[-1]
    h1 = target.register_forward_hook(fwd)
    h2 = target.register_full_backward_hook(bwd)

    x = inp_4ch.clone().requires_grad_(True)
    model.zero_grad()
    torch.sigmoid(model(x)).squeeze().backward()
    h1.remove(); h2.remove()

    weights = gradients["f"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["f"]).sum(dim=1)).squeeze()
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_cam(rgb_np, cam):
    heatmap = cm.jet(cam)[:, :, :3]
    return ((0.55 * rgb_np + 0.45 * heatmap).clip(0, 1) * 255).astype(np.uint8)


# ── TRANSFORMS ───────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(180),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Settings")
model_path = st.sidebar.text_input("Model Weights Path", "deepfake_hybrid_effnet_v1.pth")
st.sidebar.markdown("---")
st.sidebar.markdown("**Architecture** · EfficientNet-B0")
st.sidebar.markdown("**Channels** · RGB + FFT (4-ch)")
st.sidebar.markdown("**Dataset** · FaceForensics++ C23")
st.sidebar.markdown("**Val AUC** · 0.9270")

@st.cache_resource
def load_model(path):
    try:
        m = get_hybrid_model()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m
    except Exception as e:
        st.sidebar.error(f"❌ {e}")
        return None

model = load_model(model_path)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔬 DeepFake Forensics Lab</div>
  <p class="hero-sub">Hybrid RGB + Frequency (FFT) fusion &nbsp;·&nbsp; EfficientNet-B0 &nbsp;·&nbsp; AUC 0.9270 &nbsp;·&nbsp; FaceForensics++ C23</p>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS ────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Architecture</div>
    <div class="stat-value" style="font-size:1.05rem;color:#63b3ff;">EfficientNet-B0</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Input channels</div>
    <div class="stat-value">4</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Val AUC-ROC</div>
    <div class="stat-value" style="color:#2ecc71;">0.9270</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Fusion method</div>
    <div class="stat-value" style="font-size:1rem;color:#a78bfa;">RGB + FFT</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Explainability</div>
    <div class="stat-value" style="font-size:1rem;color:#f6ad55;">Grad-CAM</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a face image (JPG / PNG)", type=["jpg", "jpeg", "png"])

# ── ANALYSIS ──────────────────────────────────────────────────
if uploaded_file and model:
    img   = Image.open(uploaded_file).convert("RGB")
    img_t = val_tf(img)
    fft_t = get_fft_channel(img_t)
    inp   = torch.cat([img_t, fft_t], dim=0).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(inp)).item()

    verdict    = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    risk_pct   = int(prob * 100)

    # ── VERDICT ───────────────────────────────────────────────
    if verdict == "FAKE":
        st.markdown(f"""
        <div class="verdict-fake">
          <div class="verdict-label">forensic verdict</div>
          <div class="verdict-text-fake">⚠️ FAKE DETECTED</div>
          <div class="verdict-conf">Confidence: {confidence:.2%} &nbsp;·&nbsp; Fake probability: {prob:.4f}</div>
          <div class="risk-bar-bg"><div class="risk-bar-fake" style="width:{risk_pct}%;"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#5a6478;">
            <span>REAL ←</span><span>→ FAKE</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-real">
          <div class="verdict-label">forensic verdict</div>
          <div class="verdict-text-real">✅ AUTHENTIC</div>
          <div class="verdict-conf">Confidence: {confidence:.2%} &nbsp;·&nbsp; Fake probability: {prob:.4f}</div>
          <div class="risk-bar-bg"><div class="risk-bar-real" style="width:{risk_pct}%;"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#5a6478;">
            <span>REAL ←</span><span>→ FAKE</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── THREE PANELS ──────────────────────────────────────────
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    rgb_np = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="panel"><div class="panel-title">📷 Input image</div>', unsafe_allow_html=True)
        st.image(rgb_np, use_container_width=True)
        st.markdown('<div class="panel-note">Preprocessed · 224×224 · CenterCrop</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title">📡 Frequency channel (FFT)</div>', unsafe_allow_html=True)
        fft_np     = fft_t.squeeze().numpy()
        fft_colored = cm.magma(fft_np)[:, :, :3].astype(np.float32)
        st.image(fft_colored, use_container_width=True)
        st.markdown('<div class="panel-note">Real → clean cross pattern · GAN → grid periodicity</div></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel"><div class="panel-title">🧠 Grad-CAM — model focus</div>', unsafe_allow_html=True)
        with st.spinner("Computing forensic heatmap..."):
            cam     = get_gradcam(model, inp)
            overlay = overlay_cam(rgb_np, cam)
        st.image(overlay, use_container_width=True)
        st.markdown('<div class="panel-note">Red = high attention · Real → eyes/nose · Fake → blending edges</div></div>', unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧩 Forensic signal breakdown")
    m1, m2, m3, m4 = st.columns(4)
    fft_energy = float(fft_t.mean())
    fft_std    = float(fft_t.std())
    cam_peak   = float(cam.max())
    cam_area   = float((cam > 0.5).mean() * 100)

    with m1: st.metric("FFT Mean Energy", f"{fft_energy:.4f}", help="Higher in GAN faces — periodic artifacts")
    with m2: st.metric("FFT Std Dev",     f"{fft_std:.4f}",    help="Low variance = uniform GAN frequency")
    with m3: st.metric("Grad-CAM Peak",   f"{cam_peak:.4f}",   help="Intensity of most suspicious region")
    with m4: st.metric("Focus Area",      f"{cam_area:.1f}%",  help="% of image flagged as suspicious")

elif not model:
    st.error("⚠️ Model weights not found. Check the path in the sidebar.")
else:
    st.markdown("""
    <div class="upload-zone">
      <div style="font-size:2.8rem;margin-bottom:14px;">🖼️</div>
      <div style="color:#8892a4;font-size:1.05rem;">Drop a face image to begin forensic analysis</div>
      <div style="color:#5a6478;font-size:0.8rem;margin-top:8px;">JPG · PNG · Real photos and AI-generated faces</div>
    </div>
    """, unsafe_allow_html=True)
