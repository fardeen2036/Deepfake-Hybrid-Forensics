import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.cm as cm

st.set_page_config(page_title="SPECTRE · Deepfake Lab", page_icon="👁", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
  background: #020408;
  font-family: 'Rajdhani', sans-serif;
  overflow-x: hidden;
}

/* Animated grid background */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,255,180,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,180,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridPulse 8s ease-in-out infinite;
  pointer-events: none;
}
@keyframes gridPulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

/* Scanline effect */
.stApp::after {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
  pointer-events: none;
}

/* Override streamlit wrappers */
.main .block-container {
  padding: 2rem 3rem;
  max-width: 1400px;
  position: relative; z-index: 1;
}
header[data-testid="stHeader"] { display: none; }
div[data-testid="stSidebar"] { display: none; }
.stFileUploader > div { display: none; }

/* ── HEADER ── */
.header-wrap {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 3rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(0,255,180,0.15);
}
.logo-mark {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.4em;
  color: rgba(0,255,180,0.5);
  margin-bottom: 8px;
}
.header-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 5vw, 3.8rem);
  font-weight: 900;
  line-height: 1;
  color: #fff;
  text-shadow: 0 0 40px rgba(0,255,180,0.3);
  letter-spacing: -0.02em;
}
.header-title span {
  color: #00ffb4;
  text-shadow: 0 0 20px #00ffb4, 0 0 60px rgba(0,255,180,0.4);
}
.header-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  color: rgba(0,255,180,0.45);
  margin-top: 10px;
  letter-spacing: 0.08em;
}
.header-badge {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  color: #00ffb4;
  border: 1px solid rgba(0,255,180,0.3);
  padding: 6px 14px;
  border-radius: 2px;
  background: rgba(0,255,180,0.04);
  letter-spacing: 0.1em;
  margin-top: 8px;
  display: inline-block;
}

/* ── UPLOAD ZONE ── */
.upload-outer {
  border: 1px solid rgba(0,255,180,0.2);
  background: rgba(0,255,180,0.02);
  border-radius: 4px;
  padding: 3rem;
  text-align: center;
  position: relative;
  margin-bottom: 2.5rem;
  transition: border-color .3s;
}
.upload-outer::before, .upload-outer::after {
  content: '';
  position: absolute;
  width: 16px; height: 16px;
  border-color: #00ffb4;
  border-style: solid;
}
.upload-outer::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.upload-outer::after  { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }
.upload-icon { font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 12px #00ffb4); }
.upload-text { font-family: 'Orbitron', monospace; font-size: 0.8rem; color: rgba(0,255,180,0.6); letter-spacing: 0.2em; }
.upload-sub  { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: rgba(255,255,255,0.2); margin-top: 6px; }

/* ── VERDICT ── */
.verdict-wrap {
  padding: 2rem 2.5rem;
  margin-bottom: 2.5rem;
  position: relative;
  border-radius: 4px;
}
.verdict-fake {
  background: linear-gradient(135deg, rgba(255,40,40,0.06) 0%, rgba(255,0,80,0.03) 100%);
  border: 1px solid rgba(255,40,40,0.35);
  box-shadow: inset 0 0 60px rgba(255,40,40,0.05), 0 0 40px rgba(255,40,40,0.1);
}
.verdict-real {
  background: linear-gradient(135deg, rgba(0,255,180,0.06) 0%, rgba(0,200,255,0.03) 100%);
  border: 1px solid rgba(0,255,180,0.35);
  box-shadow: inset 0 0 60px rgba(0,255,180,0.05), 0 0 40px rgba(0,255,180,0.1);
}
.verdict-eyebrow {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.3em;
  color: rgba(255,255,255,0.3);
  margin-bottom: 8px;
}
.verdict-label-fake {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 6vw, 4rem);
  font-weight: 900; letter-spacing: 0.05em;
  color: #ff2850;
  text-shadow: 0 0 30px rgba(255,40,80,0.6), 0 0 80px rgba(255,40,80,0.2);
  line-height: 1;
}
.verdict-label-real {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 6vw, 4rem);
  font-weight: 900; letter-spacing: 0.05em;
  color: #00ffb4;
  text-shadow: 0 0 30px rgba(0,255,180,0.6), 0 0 80px rgba(0,255,180,0.2);
  line-height: 1;
}
.verdict-meta {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  color: rgba(255,255,255,0.35);
  margin-top: 12px;
  letter-spacing: 0.05em;
}
.conf-num { font-size: 1rem; color: rgba(255,255,255,0.7); }

/* Risk bar */
.risk-track {
  height: 3px; background: rgba(255,255,255,0.06);
  border-radius: 2px; margin: 16px 0 4px;
  position: relative; overflow: hidden;
}
.risk-fill-fake {
  height: 100%;
  background: linear-gradient(90deg, transparent, #ff2850);
  border-radius: 2px;
  box-shadow: 0 0 10px #ff2850;
}
.risk-fill-real {
  height: 100%;
  background: linear-gradient(90deg, #00ffb4, transparent);
  border-radius: 2px;
  box-shadow: 0 0 10px #00ffb4;
}
.risk-labels {
  display: flex; justify-content: space-between;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem; color: rgba(255,255,255,0.18);
  letter-spacing: 0.15em;
}

/* ── PANEL GRID ── */
.panel {
  background: rgba(255,255,255,0.015);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 4px;
  padding: 1.2rem;
  position: relative;
}
.panel::before {
  content: '';
  position: absolute;
  top: -1px; left: -1px;
  width: 30px; height: 30px;
  border-top: 1px solid rgba(0,255,180,0.4);
  border-left: 1px solid rgba(0,255,180,0.4);
  border-radius: 4px 0 0 0;
}
.panel-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.25em;
  color: rgba(0,255,180,0.45);
  text-transform: uppercase;
  margin-bottom: 10px;
}
.panel-note {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; color: rgba(255,255,255,0.18);
  margin-top: 8px; letter-spacing: 0.04em;
}

/* ── METRIC CARDS ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 2rem; }
.metric-card {
  background: rgba(255,255,255,0.015);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 4px;
  padding: 1.2rem;
  position: relative;
}
.metric-card::after {
  content: '';
  position: absolute; bottom: 0; left: 0;
  height: 1px; width: 40%;
  background: rgba(0,255,180,0.3);
}
.metric-lbl {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem; letter-spacing: 0.2em;
  color: rgba(255,255,255,0.25); margin-bottom: 8px;
}
.metric-val {
  font-family: 'Orbitron', monospace;
  font-size: 1.4rem; font-weight: 700;
  color: #fff;
}
.metric-val span { color: #00ffb4; }

/* Streamlit overrides to keep file uploader visible */
.stFileUploader { position: relative; z-index: 2; }
.stFileUploader > div {
  display: flex !important;
  background: transparent !important;
  border: 1px solid rgba(0,255,180,0.2) !important;
  border-radius: 4px !important;
  padding: 1.5rem !important;
}
.stFileUploader label { color: rgba(0,255,180,0.6) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.15em !important; }
section[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
section[data-testid="stFileUploaderDropzone"] button { background: rgba(0,255,180,0.08) !important; border: 1px solid rgba(0,255,180,0.3) !important; color: #00ffb4 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; border-radius: 2px !important; }
section[data-testid="stFileUploaderDropzone"] p { color: rgba(255,255,255,0.2) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.6rem !important; }

/* Spinner */
.stSpinner > div { border-top-color: #00ffb4 !important; }

/* Error */
.stAlert { background: rgba(255,40,80,0.08) !important; border: 1px solid rgba(255,40,80,0.3) !important; border-radius: 4px !important; color: #ff6080 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; }

img { border-radius: 3px !important; }
</style>
""", unsafe_allow_html=True)


# ── MODEL ─────────────────────────────────────────────────────
def get_hybrid_model():
    m = models.efficientnet_b0(weights=None)
    orig = m.features[0][0]
    m.features[0][0] = nn.Conv2d(4, orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride,
        padding=orig.padding, bias=False)
    m.classifier = nn.Sequential(nn.Dropout(0.4),
                                  nn.Linear(m.classifier[1].in_features, 1))
    return m

def get_fft_channel(t):
    gray  = 0.299*t[0] + 0.587*t[1] + 0.114*t[2]
    fshift = torch.fft.fftshift(torch.fft.fft2(gray))
    mag   = torch.log(torch.abs(fshift) + 1)
    mag   = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.unsqueeze(0)

def get_gradcam(model, inp):
    act, grd = {}, {}
    tgt = model.features[-1]
    h1 = tgt.register_forward_hook(lambda m,i,o: act.update({"f": o.detach()}))
    h2 = tgt.register_full_backward_hook(lambda m,gi,go: grd.update({"f": go[0].detach()}))
    x = inp.clone().requires_grad_(True)
    model.zero_grad()
    torch.sigmoid(model(x)).squeeze().backward()
    h1.remove(); h2.remove()
    w = grd["f"].mean(dim=(2,3), keepdim=True)
    c = torch.relu((w * act["f"]).sum(1)).squeeze().cpu().numpy()
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    return cv2.resize(c, (224, 224))

def overlay_cam(rgb, cam):
    h = cm.inferno(cam)[:,:,:3]
    return ((0.5*rgb + 0.5*h).clip(0,1)*255).astype(np.uint8)

val_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(180),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@st.cache_resource
def load_model(path="deepfake_hybrid_effnet_v1.pth"):
    try:
        m = get_hybrid_model()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.eval()
    except Exception as e:
        st.error(f"MODEL LOAD FAILURE · {e}")
        return None

model = load_model()

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div>
    <div class="logo-mark">// FORENSIC INTELLIGENCE SYSTEM v2.1</div>
    <div class="header-title">SPEC<span>TRE</span></div>
    <div class="header-sub">SYNTHETIC PATTERN EXAMINATION &amp; CLASSIFICATION THROUGH RESIDUAL ENCODING</div>
  </div>
  <div style="text-align:right;padding-top:8px;">
    <div class="header-badge">AUC · 0.9270</div><br>
    <div class="header-badge" style="margin-top:6px;">RGB + FFT FUSION</div><br>
    <div class="header-badge" style="margin-top:6px;">FF++ C23 TRAINED</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────
uploaded = st.file_uploader("// UPLOAD TARGET IMAGE", type=["jpg","jpeg","png"])

if not uploaded:
    st.markdown("""
    <div class="upload-outer">
      <div class="upload-icon">👁</div>
      <div class="upload-text">AWAITING TARGET ACQUISITION</div>
      <div class="upload-sub">DRAG &amp; DROP · JPG / PNG · FACE IMAGERY</div>
    </div>
    """, unsafe_allow_html=True)

# ── ANALYSIS ──────────────────────────────────────────────────
if uploaded and model:
    img   = Image.open(uploaded).convert("RGB")
    img_t = val_tf(img)
    fft_t = get_fft_channel(img_t)
    inp   = torch.cat([img_t, fft_t], dim=0).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(inp)).item()

    verdict = "FAKE" if prob > 0.5 else "REAL"
    conf    = prob if prob > 0.5 else 1 - prob
    pct     = int(prob * 100)

    mean   = np.array([0.485,0.456,0.406])
    std    = np.array([0.229,0.224,0.225])
    rgb_np = (img_t.permute(1,2,0).numpy() * std + mean).clip(0,1)

    # ── VERDICT ───────────────────────────────────────────────
    if verdict == "FAKE":
        st.markdown(f"""
        <div class="verdict-wrap verdict-fake">
          <div class="verdict-eyebrow">// ANALYSIS COMPLETE · THREAT DETECTED</div>
          <div class="verdict-label-fake">⚡ SYNTHETIC DETECTED</div>
          <div class="verdict-meta">
            FAKE PROBABILITY &nbsp;<span class="conf-num">{prob:.4f}</span>
            &nbsp;·&nbsp; CONFIDENCE &nbsp;<span class="conf-num">{conf:.2%}</span>
          </div>
          <div class="risk-track">
            <div class="risk-fill-fake" style="width:{pct}%;"></div>
          </div>
          <div class="risk-labels"><span>AUTHENTIC</span><span>SYNTHETIC</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-wrap verdict-real">
          <div class="verdict-eyebrow">// ANALYSIS COMPLETE · NO THREAT</div>
          <div class="verdict-label-real">◈ AUTHENTIC CONFIRMED</div>
          <div class="verdict-meta">
            FAKE PROBABILITY &nbsp;<span class="conf-num">{prob:.4f}</span>
            &nbsp;·&nbsp; CONFIDENCE &nbsp;<span class="conf-num">{conf:.2%}</span>
          </div>
          <div class="risk-track">
            <div class="risk-fill-real" style="width:{100-pct}%;"></div>
          </div>
          <div class="risk-labels"><span>AUTHENTIC</span><span>SYNTHETIC</span></div>
        </div>""", unsafe_allow_html=True)

    # ── THREE PANELS ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="panel"><div class="panel-title">// TARGET · RGB SPATIAL</div>', unsafe_allow_html=True)
        st.image(rgb_np, use_container_width=True)
        st.markdown('<div class="panel-note">224×224 · CENTERCROP PREPROCESSED</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title">// FREQUENCY DOMAIN · FFT MAGNITUDE</div>', unsafe_allow_html=True)
        fft_np = fft_t.squeeze().numpy()
        fft_col = cm.plasma(fft_np)[:,:,:3].astype(np.float32)
        st.image(fft_col, use_container_width=True)
        st.markdown('<div class="panel-note">REAL → CROSS PATTERN · GAN → GRID PERIODICITY</div></div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="panel"><div class="panel-title">// GRAD-CAM · NEURAL ATTENTION MAP</div>', unsafe_allow_html=True)
        with st.spinner("COMPUTING FORENSIC HEATMAP..."):
            cam     = get_gradcam(model, inp)
            overlay = overlay_cam(rgb_np, cam)
        st.image(overlay, use_container_width=True)
        st.markdown('<div class="panel-note">INFERNO COLORMAP · HIGH = SUSPICIOUS REGION</div></div>', unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────
    fft_energy = float(fft_t.mean())
    fft_std    = float(fft_t.std())
    cam_peak   = float(cam.max())
    cam_area   = float((cam > 0.5).mean() * 100)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-lbl">// FFT MEAN ENERGY</div>
        <div class="metric-val"><span>{fft_energy:.4f}</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-lbl">// FFT STD DEVIATION</div>
        <div class="metric-val"><span>{fft_std:.4f}</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-lbl">// GRAD-CAM PEAK</div>
        <div class="metric-val"><span>{cam_peak:.4f}</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-lbl">// SUSPICIOUS AREA</div>
        <div class="metric-val"><span>{cam_area:.1f}</span>%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid rgba(255,255,255,0.05);
    font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(255,255,255,0.12);
    letter-spacing:0.15em;display:flex;justify-content:space-between;">
      <span>SPECTRE · FORENSIC INTELLIGENCE SYSTEM</span>
      <span>EFFICIENTNET-B0 · RGB+FFT · FF++ C23</span>
      <span>AUC 0.9270 · GRAD-CAM XAI</span>
    </div>
    """, unsafe_allow_html=True)
