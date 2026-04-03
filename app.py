import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Deepfake Hybrid Forensic Lab", layout="wide")
st.title("🛡️ Hybrid-Forensic: Deepfake Detection")
st.write("Detecting AI-generated artifacts using **RGB + Frequency (FFT) Fusion**.")

# --- 2. MODEL ARCHITECTURE ---
# Note: You must match the exact logic used in training
def get_hybrid_model():
    model = models.efficientnet_b0(weights=None)
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        4, original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding, 
        bias=False
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

# --- 3. FFT UTILITY ---
def get_fft_channel(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    # Handle normalization for FFT
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
    # Normalize to [0, 1]
    mag_min, mag_max = magnitude_spectrum.min(), magnitude_spectrum.max()
    magnitude_spectrum = (magnitude_spectrum - mag_min) / (mag_max - mag_min + 1e-8)
    return torch.tensor(magnitude_spectrum, dtype=torch.float32).unsqueeze(0)

# --- 4. PREPROCESSING ---
val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(180),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. SIDEBAR & MODEL LOADING ---
st.sidebar.header("Settings")
# Ensure deepfake_hybrid_effnet_v1.pth is in the same directory
model_path = st.sidebar.text_input("Model Weights Path", "deepfake_hybrid_effnet_v1.pth")

@st.cache_resource
def load_model(path):
    try:
        model = get_hybrid_model()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)

# --- 6. UPLOAD & ANALYSIS ---
uploaded_file = st.file_uploader("Upload a face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    img_t = val_tf(img)
    fft_t = get_fft_channel(img_t)
    input_tensor = torch.cat([img_t, fft_t], dim=0).unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).item()

    # UI Results
    verdict = "FAKE" if output > 0.5 else "REAL"
    confidence = output if output > 0.5 else 1 - output
    color = "red" if verdict == "FAKE" else "green"

    st.markdown(f"### Verdict: <span style='color:{color}'>{verdict}</span> ({confidence:.2%})", unsafe_content_allowed=True)

    # Visualization Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img, caption="Original Upload", use_container_width=True)
    
    with col2:
        fft_display = fft_t.squeeze().numpy()
        st.image(fft_display, caption="Frequency Channel (FFT)", use_container_width=True)
        
    with col3:
        # Simple placeholder for Model Focus/Heatmap
        st.info("Generating forensic heatmap...")
        # (You can add your Grad-CAM logic here later if needed)
        st.image(img.crop((img.width/4, img.height/4, 3*img.width/4, 3*img.height/4)), caption="Forensic Zoom")

else:
    st.info("Please upload an image and ensure model weights are present.")
