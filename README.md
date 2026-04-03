# 🛡️ Hybrid-Forensic: Deepfake Detection via RGB-Frequency Fusion

A real-time **forensic deepfake detection pipeline** using a hybrid 4-channel approach. This system leverages spatial facial features and frequency-domain artifacts (FFT) to distinguish natural camera sensor noise from AI-generated grid patterns.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/EfficientNet-B0-green?style=flat-square" alt="Model">
  <img src="https://img.shields.io/badge/Forensics-Signal%20Processing-orange?style=flat-square" alt="Forensics">
</p>

---

## 🧠 Project Summary

Standard detectors are often fooled by **identity leakage**. This system mitigates that by using:

* **Computer Vision** for facial localized analysis using an `EfficientNet-B0` backbone.
* **Signal Processing (FFT)** to identify "checkerboard" artifacts in the frequency domain.
* **Grad-CAM Explainability** to visualize the specific facial zones triggering the "Fake" verdict.
* **Robust Validation** using video-level splits to ensure zero data leakage.

---

## 📊 Model Performance

Based on the final hybrid training epoch, the model achieved **high reliability** across unseen video datasets:

| Metric | Result |
| :--- | :--- |
| **Validation AUC** | `0.9270` |
| **Overall Accuracy** | `87%` |
| **Fake Detection Recall** | `80%` |
| **Precision (Fake Class)** | `73%` |

---

## 🛠️ Key Functionalities

| Feature | Description |
| :--- | :--- |
| **4-Channel Input** | Processes RGB + a custom **FFT Magnitude** channel for forensic depth. |
| **CenterCrop(180)** | Focuses purely on facial **"Forensic Zones"** to ignore background bias. |
| **Frequency Masking** | Isolates AI-generated periodic noise from standard JPEG compression. |
| **Explainable AI** | Heatmaps (**Grad-CAM**) pinpointing exactly where the AI detected manipulation. |

---

## 🔍 Forensic Visualization

### **Spatial vs. Frequency Analysis**
The model doesn't just "look" at pixels; it **"listens"** to the mathematical noise.

* **Spatial (RGB):** Detects texture inconsistencies, blending errors, and lighting mismatches.
* **Frequency (FFT):** Detects mathematical **"stars"** or grids left by GAN upsampling layers.

---
