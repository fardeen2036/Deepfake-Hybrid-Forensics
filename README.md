# 🛡️ Hybrid-Forensic: Deepfake Detection via RGB-Frequency Fusion

A real-time **voice-activated and gesture-controlled system** using OpenCV, MediaPipe, and PyAutoGUI for smart human-computer interaction. This Python-based project enables users to control the mouse cursor and media functions (like volume and playback) through hand gestures and switch modes using voice commands.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hands-orange?style=flat-square" alt="MediaPipe">
  <img src="https://img.shields.io/badge/PyAutoGUI-Automation-grey?style=flat-square" alt="PyAutoGUI">
  <img src="https://img.shields.io/badge/SpeechRecognition-Voice%20Control-red?style=flat-square" alt="Speech">
</p>

---

## 🧠 Project Summary

This system uses:

* **Computer Vision** for facial localized analysis using an `EfficientNet-B0` backbone.
* **Signal Processing (FFT)** to identify "checkerboard" artifacts in the frequency domain.
* **Grad-CAM Explainability** to visualize the specific facial zones triggering the "Fake" verdict.
* **Robust Validation** using video-level splits to ensure zero data leakage.

## 📊 Model Performance

| Metric | Result | Functionality |
| :--- | :--- | :--- |
| **Validation AUC** | `0.9270` | High reliability across unseen video datasets |
| **Overall Accuracy** | `87%` | General classification performance |
| **Recall** | `80%` | Ability to catch actual fake videos |
| **Precision** | `73%` | Accuracy of "Fake" predictions |

## 🔍 Forensic Visualization

* **Spatial (RGB):** Detects texture inconsistencies and blending errors.
* **Frequency (FFT):** Detects mathematical "stars" or grids left by GAN upsampling.
