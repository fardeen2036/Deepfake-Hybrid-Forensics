🛡️ Hybrid-Forensic: Deepfake Detection via RGB-Frequency Fusion🚀 
The Problem: Why Standard AI Fails? - Most Deepfake detectors are easily "fooled" by specific faces or lighting they've seen before (Identity Leakage). They look at the image, but they don't "listen" to the math.
💡 The Solution: A 4-Channel Hybrid Approach - This project moves beyond standard RGB analysis. By injecting a 4th channel containing the 2D Discrete Fourier Transform (DFT), the model analyzes mathematical frequency artifacts (periodic noise) that are invisible to the human eye but are the "fingerprints" of AI generation.🛠️ Technical Innovation Architecture: Modified EfficientNet-B0 patched to accept 4-channel input (RGB + FFT).
Feature Engineering: Integrated a real-time FFT generator to capture "checkerboard artifacts" from GAN upsampling. 
Robust Validation: Implemented a strict Video-Level Split (ensuring no frames from the same video appear in both Train/Val) to prevent "cheating" by memorization.
Forensic Precision: Utilized CenterCrop(180) to force the model to focus on facial features, eliminating background bias.
📊 Performance & Results Metric Result Validation: AUC =  0.9270, Accuracy =  87%, Recall = (Fakes Caught) 80%
