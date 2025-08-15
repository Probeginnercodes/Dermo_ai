---
title: DermoAI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: streamlit  # Must match your framework
sdk_version: "1.32.0"
app_file: app.py  # Must point to your main file
pinned: false
---

**🩺 DermoAI — Multimodal Skin Disease Diagnosis**
📌 Overview

DermoAI is an AI-powered diagnostic tool that leverages multimodal learning (image, text, and audio inputs) to assist in skin disease detection. The system uses deep learning models to extract features from each modality, applies an attention-based fusion layer to weight their importance dynamically, and outputs a unified diagnosis.

This project is implemented in PyTorch and deployed via Streamlit for an interactive web interface.

**🚀 Features**

Image Analysis — Detects skin abnormalities from uploaded images.
Text Understanding — Processes patient-provided symptom descriptions.
Audio Processing — Extracts diagnostic cues from recorded speech.
Attention-Based Fusion — Dynamically learns which modality is most important for each case.
Interactive Web App — User-friendly interface built with Streamlit.

**📚 Model Architecture — AttentionFusionLayer**

Extract Features from each modality.
Concatenate Features into a single vector.
Attention Network learns weights [w_image, w_text, w_audio].
Weighted Summation combines features.
Fusion Network refines the combined feature.
L2 Normalization ensures stable output.

**⚙️ Installation & Setup
1️⃣ Clone the repository**
git clone https://github.com/yourusername/DermoAI.git
cd DermoAI

**2️⃣ Install dependencies**
pip install -r requirements.txt

**3️⃣ Run the Streamlit app**
streamlit run app.py

**🖼️ Usage**
Upload a skin image.
Optionally provide symptom descriptions in text.
Optionally record or upload audio describing symptoms.
Click "Analyze" to receive predictions.

**📦 Requirements**
Python 3.8+
PyTorch
Transformers
Streamlit
Librosa
OpenCV
Pillow
Install via:
pip install torch torchvision torchaudio transformers streamlit librosa opencv-

**👨‍💻 Author**
Ashtosh Tiwari
MSc Artificial Intelligence — Berlin, Germany

# Your App Description
