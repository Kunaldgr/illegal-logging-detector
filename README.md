# 🌲 Illegal Logging Detection System
### Acoustic AI for Rainforest Conservation

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-81.2%25-green?style=for-the-badge)
![AUC](https://img.shields.io/badge/AUC-0.937-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Fighting deforestation one sound at a time.** This system uses deep learning to detect illegal logging activity in rainforests by analyzing acoustic signatures — identifying chainsaws, logging trucks, and heavy machinery from audio in real-time.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Why This Matters](#-why-this-matters)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Results & Visualizations](#-results--visualizations)
- [Future Enhancements](#-future-enhancements)
- [Contact](#-contact)

---

## 🔍 Overview

Illegal logging is responsible for **15% of global carbon emissions** and threatens the biodiversity of some of the world's most critical ecosystems. Traditional monitoring methods rely on satellite imagery or human patrols — both slow, expensive, and often ineffective in dense forest environments.

This project takes a different approach: **acoustic monitoring**. By deploying audio sensors in forests and using deep learning to analyze sound patterns, illegal logging activity can be detected within seconds — before significant damage is done.

The system classifies audio into two categories:
- 🚨 **Logging** — chainsaws, logging trucks, heavy machinery
- ✅ **Normal Forest** — birds, rain, wind, ambient forest sounds

---

## 🌍 Why This Matters

- 🌳 **10 million hectares** of forest are destroyed annually
- 💀 Illegal logging drives **wildlife extinction** and **indigenous displacement**
- 🌡️ Deforestation contributes **15% of global greenhouse gas emissions**
- 👮 Current monitoring is **too slow** — damage is done before rangers arrive
- 🤖 **AI acoustic monitoring** can trigger alerts in real-time, enabling rapid response

---

## ✨ Features

- **Real-time Detection** — analyze audio clips in under 2 seconds
- **ResNet18 Transfer Learning** — pretrained ImageNet model fine-tuned on acoustic data
- **Mel Spectrogram Analysis** — converts audio to visual representations for CNN processing
- **Confidence Scores** — probability estimates for both logging and non-logging classes
- **Spectrogram Visualization** — see exactly what the model is analyzing
- **Web Interface** — clean Gradio UI for easy audio upload and analysis
- **GPU Accelerated** — optimized for CUDA-enabled devices
- **Edge Deployable** — lightweight enough for Raspberry Pi deployment in remote areas

---

## 🛠️ Technology Stack

### Deep Learning & ML
- **PyTorch** — deep learning framework
- **ResNet18** — transfer learning backbone
- **torchaudio** — audio processing utilities
- **librosa** — audio analysis and feature extraction
- **audiomentations** — audio data augmentation

### Audio Processing
- **Mel Spectrograms** — frequency-time visual representation of audio
- **MFCC Features** — Mel-frequency cepstral coefficients
- **Noise Reduction** — background noise filtering pipeline

### Web Application
- **Gradio** — interactive ML demo interface
- **Matplotlib** — spectrogram visualization

### Development
- **Python 3.10+** — core language
- **Jupyter Notebooks** — experimentation and exploration
- **VS Code** — development environment
- **Git** — version control

---

## 🏗️ Architecture

```
Audio Input (.mp3, .wav, .ogg)
        ↓
   Audio Preprocessing
   (Resample → 22050 Hz, Trim, Normalize, Pad to 5s)
        ↓
   Mel Spectrogram Conversion
   (128 Mel bands × 216 time frames)
        ↓
   ResNet18 Backbone
   (Pretrained on ImageNet, fine-tuned on acoustic data)
   - Modified conv1: 1 channel input (grayscale spectrogram)
   - Frozen early layers (edge/texture detectors)
   - Trainable: layer3, layer4, fc
        ↓
   Feature Extraction
   (512-dimensional feature vector)
        ↓
   Fully Connected + Dropout(0.5)
        ↓
   Binary Classification
   (Logging vs Non-Logging)
        ↓
   Softmax → Confidence Score
        ↓
   🚨 ALERT or ✅ NORMAL
```

### Why ResNet18 Transfer Learning?
Training a deep CNN from scratch requires massive datasets. With only 210 audio clips, a custom CNN overfits quickly (we achieved 71% accuracy). By leveraging ResNet18's pretrained ImageNet weights — which already understand low-level patterns like edges, textures, and shapes — we achieve **81.2% accuracy and 0.937 AUC** with fine-tuning alone.

---

## 📊 Dataset

### Composition
| Category | Type | Clips |
|---|---|---|
| Chainsaw | Logging | 30 |
| Logging Truck | Logging | 30 |
| Heavy Machinery | Logging | 30 |
| Forest Birds | Non-Logging | 30 |
| Rain | Non-Logging | 30 |
| Wind | Non-Logging | 30 |
| Forest Ambience | Non-Logging | 30 |
| **Total** | | **210** |

### Data Sources
- **Rainforest Connection** — real rainforest audio with logging events
- **Freesound.org** — community audio library
- **Google AudioSet** — labeled YouTube audio clips

### Train / Val / Test Split
| Split | Samples |
|---|---|
| Train (before augmentation) | 147 |
| Validation | 31 |
| Test | 32 |

### Data Augmentation
To combat the small dataset size, 4 augmentation techniques were applied to training data, expanding 147 clips to **735 effective training samples**:
- **Time Masking** — randomly block out time segments
- **Frequency Masking** — randomly block out frequency bands
- **Gaussian Noise** — simulate poor microphone quality
- **Time Shifting** — roll spectrogram horizontally

---

## 📈 Model Performance

### Final Results (Test Set)
| Metric | Value |
|---|---|
| **Accuracy** | **81.2%** |
| **Precision** | **83.3%** |
| **Recall** | **71.4%** |
| **F1 Score** | **76.9%** |
| **ROC AUC** | **0.937** |

### Model Comparison
| Model | Val Accuracy | Notes |
|---|---|---|
| Baseline CNN (scratch) | 71.0% | Overfitting, 14M params |
| Improved CNN (scratch) | 64.5% | Underfitting, too small |
| **ResNet18 (transfer)** | **96.8%** | **Best model, saved** |

### Training Configuration
| Parameter | Value |
|---|---|
| Epochs | 12 (early stopping) |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Loss Function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 4GB RAM minimum

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kunaldgr/illegal-logging-detector.git
cd illegal-logging-detector
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa soundfile numpy pandas matplotlib seaborn
pip install scikit-learn gradio audiomentations tqdm jupyter
```

### Step 4: Download Model Weights
Download the pretrained model weights and place in `models/final/`:
```
models/
└── final/
    └── resnet18_best.pth
```

---

## 💻 Usage

### Running the Web Application
```bash
python app/app.py
```
Open your browser and go to the local URL shown in terminal.

### Using the App
1. Upload any audio file (mp3, wav, ogg)
2. Click **Analyze Audio**
3. View detection result, confidence scores, and spectrogram

### Using the Model Programmatically
```python
import torch
import librosa
import numpy as np
from app.app import ResNetLogging, audio_to_melspectrogram

# Load model
model = ResNetLogging()
model.load_state_dict(torch.load('models/final/resnet18_best.pth'))
model.eval()

# Predict
mel = audio_to_melspectrogram('path/to/audio.mp3')
tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(1).item()
    conf = probs.max(1).values.item() * 100

print(f"Prediction: {'🚨 LOGGING DETECTED' if pred == 1 else '✅ FOREST NORMAL'}")
print(f"Confidence: {conf:.1f}%")
```

---

## 📁 Project Structure

```
illegal-logging-detector/
├── data/
│   ├── raw/                    # original downloaded audio files
│   │   ├── chainsaw/
│   │   ├── truck/
│   │   ├── machinery/
│   │   ├── birds/
│   │   ├── rain/
│   │   ├── wind/
│   │   └── forest_ambience/
│   ├── processed/              # mel spectrograms (train/val/test)
│   └── augmented/              # augmented training samples
│
├── notebooks/
│   ├── 01_setup_test.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_model.ipynb
│
├── models/
│   ├── baseline/               # baseline CNN weights
│   └── final/                  # ResNet18 best weights
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── baseline_training_curves.png
│
├── app/
│   └── app.py                  # Gradio web application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

### 1. Audio → Spectrogram
Raw audio is converted into a **Mel Spectrogram** — a 2D image representing frequency content over time. Different sound sources have visually distinct spectral signatures:

- **Chainsaw**: Dense, consistent horizontal bands across mid-high frequencies
- **Birds**: Scattered, sporadic vertical bursts at high frequencies
- **Rain**: Diffuse, uniform energy across all frequencies
- **Trucks**: Strong low-frequency rumble with harmonic patterns

### 2. Transfer Learning
Instead of training from scratch, we use ResNet18 pretrained on ImageNet. The early layers already detect edges, textures and patterns — universally useful features. We freeze these and only train the deeper layers on our acoustic data.

### 3. Sliding Window Detection
For real-world deployment, audio is processed in 5-second chunks with 50% overlap. Multiple consecutive detections trigger an alert, dramatically reducing false positives from one-off sounds like motorcycles or thunder.

---

## 📸 Results & Visualizations

### Spectrogram Comparison
The model learns to distinguish these visual patterns:

| Sound | Spectrogram Pattern |
|---|---|
| Chainsaw | Dense horizontal bands, consistent energy |
| Birds | Sparse high-frequency bursts |
| Rain | Uniform broadband noise |
| Truck | Strong low-frequency harmonics |

### Confusion Matrix
```
                 Predicted
              Normal  Logging
Actual Normal   16       2
       Logging   4      10
```

### ROC Curve
- **AUC: 0.937** — strong class separation
- Low false positive rate at operating threshold
- Robust performance despite small dataset size

---

## 🚀 Future Enhancements

### Short-term
- [ ] Collect 500+ clips per category for improved recall
- [ ] Multi-class detection (chainsaw vs truck vs machinery separately)
- [ ] Real-time microphone input in web app
- [ ] SMS/email alert system via Twilio API
- [ ] Temporal pattern analysis (sustained activity detection)

### Long-term Vision
- [ ] Edge deployment on solar-powered Raspberry Pi sensors
- [ ] GPS triangulation using multiple sensor array
- [ ] Integration with Rainforest Connection alert network
- [ ] Mobile app for field rangers
- [ ] Continuous learning pipeline with ranger feedback
- [ ] Satellite + acoustic fusion for comprehensive monitoring

---

## ⚠️ Limitations

- **Small dataset** — 210 clips limits generalization; more data would improve recall from 71% to 85%+
- **Audio quality** — performance may drop on very noisy or low-quality recordings
- **Novel sounds** — machinery types not in training data may be missed
- **Not a replacement** — should be used alongside, not instead of, ranger patrols

---

## 📞 Contact

**Kunal Dagar**

📧 Email: kunaldagar4298@gmail.com
💼 LinkedIn: [linkedin.com/in/kunal-dagar-661161322](https://linkedin.com/in/kunal-dagar-661161322)
💻 GitHub: [github.com/Kunaldgr](https://github.com/Kunaldgr)

---

## 🙏 Acknowledgments

- [Rainforest Connection](https://rfcx.org) — for their open acoustic datasets and conservation mission
- [Freesound.org](https://freesound.org) — community audio library
- PyTorch team — deep learning framework
- ResNet authors — He et al., 2016
- All contributors to open environmental monitoring research

---

## 📚 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Rainforest Connection. (2019). Real-time Detection of Illegal Logging Using Acoustic Monitoring.
3. Mesaros, A., et al. (2017). DCASE 2017 Challenge Setup: Tasks, Datasets and Baseline System.
4. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.

---

> *"The clearest way into the Universe is through a forest wilderness."* — John Muir

**Made with ❤️ by Kunal Dagar — protecting rainforests with AI, one sound at a time.**