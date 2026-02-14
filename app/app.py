import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\illegal-logging-detector"
MODEL_PATH = os.path.join(BASE_DIR, "models", "final", "resnet18_best.pth")
SR         = 22050
DURATION   = 5
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ────────────────────────────────────────────────────────────────────
class ResNetLogging(nn.Module):
    def __init__(self):
        super(ResNetLogging, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.resnet(x)

# Load model
model = ResNetLogging().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# ── Preprocessing ─────────────────────────────────────────────────────────
def audio_to_melspectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    target_length = SR * DURATION
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db

def plot_spectrogram(mel):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Mel Spectrogram of Input Audio")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bands")
    plt.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig

# ── Prediction ────────────────────────────────────────────────────────────
def predict(audio_file):
    if audio_file is None:
        return "⚠️ Please upload an audio file", 0.0, 0.0, None
    
    try:
        # Preprocess
        mel = audio_to_melspectrogram(audio_file)
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        pred_label  = pred.item()
        confidence  = conf.item() * 100
        logging_prob    = probs[0][1].item() * 100
        non_logging_prob = probs[0][0].item() * 100
        
        # Result
        if pred_label == 1:
            result = f"🚨 ILLEGAL LOGGING DETECTED ({confidence:.1f}% confidence)"
        else:
            result = f"✅ FOREST NORMAL ({confidence:.1f}% confidence)"
        
        # Plot
        fig = plot_spectrogram(mel)
        
        return result, round(logging_prob, 1), round(non_logging_prob, 1), fig
    
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, 0.0, None

# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Illegal Logging Detector", theme=gr.themes.Monochrome()) as demo:
    
    gr.Markdown("""
    # 🌲 Illegal Logging Detection System
    ### Acoustic AI for Rainforest Conservation
    Upload a forest audio clip to detect illegal logging activity using deep learning.
    The model analyzes audio spectrograms using a ResNet18 architecture trained on 
    rainforest and logging sounds.
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio File (mp3, wav, ogg)",
                type="filepath"
            )
            submit_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
            
            gr.Markdown("""
            **Detects:**
            - 🪚 Chainsaw sounds
            - 🚛 Logging trucks
            - ⚙️ Heavy machinery
            
            **Normal forest sounds:**
            - 🐦 Birds
            - 🌧️ Rain
            - 💨 Wind
            """)
        
        with gr.Column(scale=2):
            result_text = gr.Textbox(
                label="Detection Result",
                lines=2,
                interactive=False
            )
            with gr.Row():
                logging_prob    = gr.Number(label="🚨 Logging Probability %")
                non_logging_prob = gr.Number(label="✅ Normal Forest Probability %")
            
            spectrogram_plot = gr.Plot(label="Audio Spectrogram")
    
    gr.Markdown("""
    ---
    **How it works:** Audio is converted to a Mel Spectrogram (visual representation of sound),
    which is then analyzed by a ResNet18 convolutional neural network fine-tuned on 
    rainforest acoustic data. 
    
    **Model Performance:** 81.2% Accuracy | 83.3% Precision | 0.937 AUC
    """)
    
    submit_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[result_text, logging_prob, non_logging_prob, spectrogram_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True)