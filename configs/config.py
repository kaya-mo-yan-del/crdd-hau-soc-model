import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODEL_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "results"

# Audio & Spectrogram Parameters
SAMPLE_RATE = 16000
DURATION = 5                          # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION  # fixed waveform length (80,000 samples)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
IMAGE_SIZE = 224

# Training Hyperparameters
CLASSES = ["Healthy", "Sick", "None"]          # label order, fixed
CLASS_DISPLAY_NAMES = ["Healthy", "Respiratory Distress", "Background Noise"]  # thesis figure labels
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 16
LEARNING_RATE = 3e-4        # fine-tuning rate for a pretrained backbone
WEIGHT_DECAY = 1e-4         # L2 regularization (AdamW)
LABEL_SMOOTHING = 0.1       # regularization for a small dataset
EPOCHS = 40
EARLY_STOP_PATIENCE = 8     # epochs without val improvement before stopping
SEED = 42

# DataLoader
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Ensure output directories exist
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
