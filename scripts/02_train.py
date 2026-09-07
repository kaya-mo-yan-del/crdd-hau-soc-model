import sys
import random
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

import json

from configs import config
from src.data.audio_dataset import ChickenAudioDataset
from src.models.builder import build_model
from src.training.trainer import train_one_epoch, validate


def set_seed(seed: int):
    """Sets the seed for python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(split_file: Path, num_classes: int, device):
    counts = Counter()
    with open(split_file, "r") as f:
        for line in f:
            _, label = line.strip().rsplit(",", 1)
            counts[int(label)] += 1

    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    print(f"Class counts:  { {config.CLASSES[k]: v for k, v in sorted(counts.items())} }")
    print(f"Class weights: {weights.cpu().numpy().round(3)}")
    return weights


def save_history(history: dict, save_path: Path):
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Load Data (augmentation enabled for training only)
    train_ds = ChickenAudioDataset(
        config.SPLITS_DIR / "train.txt",
        sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS,
        image_size=config.IMAGE_SIZE, duration=config.DURATION,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        is_training=True,
    )
    val_ds = ChickenAudioDataset(
        config.SPLITS_DIR / "val.txt",
        sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS,
        image_size=config.IMAGE_SIZE, duration=config.DURATION,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        is_training=False,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=pin)

    # 2. Setup Model, Loss, Optimizer, Scheduler
    model = build_model(config.NUM_CLASSES).to(device)

    class_weights = compute_class_weights(config.SPLITS_DIR / "train.txt",
                                          config.NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=config.LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 3. Train Loop with history tracking + early stopping
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [], "val_mcc": [],
    }
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        # Macro-averaged metrics, so all three classes are weighted equally
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, labels=range(config.NUM_CLASSES),
            average="macro", zero_division=0)
        mcc = matthews_corrcoef(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)
        history["val_mcc"].append(mcc)

        save_history(history, config.RESULTS_DIR / "training_history.json")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{config.EPOCHS}  (lr={current_lr:.2e})")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"Val Macro — Precision: {precision:.4f} | Recall: {recall:.4f} | "
              f"F1: {f1:.4f} | MCC: {mcc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_path = config.MODEL_DIR / "crdd_model_v1.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} (Acc: {best_val_acc*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping: no val improvement for "
                      f"{config.EARLY_STOP_PATIENCE} epochs.")
                print("-" * 30)
                break
        print("-" * 30)

        scheduler.step()

    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Training history saved to {config.RESULTS_DIR / 'training_history.json'}")
    print("Run scripts/06_evaluation.py to generate the learning-curve, "
          "confusion-matrix, and metrics plots.")


if __name__ == "__main__":
    main()
