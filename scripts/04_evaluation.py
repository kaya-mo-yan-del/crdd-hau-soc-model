import sys
import json
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, matthews_corrcoef,
                             classification_report)

from configs import config
from src.data.audio_dataset import ChickenAudioDataset
from src.models.builder import build_model

SAVE_PREFIX = "Model_A_Raw"

# Label indices come from config.CLASSES order; figures use display names
CLASSES = config.CLASS_DISPLAY_NAMES


# ----------------------------------------------------------------------
# Part 1 — Per-epoch results from the saved training history
# ----------------------------------------------------------------------

def _plot_metric_per_epoch(epochs, values, metric_name, save_path,
                           train_values=None, as_percent=False):
    """Plots one metric across epochs. Blue = validation, red = train."""
    plt.figure(figsize=(9, 4.5))
    scale = 100 if as_percent else 1
    plt.plot(epochs, [v * scale for v in values],
             color="tab:blue", linewidth=1.4, label="Validation")
    if train_values is not None:
        plt.plot(epochs, [v * scale for v in train_values],
                 color="tab:red", linewidth=1.4, label="Train")
    unit = " (%)" if as_percent else ""
    plt.title(f"{metric_name} over Training", fontsize=10)
    plt.xlabel("Epoch", fontsize=9)
    plt.ylabel(f"{metric_name}{unit}", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc="best", fontsize=8, frameon=True, edgecolor="#888888")
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def per_epoch_outputs(results_dir: Path):
    """Builds the per-epoch table and one figure per metric from
    training_history.json."""
    history_path = config.RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        print(f"(No training history at {history_path} — skipping per-epoch "
              f"table and figures. Run 02_train.py first.)")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    n = len(history["val_loss"])
    epochs = list(range(1, n + 1))
    best_epoch = int(np.argmax(history["val_acc"]))  # 0-indexed

    # Per-epoch validation summary table
    columns = [
        ("Epoch", None),
        ("Val Accuracy", "val_acc"),
        ("Val Loss", "val_loss"),
        ("Macro Precision", "val_precision"),
        ("Macro Recall", "val_recall"),
        ("Macro F1", "val_f1"),
        ("MCC", "val_mcc"),
    ]
    available = [(name, key) for name, key in columns
                 if key is None or key in history]

    csv_path = results_dir / f"{SAVE_PREFIX}_Per_Epoch_Table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in available])
        for i in range(n):
            row = []
            for name, key in available:
                if key is None:
                    row.append(f"{i+1}*" if i == best_epoch else f"{i+1}")
                else:
                    row.append(f"{history[key][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    print("\n=== Validation Performance Summary (per epoch) ===")
    header = " | ".join(f"{name:>15s}" for name, _ in available)
    print(header)
    print("-" * len(header))
    for i in range(n):
        cells = []
        for name, key in available:
            if key is None:
                cells.append(f"{(str(i+1) + ('*' if i == best_epoch else '')):>15s}")
            else:
                cells.append(f"{history[key][i]:>15.6f}")
        print(" | ".join(cells))
    print("* peak validation accuracy (saved checkpoint)\n")

    # Per-metric figures
    _plot_metric_per_epoch(epochs, history["val_loss"], "Loss",
                           results_dir / f"{SAVE_PREFIX}_Epoch_Loss.png",
                           train_values=history.get("train_loss"))
    _plot_metric_per_epoch(epochs, history["val_acc"], "Accuracy",
                           results_dir / f"{SAVE_PREFIX}_Epoch_Accuracy.png",
                           train_values=history.get("train_acc"),
                           as_percent=True)
    for key, label in [("val_precision", "Precision"),
                       ("val_recall", "Recall"),
                       ("val_f1", "F1-score"),
                       ("val_mcc", "MCC")]:
        if key in history:
            fname = key.replace("val_", "").split("_")[0].capitalize()
            _plot_metric_per_epoch(
                epochs, history[key], label,
                results_dir / f"{SAVE_PREFIX}_Epoch_{fname}.png")
        else:
            print(f"({key} not in history — retrain with the updated "
                  f"02_train.py to get this curve.)")


# ----------------------------------------------------------------------
# Part 2 — Final evaluation on the held-out split
# ----------------------------------------------------------------------

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating Model A (Approach 1: Raw + SpecAugment)...\n")

    results_dir = config.RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    per_epoch_outputs(results_dir)

    # Held-out data: test.txt preferred, val.txt fallback
    split_path = config.SPLITS_DIR / "test.txt"
    if not split_path.exists() or split_path.stat().st_size == 0:
        print("Warning: test.txt not found/empty — falling back to val.txt "
              "(results will be optimistic).")
        split_path = config.SPLITS_DIR / "val.txt"
    print(f"Evaluating on: {split_path.name}\n")

    test_ds = ChickenAudioDataset(
        split_path,
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        image_size=config.IMAGE_SIZE,
        duration=config.DURATION,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        is_training=False,
    )
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    model = build_model(config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_DIR / "crdd_model_v1.pth",
                                     map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    print("Running audio files through the network. Please wait...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall system metrics
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(config.NUM_CLASSES),
        average="macro", zero_division=0)

    print("\n=== Overall System Metrics (Macro-Averaged) ===")
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {macro_p:.4f}")
    print(f"Recall   : {macro_r:.4f}")
    print(f"F1-Score : {macro_f1:.4f}")
    print(f"MCC      : {mcc:.4f}\n")

    print(classification_report(all_labels, all_preds,
                                labels=range(config.NUM_CLASSES),
                                target_names=CLASSES, digits=4,
                                zero_division=0))

    summary_path = results_dir / f"{SAVE_PREFIX}_Overall_Metrics.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{acc:.6f}"])
        writer.writerow(["Macro Precision", f"{macro_p:.6f}"])
        writer.writerow(["Macro Recall", f"{macro_r:.6f}"])
        writer.writerow(["Macro F1-Score", f"{macro_f1:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
    print(f"Saved: {summary_path}")

    # Testing prediction results
    total = len(all_labels)
    correct = int((all_preds == all_labels).sum())
    wrong = total - correct

    pred_table_path = results_dir / f"{SAVE_PREFIX}_Prediction_Results.csv"
    with open(pred_table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Total Samples", "Correct Prediction",
                         "Wrong Prediction", "Accuracy"])
        writer.writerow([total, correct, wrong, f"{acc*100:.2f}%"])
    print(f"Saved: {pred_table_path}")

    plt.figure(figsize=(7, 6))
    plt.pie([correct, wrong],
            labels=[f"Correct\n{correct:,}", f"Wrong\n{wrong:,}"],
            colors=["tab:blue", "tab:red"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    plt.title("Testing Prediction Results", fontsize=11)
    plt.tight_layout()
    pie_path = results_dir / f"{SAVE_PREFIX}_Prediction_Pie.png"
    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pie_path}")

    # Confusion matrix (rows = true label, columns = predicted label)
    cm = confusion_matrix(all_labels, all_preds, labels=range(config.NUM_CLASSES))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix", fontsize=11)
    plt.ylabel("True Label (Actual Audio)")
    plt.xlabel("Predicted Label (Model Output)")
    plt.tight_layout()
    cm_path = results_dir / f"{SAVE_PREFIX}_Confusion_Matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

    # Per-class precision/recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(config.NUM_CLASSES),
        average=None, zero_division=0)

    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label="Precision", color="tab:blue")
    rects2 = ax.bar(x, recall, width, label="Recall", color="tab:red")
    rects3 = ax.bar(x + width, f1, width, label="F1-score", color="#9aa0a6")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Precision, Recall, and F1-score per Class", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])
    ax.bar_label(rects1, fmt="%.2f", padding=3)
    ax.bar_label(rects2, fmt="%.2f", padding=3)
    ax.bar_label(rects3, fmt="%.2f", padding=3)
    plt.tight_layout()
    metrics_path = results_dir / f"{SAVE_PREFIX}_Metrics_Chart.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {metrics_path}")

    print("\nEvaluation Complete! Check the 'results' folder for your thesis "
          "tables (CSV) and figures (PNG).")


if __name__ == "__main__":
    evaluate()
