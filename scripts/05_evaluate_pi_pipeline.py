import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, matthews_corrcoef,
                             classification_report)

from configs import config
from pi_deploy.pi_monitor import preprocess, softmax

SAVE_PREFIX = "Model_A_Pi"
CLASSES = config.CLASS_DISPLAY_NAMES


def load_split(split_file: Path):
    """Reads path,label pairs. Missing paths are rebuilt from the current
    DATA_DIR in case the project folder was moved."""
    items = []
    with open(split_file, "r") as f:
        for line in f:
            path, label = line.strip().rsplit(",", 1)
            p = Path(path)
            if not p.exists():
                p = config.DATA_DIR / p.parent.name / p.name
            items.append((str(p), int(label)))
    return items


def load_waveform(wav_path: str) -> np.ndarray:
    """Returns mono float32 at 16 kHz, matching what pi_monitor.py feeds
    into preprocess()."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr,
                                 target_sr=config.SAMPLE_RATE)
    return audio


def main():
    results_dir = config.RESULTS_DIR / "librosa_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Test split: test.txt preferred, val.txt fallback
    split_path = config.SPLITS_DIR / "test.txt"
    if not split_path.exists() or split_path.stat().st_size == 0:
        print("Warning: test.txt not found/empty — falling back to val.txt "
              "(results will be optimistic).")
        split_path = config.SPLITS_DIR / "val.txt"
    print(f"Evaluating Pi pipeline on: {split_path.name}\n")
    items = load_split(split_path)

    # ONNX model: the exact file deployed to the Pi
    onnx_path = config.MODEL_DIR / "crdd_model_v1.onnx"
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found. Run 03_export_onnx.py first.")
        sys.exit(1)
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    print(f"Model: {onnx_path.name} (ONNX Runtime, CPU)\n")

    all_preds, all_labels, all_confs, filepaths = [], [], [], []
    print(f"Running {len(items)} clips through librosa preprocess + ONNX...")
    for i, (path, label) in enumerate(items, 1):
        waveform = load_waveform(path)
        x = preprocess(waveform)
        logits = sess.run(["logits"], {"input": x})[0][0]
        probs = softmax(logits)
        pred = int(np.argmax(probs))

        all_preds.append(pred)
        all_labels.append(label)
        all_confs.append(float(probs[pred]))
        filepaths.append(path)

        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(config.NUM_CLASSES),
        average="macro", zero_division=0)

    print("\n=== Deployed-Pipeline Metrics (librosa + ONNX, Macro-Averaged) ===")
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

    # Per-clip predictions
    preds_path = results_dir / f"{SAVE_PREFIX}_Predictions.csv"
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "predicted_label",
                         "confidence"])
        for path, t, p, c in zip(filepaths, all_labels, all_preds, all_confs):
            writer.writerow([path, config.CLASSES[t], config.CLASSES[p],
                             f"{c:.4f}"])
    print(f"Saved: {preds_path}")

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
    plt.title("Testing Prediction Results — Deployed Pipeline (librosa + ONNX)",
              fontsize=11)
    plt.tight_layout()
    pie_path = results_dir / f"{SAVE_PREFIX}_Prediction_Pie.png"
    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pie_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds,
                          labels=range(config.NUM_CLASSES))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Deployed Pipeline (librosa + ONNX)",
              fontsize=11)
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
    r1 = ax.bar(x - width, precision, width, label="Precision", color="tab:blue")
    r2 = ax.bar(x, recall, width, label="Recall", color="tab:red")
    r3 = ax.bar(x + width, f1, width, label="F1-score", color="#9aa0a6")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Per-Class Metrics — Deployed Pipeline (librosa + ONNX)",
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])
    for r in (r1, r2, r3):
        ax.bar_label(r, fmt="%.2f", padding=3)
    plt.tight_layout()
    chart_path = results_dir / f"{SAVE_PREFIX}_Metrics_Chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart_path}")

    # Side-by-side comparison against the training-pipeline evaluation
    ref_path = config.RESULTS_DIR / "Model_A_Raw_Overall_Metrics.csv"
    if ref_path.exists():
        with open(ref_path) as f:
            ref = {row[0]: float(row[1]) for row in csv.reader(f)
                   if row and row[0] != "Metric"}
        print("\n=== Pipeline Comparison ===")
        print(f"{'Metric':<18}{'Training (PyTorch)':>20}{'Deployed (ONNX)':>18}")
        for name, val in [("Accuracy", acc), ("Macro Precision", macro_p),
                          ("Macro Recall", macro_r), ("Macro F1-Score", macro_f1),
                          ("MCC", mcc)]:
            print(f"{name:<18}{ref.get(name, float('nan')):>20.4f}{val:>18.4f}")
        comp_path = results_dir / "Pipeline_Comparison.csv"
        with open(comp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Training Pipeline (PyTorch)",
                             "Deployed Pipeline (librosa+ONNX)"])
            for name, val in [("Accuracy", acc), ("Macro Precision", macro_p),
                              ("Macro Recall", macro_r),
                              ("Macro F1-Score", macro_f1), ("MCC", mcc)]:
                writer.writerow([name, f"{ref.get(name, float('nan')):.6f}",
                                 f"{val:.6f}"])
        print(f"\nSaved: {comp_path}")
    else:
        print("\n(Run 04_evaluation.py first to also get the side-by-side "
              "Pipeline_Comparison.csv table.)")

    print("\nDone. Per-clip predictions are in the Predictions CSV for "
          "inspecting any disagreements between the two pipelines.")


if __name__ == "__main__":
    main()
