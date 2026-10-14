# Chicken Respiratory Distress Detector (CRDD)

**Live dashboard: [crdd-hau-soc.vercel.app](https://crdd-hau-soc.vercel.app/)**
**Docker image: [`docker.io/k1mchir1ce/crdd-train`](https://hub.docker.com/r/k1mchir1ce/crdd-train)** · [Docker setup guide](docker/DOCKER.md)

CRDD is an edge AI system that listens to chickens and detects signs of respiratory
distress from their vocalizations. A Raspberry Pi 4B with a USB microphone records audio
in 5-second windows, classifies each window on-device, shows the result on an LCD, and
sends respiratory distress detections to a web dashboard so farm staff can monitor the flock remotely.

## Dashboard

[**crdd-hau-soc.vercel.app**](https://crdd-hau-soc.vercel.app/) is the web dashboard for
the deployed device. It shows the respiratory distress detections uploaded by the
Pi, with confidence score, detection time, and the recorded audio clip, plus whether the
Pi is currently online (based on a heartbeat the device sends every cycle).

Detections arrive in real time through Supabase (Postgres + Storage + Realtime). The Pi
writes to Supabase directly, and the dashboard reads from it.

## How the System Works

```
 USB mic ──> Raspberry Pi 4B ──────────────────────────────> 16x2 LCD
             record 5 s → log-mel spectrogram → ONNX model     (label + confidence)
                     │
                     │  Sick detections ≥ 70% confidence
                     v
             Supabase (clip storage + detections table + heartbeat)
                     │
                     v
             Web dashboard (crdd-hau-soc.vercel.app)
```

Each 5-second window is classified into one of three classes

| Label | Display Name | Meaning |
|---|---|---|
| `Healthy` | Healthy | Normal chicken vocalization |
| `Sick` | Respiratory Distress | Vocalization showing signs of respiratory illness |
| `None` | Background Noise | No relevant chicken vocalization |

On the Pi, any detection at or above 70% confidence is saved to the SD card. Only `Sick`
clips are uploaded to Supabase. Both local and cloud storage clean themselves up when
space runs low, deleting the lowest-confidence clips first.

## The Model

| Property | Value |
|---|---|
| Architecture | MobileNetV4 conv-small (`timm`) |
| Pretrained weights | ImageNet (`mobilenetv4_conv_small.e1200_r224_in1k`) |
| Input | 5 s of 16 kHz mono audio → 128-band log-mel spectrogram → 3 × 224 × 224 |
| Output | 3 class logits (Healthy, Sick, None) |
| Loss | Cross-entropy with inverse-frequency class weights and label smoothing (0.1) |
| Optimizer | AdamW (lr 3e-4, weight decay 1e-4) with cosine annealing |
| Regularization | Dropout, gradient clipping, waveform augmentation, SpecAugment |
| Deployment format | ONNX opset 17, run with ONNX Runtime on the Pi's CPU |

The spectrogram is treated like an image, so an ImageNet-pretrained CNN can be fine-tuned
on a small audio dataset. MobileNetV4 conv-small was chosen because it is small and fast
enough for real-time CPU inference on a Raspberry Pi 4B.

Class weights in the loss compensate for `Sick` being the minority class. Augmentation
(random gain, noise, time shift, time and frequency masking) is applied during training
only, never during validation or testing.

### Results (held-out test set, 900 clips)

| Metric | Training pipeline (PyTorch + torchaudio) | Deployed pipeline (ONNX + librosa) |
|---|---|---|
| Accuracy | 86.33% | 86.44% |
| Macro Precision | 0.8633 | 0.8638 |
| Macro Recall | 0.8633 | 0.8644 |
| Macro F1 | 0.8620 | 0.8636 |
| MCC | 0.7961 | 0.7972 |

Source `results/pipeline_agreement/Pipeline_Metrics_Comparison.csv`. The two pipelines
disagree on 20 of 900 clips, and PyTorch vs ONNX on the same input agree on all 900
(max logit difference 1.13e-4). See [Dual-pipeline design](#dual-pipeline-design).

### Dual-pipeline design

The project keeps two preprocessing implementations on purpose

- **Training pipeline**, PyTorch + torchaudio, used for training and the reported metrics.
- **Deployment pipeline**, NumPy + librosa, used on the Pi where PyTorch is too heavy.

The Pi only receives the `pi_deploy/` folder, so its audio constants are duplicated in
`pi_monitor.py`. `pi_deploy/verify_preprocessing.py` checks a single clip through both
pipelines, and `scripts/05_evaluate_pi_pipeline.py` runs the whole test set through the
deployed path to measure the gap.

## Quick Start with Docker

The easiest way to reproduce training and evaluation is the prebuilt Docker image. It
contains the code and all dependencies (CUDA 12.1, PyTorch, timm, ONNX Runtime).

```bash
docker pull k1mchir1ce/crdd-train@sha256:78c97d6e745b98c36ed720447683c6734d25c43de05a053608f78f8dd4a53ab3
```

| | |
|---|---|
| Image | [`docker.io/k1mchir1ce/crdd-train`](https://hub.docker.com/r/k1mchir1ce/crdd-train) |
| Digest | `sha256:78c97d6e745b98c36ed720447683c6734d25c43de05a053608f78f8dd4a53ab3` |

Pulling by digest gets the exact image used for the thesis results, even if `latest` is
updated later.

**[→ Full Docker instructions (docker/DOCKER.md)](docker/DOCKER.md)** covers GPU setup,
running each pipeline step with `docker compose`, plain `docker run` on Windows, building
the Raspberry Pi image, and troubleshooting.

Short version, run from the project root

```bash
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/01_prepare_data.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/02_train.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/04_evaluation.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/03_export_onnx.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/05_evaluate_pi_pipeline.py
```

Datasets, checkpoints, and results are mounted from your machine, not baked into the
image.

## Manual Setup (without Docker)

### Training machine

- Python 3.10 or newer
- NVIDIA GPU with CUDA 12.1 (CPU training works but is slow)

```bash
pip install -r requirements.txt
```

`requirements.lock.txt` lists exact pinned versions for full reproducibility.

### Raspberry Pi 4B

- Raspberry Pi OS 64-bit
- USB microphone
- 16x2 character LCD over I2C with a PCF8574 backpack (optional, the monitor runs without it)
- `libportaudio2` system package, required by `sounddevice`

```bash
sudo apt install libportaudio2
pip install -r pi_deploy/pi_requirements.txt
```

PyTorch is not installed on the Pi. Inference runs entirely through ONNX Runtime.

## Usage

Run the scripts in this order from the project root.

**1. Organize the dataset**

Place `.wav` files into `data/raw/Healthy/`, `data/raw/Sick/`, and `data/raw/None/`.
Raw audio is not included in this repository.

**2. Generate splits**

```bash
python scripts/01_prepare_data.py
```

Creates `data/splits/train.txt`, `val.txt`, and `test.txt` (70/15/15, stratified per class).

**3. Train**

```bash
python scripts/02_train.py
```

Saves `models/saved/crdd_model_v1.pth` (best validation accuracy) and
`results/training_history.json`.

**4. Evaluate the training pipeline**

```bash
python scripts/04_evaluation.py
```

**5. Export to ONNX**

```bash
python scripts/03_export_onnx.py
```

Verifies that PyTorch and ONNX outputs match within 1e-4 before you deploy.

**6. Evaluate the deployed pipeline**

```bash
python scripts/05_evaluate_pi_pipeline.py
```

Writes deployed-path metrics and `Pipeline_Comparison.csv` to `results/librosa_results/`.

**7. Deploy to the Raspberry Pi**

Copy `models/saved/crdd_model_v1.onnx` into `pi_deploy/`, copy the `pi_deploy/` folder to
the Pi (e.g. `/home/pi/pi_deploy`), then run

```bash
python3 pi_deploy/pi_monitor.py
```

**8. Run as a service (optional)**

Starts the monitor at boot and restarts it if it crashes

```bash
sudo cp pi_deploy/poultry-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now poultry-monitor.service
```

Check status and logs

```bash
systemctl status poultry-monitor.service
journalctl -u poultry-monitor.service -f
```

Run either the systemd service or the Pi Docker container, not both, or they will fight
over the microphone.

## Directory Structure

```
crdd-hau-soc-model/
├── configs/
│   └── config.py                       Central configuration
├── src/
│   ├── data/
│   │   ├── audio_dataset.py            PyTorch Dataset: wav → spectrogram tensor
│   │   └── split_dataset.py            Train/val/test split generation
│   ├── models/
│   │   └── builder.py                  MobileNetV4 model construction
│   └── training/
│       └── trainer.py                  Training and validation loops
├── scripts/
│   ├── 01_prepare_data.py              Build data splits
│   ├── 02_train.py                     Train the model
│   ├── 03_export_onnx.py               Export to ONNX
│   ├── 04_evaluation.py                Evaluate the training pipeline
│   └── 05_evaluate_pi_pipeline.py      Evaluate the deployed pipeline
├── pi_deploy/
│   ├── pi_monitor.py                   Live inference on the Raspberry Pi
│   ├── verify_preprocessing.py         Training vs Pi preprocessing check
│   ├── poultry-monitor.service         systemd unit for continuous operation
│   └── pi_requirements.txt             Pi-only dependencies
├── docker/
│   ├── Dockerfile                      Training image (CUDA 12.1)
│   ├── Dockerfile.pi                   Raspberry Pi image (arm64)
│   ├── docker-compose.yml              Compose file for the training image
│   └── DOCKER.md                       Docker instructions
├── models/saved/
│   ├── crdd_model_v1.pth               Best checkpoint
│   └── crdd_model_v1.onnx              Exported ONNX model
├── results/                            Metrics, tables, and figures
│   ├── librosa_results/                Deployed-pipeline outputs
│   └── pipeline_agreement/             Per-clip training vs deployed comparison
├── data/                               Not committed (raw audio + splits)
├── .dockerignore
├── requirements.txt
├── requirements.lock.txt
└── README.md
```

## File Reference

| File | Description |
|---|---|
| `configs/config.py` | Single source of truth for paths, audio parameters, class names, and hyperparameters. |
| `src/data/split_dataset.py` | Stratified 70/15/15 split per class with a fixed seed. |
| `src/data/audio_dataset.py` | `ChickenAudioDataset`. Mono mixdown, resampling, pad/crop, augmentation, log-mel, SpecAugment, standardization, resize. |
| `src/models/builder.py` | MobileNetV4 conv-small via `timm` with a 3-class head and dropout. |
| `src/training/trainer.py` | `train_one_epoch()` and `validate()`, with gradient clipping. |
| `scripts/02_train.py` | Class-weighted training with AdamW, cosine annealing, per-epoch metrics, early stopping. |
| `scripts/03_export_onnx.py` | Exports to ONNX (opset 17) and verifies the output matches PyTorch. |
| `scripts/04_evaluation.py` | Test-set metrics, confusion matrix, per-class chart, per-epoch curves. Outputs prefixed `Model_A_Raw_`. |
| `scripts/05_evaluate_pi_pipeline.py` | Same test set through librosa + ONNX. Outputs prefixed `Model_A_Pi_`. Runs on the PC, not the Pi. |
| `pi_deploy/pi_monitor.py` | Live loop on the Pi. Record, preprocess, infer, LCD, save, Supabase upload, heartbeat, storage cleanup. |
| `pi_deploy/verify_preprocessing.py` | Runs one `.wav` through both pipelines and compares tensors and predictions. |
| `pi_deploy/poultry-monitor.service` | systemd unit that keeps the monitor running unattended. |

## Configuration

Audio and training parameters live in `configs/config.py`

| Parameter | Value |
|---|---|
| Sample rate | 16000 Hz |
| Clip duration | 5 seconds |
| FFT size / hop length | 1024 / 512 |
| Mel bands | 128 |
| Input image size | 224 × 224 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Max epochs / early stop patience | 40 / 8 |
| Random seed | 42 |

Pi-side behavior (confidence threshold, LCD hold time, storage limits, timezone, Supabase
settings) is set at the top of `pi_deploy/pi_monitor.py`.

> **Changing audio parameters requires three coordinated updates.** Any change to sample
> rate, duration, FFT size, hop length, mel bands, or image size must be made in both
> `configs/config.py` and `pi_deploy/pi_monitor.py`, and the ONNX model must be
> re-exported. A mismatch silently lowers real-world accuracy and will not show up in any
> training metric.

## Troubleshooting

**`ERROR: no working sample rate found for the microphone`**
Run `python3 -m sounddevice` to list devices, then set `MIC_DEVICE` to the right index at
the top of `pi_monitor.py`.

**PortAudio error -9997 (invalid sample rate)**
The mic cannot capture at 16 kHz directly. The monitor already falls back to the mic's
native rate and resamples in software. Confirm `libportaudio2` is installed.

**LCD shows nothing**
Find the I2C address with `i2cdetect -y 1` and update `LCD_ADDRESS` (usually `0x27` or
`0x3F`). The monitor keeps running without the LCD.

**`ERROR: model not found`**
Copy `crdd_model_v1.onnx` from `models/saved/` into `pi_deploy/`.

**PyTorch and ONNX outputs differ by more than 1e-4**
Do not deploy. Match the package versions in `requirements.lock.txt` and re-run
`03_export_onnx.py`.

**Uploads fail with a Supabase error**
The clip stays local and the loop continues. Check the network, the bucket name, and the
Row Level Security policies.

**Dashboard shows the device as offline**
The Pi updates its heartbeat every cycle. Check that the Pi has internet and that the
monitor is running (`systemctl status poultry-monitor.service` or `docker logs -f crdd`).

For Docker-specific errors, see the [Docker troubleshooting section](docker/DOCKER.md#troubleshooting).

## Acknowledgments

Training data includes the [SmartEars dataset](https://data.mendeley.com/) (Mendeley Data)
alongside locally captured field recordings.

Developed as an undergraduate thesis at Holy Angel University, School of Computing.
