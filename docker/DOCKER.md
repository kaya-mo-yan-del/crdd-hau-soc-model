# Docker Setup — Chicken Respiratory Distress Detector

Containerized environment for reproducing the training, evaluation and deployment
pipeline. All Docker files live in this `docker/` folder; commands are run from the
project root unless noted otherwise.

| Image | Host | Contents |
|---|---|---|
| `crdd-train` (`Dockerfile`) | Training PC with an NVIDIA GPU | PyTorch, torchaudio, timm, ONNX Runtime |
| `crdd-pi` (`Dockerfile.pi`) | Raspberry Pi 4B (arm64) | ONNX Runtime, librosa, sounddevice, RPLCD |

The images contain code and dependencies only. Datasets, checkpoints and results are
bind-mounted from the host, so nothing large or private is baked into an image.

## Prerequisites

Training machine

- Docker Desktop (Windows/Mac) or Docker Engine (Linux), version 24 or newer
- NVIDIA driver supporting CUDA 12.1
- On Windows: WSL2 enabled, virtualization enabled in BIOS/UEFI

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If this prints the GPU table, GPU passthrough is working.

Raspberry Pi

- Raspberry Pi OS 64-bit
- Docker Engine installed through `curl -fsSL https://get.docker.com | sh`

## Directory layout

```
crdd-hau-soc-model/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.pi
│   ├── docker-compose.yml
│   └── DOCKER.md          <- this file
├── .dockerignore           <- must stay at project root
├── requirements.txt
├── requirements.lock.txt
├── configs/
├── src/
├── scripts/
├── pi_deploy/
├── data/                   mounted, not copied
├── models/                 mounted, not copied
└── results/                mounted, not copied
```

`.dockerignore` stays at the project root because Docker looks for it next to the
build context, not next to the Dockerfile. `Dockerfile` and `Dockerfile.pi` reference
paths like `src/` and `scripts/` relative to that context (the project root), not to
their own location in `docker/`.

## Prebuilt image

```powershell
docker pull k1mchir1ce/crdd-train:latest
```

## Training pipeline — run in this order

From the project root, using `docker compose` with `-f` to point at the file in
`docker/`:

```powershell
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/01_prepare_data.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/02_train.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/04_evaluation.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/03_export_onnx.py
docker compose -f docker/docker-compose.yml run --rm crdd python scripts/05_evaluate_pi_pipeline.py
```

`01_prepare_data.py` must run first — it generates `data/splits/train.txt`,
`val.txt`, `test.txt` from `data/raw/<class>/*.wav`. Every later script depends on
these files existing.

Outputs appear directly in the host `models/saved/` and `results/` folders because
those paths are mounted, not copied into the image.

Interactive shell for debugging

```powershell
docker compose -f docker/docker-compose.yml run --rm crdd bash
```

Confirm the GPU is visible from inside the container

```powershell
docker compose -f docker/docker-compose.yml run --rm crdd python -c "import torch; print(torch.cuda.is_available())"
```

### Without compose (plain `docker run` on Windows)

```powershell
docker run --rm --gpus all `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\models:/app/models" `
  k1mchir1ce/crdd-train:latest python scripts/01_prepare_data.py
```

Run from the project root so `${PWD}` resolves to the right folder.

## Raspberry Pi image

Build on the Pi itself, from the project root:

```bash
docker build -f docker/Dockerfile.pi -t crdd-pi:1.0 .
```

Copy `models/saved/crdd_model_v1.onnx` into `pi_deploy/` before building, or mount it
at runtime as shown below.

Run

```bash
docker run -d --name crdd \
  --restart unless-stopped \
  --device /dev/snd \
  --device /dev/i2c-1 \
  --group-add audio \
  -v $(pwd)/pi_deploy/recordings:/app/recordings \
  -v $(pwd)/pi_deploy/crdd_model_v1.onnx:/app/crdd_model_v1.onnx:ro \
  crdd-pi:1.0
```

| Flag | Reason |
|---|---|
| `--device /dev/snd` | USB microphone access for `sounddevice` |
| `--device /dev/i2c-1` | I2C bus for the 16x2 LCD |
| `--group-add audio` | Permission to open the capture device |
| `--restart unless-stopped` | Replaces the `crdd.service` restart behavior |
| `-v .../recordings` | Saved clips survive container rebuilds |

Logs

```bash
docker logs -f crdd
```

The container and `crdd.service` both manage the same script. Run one or the other,
never both, or two processes will compete for the microphone:

```bash
sudo systemctl disable --now crdd.service
```

## Reproducibility notes

**Pinned dependencies.** Generate a locked version list once training works:

```powershell
docker compose -f docker/docker-compose.yml run --rm crdd pip freeze > requirements.lock.txt
```

Commit the lock file at the project root, alongside `requirements.txt`.

**Remaining nondeterminism.** `config.SEED = 42` fixes Python, NumPy and PyTorch
seeds, so splits and augmentation are reproducible. cuDNN kernel selection is still
nondeterministic, so two runs of `02_train.py` on the same data can differ slightly in
final accuracy. Worth stating in the methodology rather than claiming bit-exact
reproducibility.

**Image digest.** Cite the exact image in the manuscript, not `latest`:

```powershell
docker images --digests k1mchir1ce/crdd-train
```

## Troubleshooting

**`Virtualization Enabled In Firmware: No`**
Enable virtualization (Intel VT-x / AMD SVM) in BIOS/UEFI before Docker Desktop can
start on Windows.

**`request returned 500 Internal Server Error ... dockerDesktopLinuxEngine/_ping`**
Docker Desktop engine isn't running or crashed. Restart it, or `wsl --shutdown` then
reopen Docker Desktop.

**`docker: invalid reference format`**
Usually caused by `$(pwd)` (bash syntax) used in PowerShell. Use `${PWD}` and quote
paths that contain spaces, or use `docker compose run` instead.

**`unable to prepare context: path "docker" not found` or COPY errors during build**
The build was run from inside `docker/` instead of the project root. `context: ..` in
`docker-compose.yml` expects the command to resolve relative to the compose file's own
location, which it does automatically — but a plain `docker build` command needs the
project root as its final `.` argument, run from the project root.

**`FileNotFoundError: data/splits/train.txt`**
`01_prepare_data.py` wasn't run yet, or `data/raw/<class>/` has no `.wav` files on
the host.

**`could not select device driver "nvidia"`**
The NVIDIA Container Toolkit is not installed, or Docker wasn't restarted after
installing it.

**Bus error or DataLoader worker crash**
Shared memory too small. `shm_size: "2gb"` in `docker-compose.yml` handles this. With
plain `docker run`, add `--shm-size=2g`.

**`ModuleNotFoundError: No module named 'sounddevice'` when running script 05**
`05_evaluate_pi_pipeline.py` imports `pi_deploy/pi_monitor.py`, which imports
`sounddevice` and `supabase` at module level. Both must be in `requirements.txt`.

**Files written by the container are owned by root (Linux host)**
Add `user: "${UID}:${GID}"` to the compose service, or `sudo chown -R $USER:$USER
results models` afterwards. Not an issue on Windows/WSL2.

**PortAudio finds no input device inside the Pi container**
Confirm `--device /dev/snd` and `--group-add audio` are present, then check
`docker exec crdd python -m sounddevice`.
