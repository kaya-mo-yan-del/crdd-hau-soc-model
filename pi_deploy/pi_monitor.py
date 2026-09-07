import os
import re
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import librosa                    # mel spectrogram (matched to torchaudio)
import soundfile as sf            # save .wav recordings
import sounddevice as sd          # microphone capture
import onnxruntime as ort         # run the exported model
from PIL import Image             # bilinear resize (matches torchvision antialias)
from supabase import create_client  # cloud sync (Storage + Postgres)

# ======================================================================
# CONFIGURATION
# ======================================================================
BASE_DIR = Path(__file__).resolve().parent

# Model file: preferred name first, otherwise any .onnx file in this folder
MODEL_NAME = "crdd_model_v1.onnx"
_candidates = ([BASE_DIR / MODEL_NAME]
               + sorted(BASE_DIR.glob("*.onnx")))
MODEL_PATH = next((p for p in _candidates if p.exists()),
                  BASE_DIR / MODEL_NAME)
RECORDINGS_DIR = BASE_DIR / "recordings"    # saved clips (on the SD card)

# Fixed timezone for all timestamps, independent of the Pi's OS clock
LOCAL_TZ = ZoneInfo("Asia/Manila")

# --- Audio / preprocessing (must match configs/config.py used in training) ---
SAMPLE_RATE = 16000
AUDIO_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
IMAGE_SIZE = 224

# --- Classes (same order as training: index 0,1,2) ---
CLASSES = ["Healthy", "Sick", "None"]
LCD_LABELS = {"Healthy": "HEALTHY", "Sick": "SICK!", "None": "NOISE"}  # <=7 chars
SAVE_CLASSES = {"Healthy", "Sick", "None"}  # which detections get saved locally

# --- Behavior ---
MIN_CONF_TO_SAVE = 0.70      # confidence threshold for saving/uploading a clip
MIN_DISPLAY_SECONDS = 5      # LCD holds a result at least this long (anti-flicker)
MIN_FREE_MB = 500            # local cleanup threshold
MIC_DEVICE = None            # None = system default input; or a device index

# --- LCD (16x2 over I2C, PCF8574 backpack) ---
LCD_ENABLED = True
LCD_ADDRESS = 0x27           # some boards use 0x3F (check: i2cdetect -y 1)
LCD_COLS, LCD_ROWS = 16, 2

# --- Supabase (cloud sync for the web dashboard) ---
# SUPABASE_KEY is the publishable (anon) key — safe to ship on-device;
# access is controlled by Row Level Security, not by keeping this secret.
SUPABASE_URL = "https://ppkatqpqvqbbxwgdmczm.supabase.co"
SUPABASE_KEY = "sb_publishable_9MuP7gohPyFHrkXMuF_Ing_qPtQH6h9"
SUPABASE_BUCKET = "audio-clips"
SUPABASE_UPLOAD_CLASSES = {"Sick"}   # only Sick clips reach the dashboard
SUPABASE_MAX_STORAGE_MB = 900   # cleanup threshold, near the 1 GB free-tier cap
DEVICE_STATUS_ID = 1   # single-row heartbeat read by the dashboard

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ======================================================================


# ----------------------------------------------------------------------
# Preprocessing — numpy re-implementation of the training pipeline,
# verified against torchaudio by pi_deploy/verify_preprocessing.py.
# ----------------------------------------------------------------------
_MEL_BASIS = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
    htk=True, norm=None,          # matches torchaudio's mel scale
)


def preprocess(waveform: np.ndarray) -> np.ndarray:
    """(samples,) float32 -> (1, 3, 224, 224) float32, matching training."""
    # 1. Fixed length: pad with zeros or center-crop
    if len(waveform) < NUM_SAMPLES:
        waveform = np.pad(waveform, (0, NUM_SAMPLES - len(waveform)))
    elif len(waveform) > NUM_SAMPLES:
        start = (len(waveform) - NUM_SAMPLES) // 2
        waveform = waveform[start:start + NUM_SAMPLES]

    # 2. Power spectrogram (hann window, centered, reflect padding)
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        window="hann", center=True, pad_mode="reflect")
    power_spec = np.abs(stft) ** 2

    # 3. Mel projection
    mel = _MEL_BASIS @ power_spec                     # (128, T)

    # 4. AmplitudeToDB(top_db=80)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)

    # 5. Per-sample standardization
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # 6. Resize to 224x224 (bilinear, matches torchvision antialias)
    img = Image.fromarray(mel_db.astype(np.float32), mode="F")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    mel_224 = np.asarray(img, dtype=np.float32)

    # 7. 1 channel -> 3 channels, add batch dimension
    x = np.stack([mel_224, mel_224, mel_224])          # (3, 224, 224)
    return x[np.newaxis, :]                            # (1, 3, 224, 224)


# ----------------------------------------------------------------------
# Microphone capture-rate detection
# ----------------------------------------------------------------------
CANDIDATE_RATES = [16000, 48000, 44100, 32000, 22050, 8000]


def find_capture_rate(device):
    """Returns the first sample rate the microphone supports."""
    try:
        info = sd.query_devices(device, "input")
        default_rate = int(info["default_samplerate"])
        rates = [default_rate] + [r for r in CANDIDATE_RATES if r != default_rate]
        print(f"Microphone: {info['name']} (default {default_rate} Hz)")
    except Exception:
        rates = CANDIDATE_RATES

    for rate in rates:
        try:
            sd.check_input_settings(device=device, samplerate=rate,
                                    channels=1, dtype="float32")
            return rate
        except Exception:
            continue

    print("ERROR: no working sample rate found for the microphone.\n"
          "Available devices:")
    print(sd.query_devices())
    print("\nPick your USB mic's index from the list above and set "
          "MIC_DEVICE = <index> at the top of this file.")
    sys.exit(1)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def now_local() -> datetime:
    """Current time in Philippine time, regardless of the Pi's system tz."""
    return datetime.now(LOCAL_TZ)


# ----------------------------------------------------------------------
# LCD helper — falls back to terminal-only output if no LCD is wired
# ----------------------------------------------------------------------
class LcdDisplay:
    def __init__(self):
        self.lcd = None
        self.last_change = 0.0
        self.current_label = None
        if not LCD_ENABLED:
            return
        try:
            from RPLCD.i2c import CharLCD
            self.lcd = CharLCD(i2c_expander="PCF8574", address=LCD_ADDRESS,
                               port=1, cols=LCD_COLS, rows=LCD_ROWS,
                               auto_linebreaks=False)
            self.lcd.clear()
            self._write("PoultryMonitor", "starting...")
        except Exception as e:
            print(f"[LCD] Not available ({e}) — continuing without LCD.")
            self.lcd = None

    def _write(self, line1: str, line2: str):
        """Writes two lines, each padded/truncated to 16 characters."""
        if self.lcd is None:
            return
        try:
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1[:LCD_COLS].ljust(LCD_COLS))
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(line2[:LCD_COLS].ljust(LCD_COLS))
        except Exception as e:
            print(f"[LCD] Write failed ({e}) — disabling LCD.")
            self.lcd = None

    def show_result(self, label: str, confidence: float, when: datetime):
        """Shows the top category, holding the previous result for at
        least MIN_DISPLAY_SECONDS to avoid flicker."""
        now = time.monotonic()
        if (self.current_label is not None
                and label != self.current_label
                and (now - self.last_change) < MIN_DISPLAY_SECONDS):
            return  # too soon to change

        # Line 1: label left, confidence right  -> "SICK!      87.4%"
        short = LCD_LABELS.get(label, label.upper())[:9]
        conf_txt = f"{confidence * 100:4.1f}%"                # 5 chars
        line1 = short.ljust(LCD_COLS - len(conf_txt)) + conf_txt
        # Line 2: date + time (Philippine time) -> "07/05  14:30:22"
        line2 = when.strftime("%m/%d  %H:%M:%S")
        self._write(line1, line2)

        if label != self.current_label:
            self.current_label = label
            self.last_change = now


# ----------------------------------------------------------------------
# Storage: saving detections + cleanup
# ----------------------------------------------------------------------
# Filename format (cleanup parses it):
#   Sick_conf087.4_2026-07-05_14-30-22-12.wav
FNAME_RE = re.compile(r"^(?P<label>\w+)_conf(?P<conf>\d{3}\.\d)_"
                      r"(?P<stamp>[\d\-_]+)\.wav$")


def save_detection(waveform: np.ndarray, label: str, confidence: float,
                   when: datetime) -> Path:
    folder = RECORDINGS_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-4]  # keep hundredths
    fname = f"{label}_conf{confidence * 100:05.1f}_{stamp}.wav"
    path = folder / fname
    sf.write(path, waveform, SAMPLE_RATE)
    return path


def free_mb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e6


def cleanup_category(label: str):
    """Deletes recordings from this category, lowest confidence first,
    oldest first among equals, until free space clears the threshold."""
    folder = RECORDINGS_DIR / label
    if not folder.exists():
        return
    candidates = []
    for f in folder.glob("*.wav"):
        m = FNAME_RE.match(f.name)
        conf = float(m.group("conf")) if m else 0.0
        candidates.append((conf, f.stat().st_mtime, f))
    candidates.sort(key=lambda t: (t[0], t[1]))

    while candidates and free_mb(RECORDINGS_DIR) < MIN_FREE_MB:
        conf, _, victim = candidates.pop(0)
        try:
            victim.unlink()
            print(f"[CLEANUP] Deleted {victim.name} (conf {conf:.1f}%) "
                  f"to free space.")
        except OSError as e:
            print(f"[CLEANUP] Could not delete {victim.name}: {e}")


# ----------------------------------------------------------------------
# Supabase sync
# ----------------------------------------------------------------------
def send_heartbeat(when: datetime):
    """Updates the device_status row so the dashboard shows the Pi as
    online/offline. Called once per cycle regardless of detection."""
    try:
        supabase.table("device_status").update(
            {"last_seen": when.isoformat()}
        ).eq("id", DEVICE_STATUS_ID).execute()
    except Exception as e:
        print(f"[SUPABASE] Heartbeat failed ({e}).")


def upload_to_supabase(local_path: Path, label: str, confidence: float,
                       when: datetime):
    if label not in SUPABASE_UPLOAD_CLASSES:
        return

    remote_path = f"{label}/{local_path.name}"
    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=remote_path,
                file=f.read(),
                file_options={"content-type": "audio/wav"},
            )
        supabase.table("detections").insert({
            "label": label,
            "confidence": confidence,
            "audio_path": remote_path,
            "detected_at": when.isoformat(),
        }).execute()
        print(f"[SUPABASE] Uploaded -> {remote_path}")
        cleanup_supabase_storage()
    except Exception as e:
        print(f"[SUPABASE] Upload failed ({e}) — clip stays local only.")


def _list_supabase_objects() -> list[dict]:
    objects = []
    for label in SUPABASE_UPLOAD_CLASSES:
        try:
            entries = supabase.storage.from_(SUPABASE_BUCKET).list(label)
        except Exception as e:
            print(f"[SUPABASE] Could not list '{label}/': {e}")
            continue
        for entry in entries or []:
            name = entry.get("name")
            size = (entry.get("metadata") or {}).get("size")
            if not name or not size:
                continue
            m = FNAME_RE.match(name)
            conf = float(m.group("conf")) if m else 0.0
            objects.append({
                "path": f"{label}/{name}",
                "size": size,
                "conf": conf,
                "created_at": entry.get("created_at", ""),
            })
    return objects


def cleanup_supabase_storage():
    objects = _list_supabase_objects()
    total_mb = sum(o["size"] for o in objects) / 1e6
    if total_mb < SUPABASE_MAX_STORAGE_MB:
        return

    objects.sort(key=lambda o: (o["conf"], o["created_at"]))

    while objects and total_mb >= SUPABASE_MAX_STORAGE_MB:
        victim = objects.pop(0)
        try:
            supabase.storage.from_(SUPABASE_BUCKET).remove([victim["path"]])
            supabase.table("detections").delete() \
                .eq("audio_path", victim["path"]).execute()
            total_mb -= victim["size"] / 1e6
            print(f"[SUPABASE CLEANUP] Deleted {victim['path']} "
                  f"(conf {victim['conf']:.1f}%) to free cloud space.")
        except Exception as e:
            print(f"[SUPABASE CLEANUP] Could not delete {victim['path']}: {e}")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    print("=" * 62)
    print(" PoultryMonitor — Live Respiratory Distress Detection (Pi 4B)")
    print("=" * 62)

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}\n"
              f"Copy best_model.onnx into the pi_deploy folder first.")
        sys.exit(1)

    session = ort.InferenceSession(str(MODEL_PATH),
                                   providers=["CPUExecutionProvider"])
    print(f"Model loaded: {MODEL_PATH.name}")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    lcd = LcdDisplay()

    capture_rate = find_capture_rate(MIC_DEVICE)
    capture_samples = capture_rate * AUDIO_SECONDS
    if capture_rate == SAMPLE_RATE:
        print(f"Capturing directly at {SAMPLE_RATE} Hz.")
    else:
        print(f"Capturing at {capture_rate} Hz (mic native), "
              f"resampling to {SAMPLE_RATE} Hz in software.")

    print(f"Timezone: {LOCAL_TZ} (Philippine time)")
    print(f"Listening in {AUDIO_SECONDS}-second windows. Ctrl+C to stop.\n")

    while True:
        try:
            # 1. Record audio (mono, mic-native rate)
            recording = sd.rec(capture_samples, samplerate=capture_rate,
                               channels=1, dtype="float32", device=MIC_DEVICE)
            sd.wait()
            waveform = recording[:, 0]
            when = now_local()

            if capture_rate != SAMPLE_RATE:
                waveform = librosa.resample(
                    waveform, orig_sr=capture_rate, target_sr=SAMPLE_RATE)

            # 2 + 3. Preprocess and run the model
            x = preprocess(waveform)
            logits = session.run(["logits"], {"input": x})[0][0]
            probs = softmax(logits)
            top_idx = int(np.argmax(probs))
            top_label = CLASSES[top_idx]
            top_conf = float(probs[top_idx])

            # 4. Terminal output
            stamp = when.strftime("%Y-%m-%d %H:%M:%S")
            per_class = " | ".join(
                f"{name}: {probs[i] * 100:5.1f}%"
                for i, name in enumerate(CLASSES))
            print(f"[{stamp}]  {per_class}   ->  {top_label.upper()}")

            send_heartbeat(when)

            # 5. LCD update
            lcd.show_result(top_label, top_conf, when)

            # 6 + 7. Save + cleanup local storage
            if top_label in SAVE_CLASSES and top_conf >= MIN_CONF_TO_SAVE:
                if free_mb(RECORDINGS_DIR) < MIN_FREE_MB:
                    cleanup_category(top_label)
                path = save_detection(waveform, top_label, top_conf, when)
                print(f"          saved -> {path.relative_to(BASE_DIR)}")

                # 8. Sync Sick clips to Supabase
                upload_to_supabase(path, top_label, top_conf, when)

        except KeyboardInterrupt:
            print("\nStopping. Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e} — retrying in 3 s...")
            time.sleep(3)


if __name__ == "__main__":
    main()
