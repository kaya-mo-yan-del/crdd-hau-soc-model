import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import soundfile as sf

from configs import config
from src.data.audio_dataset import ChickenAudioDataset
from pi_deploy.pi_monitor import preprocess as pi_preprocess


def torch_preprocess(wav_path: str) -> np.ndarray:
    """Replicates the validation-time (is_training=False) pipeline."""
    ds = ChickenAudioDataset.__new__(ChickenAudioDataset)
    ds.is_training = False
    ds.sample_rate = config.SAMPLE_RATE
    ds.num_samples = config.NUM_SAMPLES
    ds._resamplers = {}
    import torchaudio.transforms as T
    from torchvision import transforms
    ds.mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
    ds.amplitude_to_db = T.AmplitudeToDB(top_db=80)
    ds.resize = transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                                  antialias=True)

    audio, sr = sf.read(wav_path, dtype="float32")
    wf = torch.from_numpy(audio)
    wf = wf.unsqueeze(0) if wf.ndim == 1 else wf.transpose(0, 1)
    wf = ds._to_mono(wf)
    wf = ds._resample(wf, sr)
    wf = ds._fix_length(wf)
    mel = ds.mel_spectrogram(wf)
    mel_db = ds.amplitude_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db.repeat(3, 1, 1)
    mel_db = ds.resize(mel_db)
    return mel_db.unsqueeze(0).numpy()

def main():
    if len(sys.argv) != 2:
        print("Usage: python pi_deploy/verify_preprocessing.py <clip.wav>")
        sys.exit(1)
    wav_path = sys.argv[1]

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

    a = torch_preprocess(wav_path)      # training pipeline
    b = pi_preprocess(audio)            # Pi pipeline

    diff = np.abs(a - b)
    print(f"Input tensors  — max diff: {diff.max():.5f}   "
          f"mean diff: {diff.mean():.6f}")

    onnx_path = config.MODEL_DIR / "crdd_model_v1.onnx"
    if onnx_path.exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        la = sess.run(["logits"], {"input": a.astype(np.float32)})[0][0]
        lb = sess.run(["logits"], {"input": b.astype(np.float32)})[0][0]
        pa, pb = int(np.argmax(la)), int(np.argmax(lb))
        print(f"Prediction     — training pipeline: {config.CLASSES[pa]}   "
              f"Pi pipeline: {config.CLASSES[pb]}   "
              f"{'MATCH ' if pa == pb else 'MISMATCH '}")
        print(f"Logit max diff — {np.abs(la - lb).max():.5f}")
    else:
        print("(crdd_model_v1.onnx not found — run 03_export_onnx.py to also "
              "compare predictions.)")



if __name__ == "__main__":
    main()
