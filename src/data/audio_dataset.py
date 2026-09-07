import random
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms
import soundfile as sf


class ChickenAudioDataset(Dataset):
    def __init__(self, split_file, sample_rate=16000, n_mels=128,
                 image_size=224, duration=5, n_fft=1024, hop_length=512,
                 is_training=False):
        self.data = []
        self.is_training = is_training
        with open(split_file, "r") as f:
            for line in f:
                path, label = line.strip().rsplit(",", 1)
                self.data.append((path, int(label)))

        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)
        self.resize = transforms.Resize((image_size, image_size), antialias=True)

        # SpecAugment (training only)
        self.time_masking = T.TimeMasking(time_mask_param=30)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)

        # Cache resamplers per source sample rate
        self._resamplers = {}

    def __len__(self):
        return len(self.data)

    # ---------- waveform helpers ----------

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """(channels, samples) -> (1, samples)"""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = T.Resample(sr, self.sample_rate)
        return self._resamplers[sr](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads short clips with zeros; crops long clips (random crop for
        training, center crop for validation/test)."""
        length = waveform.shape[1]
        if length < self.num_samples:
            pad = self.num_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif length > self.num_samples:
            if self.is_training:
                start = random.randint(0, length - self.num_samples)
            else:
                start = (length - self.num_samples) // 2
            waveform = waveform[:, start:start + self.num_samples]
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random gain
        if random.random() < 0.5:
            gain_db = random.uniform(-6.0, 6.0)
            waveform = waveform * (10.0 ** (gain_db / 20.0))
        # Additive gaussian noise
        if random.random() < 0.3:
            snr_scale = random.uniform(0.001, 0.01)
            waveform = waveform + snr_scale * torch.randn_like(waveform)
        # Random circular time shift
        if random.random() < 0.5:
            shift = random.randint(-self.sample_rate // 2, self.sample_rate // 2)
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        return waveform

    # ---------- main ----------

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]

        audio_array, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(audio_array)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)          # (1, samples)
        else:
            waveform = waveform.transpose(0, 1)       # (channels, samples)

        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)

        if self.is_training:
            waveform = self._augment_waveform(waveform)

        mel = self.mel_spectrogram(waveform)
        mel_db = self.amplitude_to_db(mel)

        if self.is_training:
            if random.random() > 0.5:
                mel_db = self.time_masking(mel_db)
            if random.random() > 0.5:
                mel_db = self.freq_masking(mel_db)

        # Per-sample standardization
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        mel_db = mel_db.repeat(3, 1, 1)               # 1ch -> 3ch
        mel_db = self.resize(mel_db)

        return mel_db, label
