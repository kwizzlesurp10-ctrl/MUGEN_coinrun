from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from musicgen.codec import AudioCodec
from musicgen.types import CodecConfig


class MusicDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        manifest_path: str,
        codec: AudioCodec,
        sample_rate: int = 24000,
        max_duration_seconds: float = 10.0,
    ) -> None:
        self.codec = codec
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_seconds)
        self.samples: list[tuple[str, str]] = []
        self._load_manifest(manifest_path)

    def _load_manifest(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    self.samples.append((parts[0].strip(), parts[1].strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        audio_path, text = self.samples[idx]
        import soundfile as sf

        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != self.sample_rate:
            import librosa

            waveform = librosa.resample(
                waveform.astype(float), orig_sr=sr, target_sr=self.sample_rate
            )
        waveform = torch.from_numpy(waveform).float()
        if waveform.shape[0] > self.max_samples:
            start = torch.randint(0, waveform.shape[0] - self.max_samples, (1,)).item()
            waveform = waveform[start : start + self.max_samples]
        waveform = waveform.unsqueeze(0)
        device = next(self.codec.parameters()).device
        waveform = waveform.to(device)
        with torch.no_grad():
            _, indices = self.codec.encode(waveform)
        return indices.squeeze(0).cpu(), waveform.squeeze(0).cpu(), text
