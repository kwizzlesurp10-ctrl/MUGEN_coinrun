#!/usr/bin/env python3
"""Train the audio codec (Residual VQ-VAE) from scratch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from musicgen.codec import AudioCodec
from musicgen.types import CodecConfig


def train_codec(
    manifest_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    sample_rate: int = 24000,
    preset: str = "small",
) -> None:
    from musicgen.config_loader import load_preset

    config, _ = load_preset(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = AudioCodec(config).to(device)

    import soundfile as sf
    import librosa
    from torch.utils.data import Dataset

    class RawAudioDataset(Dataset):
        def __init__(self, manifest_path: str, sr: int, max_sec: float):
            self.samples: list[str] = []
            with open(manifest_path) as f:
                for line in f:
                    path = line.strip().split("|", 1)[0].strip()
                    if path:
                        self.samples.append(path)
            self.sr = sr
            self.max_samples = int(sr * max_sec)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> torch.Tensor:
            wav, sr = sf.read(self.samples[idx])
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.sr:
                wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=self.sr)
            wav = torch.from_numpy(wav).float()
            if wav.shape[0] > self.max_samples:
                start = torch.randint(0, wav.shape[0] - self.max_samples, (1,)).item()
                wav = wav[start : start + self.max_samples]
            return wav

    dataset = RawAudioDataset(manifest_path, config.sample_rate, 5.0)

    def collate(batch: list) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            [b.unsqueeze(0) for b in batch], batch_first=True
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    opt = torch.optim.AdamW(codec.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        codec.train()
        total_loss = 0.0
        for waveforms in loader:
            waveforms = waveforms.to(device)
            if waveforms.shape[-1] < 1024:
                continue
            reconstructed, quantized, indices, commitment = codec(waveforms)
            recon_loss = F.mse_loss(reconstructed, waveforms)
            loss = recon_loss + 0.25 * commitment
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss / len(loader):.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(codec.state_dict(), output_path / f"codec_epoch{epoch+1}.pt")

    torch.save(codec.state_dict(), output_path / "codec_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest (audio_path|text)")
    parser.add_argument("--output", default="checkpoints/codec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "medium", "large"])
    args = parser.parse_args()
    train_codec(
        args.manifest,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        preset=args.preset,
    )
