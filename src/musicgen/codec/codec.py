from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from musicgen.types import CodecConfig
from musicgen.codec.vq import ResidualVectorQuantizer


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=stride, padding=3)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=8, stride=stride, padding=3)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class AudioCodec(nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            EncoderBlock(1, 32, 2),
            nn.GELU(),
            EncoderBlock(32, 64, 2),
            nn.GELU(),
            EncoderBlock(64, 128, 2),
            nn.GELU(),
            nn.Conv1d(128, config.latent_dim, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(config.latent_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            DecoderBlock(128, 64, 2),
            nn.GELU(),
            DecoderBlock(64, 32, 2),
            nn.GELU(),
            DecoderBlock(32, 1, 2),
        )

        self.quantizer = ResidualVectorQuantizer(config)

    def encode(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        z = self.encoder(waveform)
        z = z.transpose(1, 2)
        quantized, indices, _ = self.quantizer(z)
        return quantized.transpose(1, 2), indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.quantizer.decode(indices)
        z = z.transpose(1, 2)
        return self.decoder(z)

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        z = self.encoder(waveform).transpose(1, 2)
        quantized, indices, commitment = self.quantizer(z)
        indices = indices.reshape(z.shape[0], z.shape[1], -1)
        reconstructed = self.decode(indices)
        commitment_loss = commitment[0] if commitment else 0.0
        return reconstructed, quantized.transpose(1, 2), indices, commitment_loss

    def get_compression_ratio(self) -> float:
        return 2**3 * self.config.hop_length
