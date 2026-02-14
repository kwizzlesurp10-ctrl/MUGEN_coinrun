from __future__ import annotations

from pathlib import Path
from typing import Literal

from musicgen.types import CodecConfig, TransformerConfig

Preset = Literal["tiny", "small", "medium", "large"]


PRESETS: dict[Preset, tuple[CodecConfig, TransformerConfig]] = {
    "tiny": (
        CodecConfig(
            sample_rate=24000,
            hop_length=320,
            n_fft=512,
            n_mels=64,
            latent_dim=64,
            n_codebooks=4,
            codebook_size=512,
            n_residual_layers=2,
        ),
        TransformerConfig(
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.1,
        ),
    ),
    "small": (
        CodecConfig(
            sample_rate=24000,
            hop_length=320,
            n_fft=1024,
            n_mels=80,
            latent_dim=96,
            n_codebooks=6,
            codebook_size=1024,
            n_residual_layers=3,
        ),
        TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
            max_seq_len=1024,
            dropout=0.1,
        ),
    ),
    "medium": (
        CodecConfig(
            sample_rate=24000,
            hop_length=320,
            n_fft=1024,
            n_mels=80,
            latent_dim=128,
            n_codebooks=8,
            codebook_size=2048,
            n_residual_layers=3,
        ),
        TransformerConfig(
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            max_seq_len=2048,
            dropout=0.1,
        ),
    ),
    "large": (
        CodecConfig(
            sample_rate=24000,
            hop_length=320,
            n_fft=1024,
            n_mels=80,
            latent_dim=128,
            n_codebooks=8,
            codebook_size=2048,
            n_residual_layers=3,
        ),
        TransformerConfig(
            d_model=1536,
            n_heads=24,
            n_layers=32,
            d_ff=6144,
            max_seq_len=4096,
            dropout=0.1,
        ),
    ),
}


def load_preset(preset: Preset) -> tuple[CodecConfig, TransformerConfig]:
    return PRESETS[preset]


def get_text_embed_dim(preset: Preset) -> int:
    dims: dict[Preset, int] = {"tiny": 512, "small": 512, "medium": 768, "large": 768}
    return dims.get(preset, 768)


def get_text_encoder_name(preset: Preset) -> str:
    names: dict[Preset, str] = {
        "tiny": "t5-small",
        "small": "t5-small",
        "medium": "t5-base",
        "large": "t5-base",
    }
    return names.get(preset, "t5-base")
