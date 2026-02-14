from __future__ import annotations

import pytest
import torch

from musicgen.codec import AudioCodec
from musicgen.types import CodecConfig


def test_codec_encode_decode_roundtrip() -> None:
    config = CodecConfig(n_codebooks=4, codebook_size=512, latent_dim=64)
    codec = AudioCodec(config)
    waveform = torch.randn(2, 1, 24000)
    reconstructed, _, indices, _ = codec(waveform)
    assert indices.shape[0] == 2
    assert indices.shape[-1] == config.n_codebooks
    decoded = codec.decode(indices)
    assert decoded.shape == waveform.shape


def test_codec_encode_output_shapes() -> None:
    config = CodecConfig()
    codec = AudioCodec(config)
    waveform = torch.randn(1, 48000)
    quantized, indices = codec.encode(waveform)
    assert quantized.dim() == 3
    assert indices.shape[-1] == config.n_codebooks
