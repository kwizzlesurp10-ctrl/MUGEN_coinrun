from __future__ import annotations

import pytest
import torch

from musicgen.model import MusicGenModel
from musicgen.types import CodecConfig, TransformerConfig


def test_model_forward() -> None:
    codec_config = CodecConfig(n_codebooks=4, codebook_size=512)
    transformer_config = TransformerConfig(d_model=256, n_heads=4, n_layers=2, d_ff=512)
    model = MusicGenModel(
        codec_config,
        transformer_config,
        text_embed_dim=256,
    )
    batch, seq, n_cb = 2, 10, 4
    audio_tokens = torch.randint(0, 512, (batch, seq, n_cb))
    text_emb = torch.randn(batch, 20, 256)
    logits = model(audio_tokens, text_emb, None)
    assert logits.shape == (batch, seq * n_cb, codec_config.codebook_size)


def test_model_generate() -> None:
    codec_config = CodecConfig(n_codebooks=4, codebook_size=512)
    transformer_config = TransformerConfig(d_model=256, n_heads=4, n_layers=2, d_ff=512)
    model = MusicGenModel(
        codec_config,
        transformer_config,
        text_embed_dim=256,
    )
    text_emb = torch.randn(1, 10, 256)
    output = model.generate(text_emb, max_frames=5, temperature=1.0)
    assert output.shape == (1, 5, codec_config.n_codebooks)
