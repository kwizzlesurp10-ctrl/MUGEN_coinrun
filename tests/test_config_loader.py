from __future__ import annotations

import pytest

from musicgen.config_loader import get_text_embed_dim, get_text_encoder_name, load_preset


def test_load_preset_tiny() -> None:
    codec_config, transformer_config = load_preset("tiny")
    assert codec_config.n_codebooks == 4
    assert codec_config.codebook_size == 512
    assert transformer_config.d_model == 256
    assert transformer_config.n_layers == 4


def test_load_preset_small() -> None:
    codec_config, transformer_config = load_preset("small")
    assert codec_config.n_codebooks == 6
    assert transformer_config.d_model == 512


def test_get_text_embed_dim() -> None:
    assert get_text_embed_dim("tiny") == 512
    assert get_text_embed_dim("medium") == 768


def test_get_text_encoder_name() -> None:
    assert get_text_encoder_name("tiny") == "t5-small"
    assert get_text_encoder_name("large") == "t5-base"
