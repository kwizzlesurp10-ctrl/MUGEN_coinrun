from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class CodecConfig:
    sample_rate: int = 24000
    hop_length: int = 320
    n_fft: int = 1024
    n_mels: int = 80
    latent_dim: int = 128
    n_codebooks: int = 8
    codebook_size: int = 2048
    n_residual_layers: int = 3


@dataclass
class TransformerConfig:
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.1


@dataclass
class GenerationResult:
    waveform: torch.Tensor
    sample_rate: int
    prompt: str
    duration_seconds: float


@dataclass
class AgentFeedback:
    score: float
    feedback_text: str
    iteration: int


FeedbackMode = Literal["none", "classifier", "llm"]
