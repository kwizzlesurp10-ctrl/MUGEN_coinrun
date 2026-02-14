from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from musicgen.codec import AudioCodec
from musicgen.model import MusicGenModel
from musicgen.text_encoder import TextEncoder
from musicgen.types import GenerationResult


def generate_music(
    prompt: str,
    codec: AudioCodec,
    model: MusicGenModel,
    text_encoder: TextEncoder,
    device: torch.device,
    duration_seconds: float = 5.0,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> GenerationResult:
    sample_rate = codec.config.sample_rate
    compression_ratio = 8
    frames_per_second = sample_rate / compression_ratio
    max_frames = int(duration_seconds * frames_per_second)

    with torch.no_grad():
        text_emb, _ = text_encoder.encode([prompt], device)
        audio_tokens = model.generate(
            text_emb,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            progress_callback=progress_callback,
        )
        waveform = codec.decode(audio_tokens)

    waveform = waveform.squeeze(1)
    return GenerationResult(
        waveform=waveform,
        sample_rate=sample_rate,
        prompt=prompt,
        duration_seconds=waveform.shape[-1] / sample_rate,
    )
