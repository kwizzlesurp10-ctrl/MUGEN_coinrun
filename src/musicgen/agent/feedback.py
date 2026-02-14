from __future__ import annotations

from typing import Optional

import torch

from musicgen.types import AgentFeedback, GenerationResult


def spectral_flatness(waveform: torch.Tensor, frame_size: int = 2048) -> float:
    x = waveform.cpu().numpy()
    if x.size < frame_size:
        return 0.0
    n_frames = x.size // frame_size
    frames = x[: n_frames * frame_size].reshape(n_frames, frame_size)
    eps = 1e-10
    geom = (frames ** 2 + eps).prod(axis=1) ** (1 / frame_size)
    arith = (frames ** 2 + eps).mean(axis=1)
    flatness = (geom / (arith + eps)).mean()
    return float(min(max(flatness, 0), 1))


def rms_energy(waveform: torch.Tensor) -> float:
    x = waveform.cpu().float()
    return float((x ** 2).mean().sqrt().clamp(0, 1))


def zero_crossing_rate(waveform: torch.Tensor) -> float:
    x = waveform.cpu().numpy()
    zcr = (x[:-1] * x[1:] < 0).mean()
    return float(min(zcr * 2, 1))


def classifier_feedback(result: GenerationResult) -> AgentFeedback:
    wav = result.waveform
    if wav.dim() > 1:
        wav = wav[0]
    flatness = spectral_flatness(wav)
    energy = rms_energy(wav)
    zcr = zero_crossing_rate(wav)
    score = 0.4 * flatness + 0.4 * energy + 0.2 * (1 - abs(zcr - 0.1))
    score = min(max(score, 0), 1)
    if score < 0.3:
        feedback_text = "add more harmonic content and increase volume"
    elif score < 0.6:
        feedback_text = "improve spectral richness and dynamics"
    else:
        feedback_text = "slight refinement for more clarity"
    return AgentFeedback(score=score, feedback_text=feedback_text, iteration=0)


def llm_feedback(
    result: GenerationResult,
    model_id: str = "google/flan-t5-small",
    device: Optional[str] = None,
) -> AgentFeedback:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        return AgentFeedback(
            score=0.5,
            feedback_text="add more variation and musical interest",
            iteration=0,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    dev = device or "cpu"
    model = model.to(dev)

    prompt = (
        f"Given a music description: \"{result.prompt}\". "
        "Suggest one short refinement to make the generated music better. "
        "Reply in 10 words or less."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=30)
    feedback_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return AgentFeedback(
        score=0.5,
        feedback_text=feedback_text or "add more variation",
        iteration=0,
    )
