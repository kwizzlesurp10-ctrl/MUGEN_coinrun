#!/usr/bin/env python3
"""Create a small demo dataset of synthetic music-like audio for quick experimentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000
DURATION_SEC = 4.0
DESCRIPTIONS = [
    "upbeat electronic dance music with synthesizers",
    "chill lo-fi hip hop beats",
    "ambient atmospheric pad with soft strings",
    "acoustic guitar fingerstyle folk",
    "driving rock drums and bass",
    "jazzy piano with walking bass",
    "minimal techno with deep bass",
    "classical piano sonata excerpt",
    "reggae rhythm with organ",
    "dreamy shoegaze guitar layers",
]


def generate_tone(freq: float, duration: float, sr: int, volume: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return volume * np.sin(2 * np.pi * freq * t)


def generate_chord(freqs: list[float], duration: float, sr: int) -> np.ndarray:
    out = np.zeros(int(sr * duration), dtype=np.float32)
    for f in freqs:
        out += generate_tone(f, duration, sr, volume=0.15)
    return out / len(freqs)


def generate_simple_melody(duration: float, sr: int) -> np.ndarray:
    freqs = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]
    beat = 0.25
    n_beats = int(duration / beat)
    out = np.zeros(int(sr * duration), dtype=np.float32)
    for i in range(n_beats):
        idx = i % len(freqs)
        start = int(i * beat * sr)
        end = int((i + 1) * beat * sr)
        out[start:end] = generate_tone(freqs[idx], beat, sr, 0.2)
    return out


def generate_pad(duration: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    base = 110.0
    return 0.2 * (
        np.sin(2 * np.pi * base * t)
        + 0.5 * np.sin(2 * np.pi * base * 1.5 * t)
        + 0.3 * np.sin(2 * np.pi * base * 2 * t)
    )


def generate_rhythmic(duration: float, sr: int) -> np.ndarray:
    beat = 0.5
    n_beats = int(duration / beat)
    out = np.zeros(int(sr * duration), dtype=np.float32)
    for i in range(n_beats):
        if i % 4 in (0, 2):
            start = int(i * beat * sr)
            end = int((i + 0.1) * sr)
            end = min(end, len(out))
            out[start:end] = 0.4
    return out


def create_sample(idx: int, output_dir: Path) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"demo_{idx:03d}.wav"

    generators = [
        lambda d, sr: generate_chord([261.63, 329.63, 392.0], d, sr),
        lambda d, sr: generate_simple_melody(d, sr),
        lambda d, sr: generate_pad(d, sr),
        lambda d, sr: generate_rhythmic(d, sr) * generate_tone(80, d, sr, 0.3),
        lambda d, sr: generate_chord([220, 277.18, 329.63], d, sr) + generate_rhythmic(d, sr) * 0.2,
    ]
    gen = generators[idx % len(generators)]
    audio = gen(DURATION_SEC, SAMPLE_RATE)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, SAMPLE_RATE)
    return str(path), DESCRIPTIONS[idx % len(DESCRIPTIONS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/demo")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.txt"

    with open(manifest_path, "w") as f:
        for i in range(args.num_samples):
            path, desc = create_sample(i, output_dir)
            f.write(f"{path}|{desc}\n")

    print(f"Created {args.num_samples} samples in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
