#!/usr/bin/env python3
"""Generate music from text prompt."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

GENERATION_STATUS_FILE = Path(__file__).resolve().parent.parent / "output" / "generation_status.json"


def _write_status(data: dict) -> None:
    GENERATION_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GENERATION_STATUS_FILE.write_text(json.dumps(data), encoding="utf-8")


def _clear_running_and_write(data: dict) -> None:
    _write_status(data)

from musicgen.agent import run_agent_workflow
from musicgen.codec import AudioCodec
from musicgen.inference import generate_music
from musicgen.model import MusicGenModel
from musicgen.text_encoder import TextEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Text description of music to generate")
    parser.add_argument("--codec", required=True, help="Path to codec checkpoint")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--agent", action="store_true", help="Use agent workflow (iterative refinement)")
    parser.add_argument("--agent-iterations", type=int, default=3)
    parser.add_argument("--feedback", default="none", choices=["none", "classifier", "llm"])
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "medium", "large"])
    args = parser.parse_args()

    def _write_progress(step: int, total: int) -> None:
        pct = min(99, int(100 * step / total)) if total else 0
        if not hasattr(_write_progress, "_last") or pct >= getattr(_write_progress, "_last", -1) + 2:
            _write_progress._last = pct  # type: ignore[attr-defined]
            _write_status({
                "status": "running",
                "prompt": args.prompt,
                "started_at": datetime.now(UTC).isoformat(),
                "progress": pct,
            })

    _write_status({
        "status": "running",
        "prompt": args.prompt,
        "started_at": datetime.now(UTC).isoformat(),
        "progress": 0,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from musicgen.config_loader import get_text_embed_dim, get_text_encoder_name, load_preset

    codec_config, transformer_config = load_preset(args.preset)
    text_embed_dim = get_text_embed_dim(args.preset)
    codec = AudioCodec(codec_config)
    codec.load_state_dict(torch.load(args.codec, map_location="cpu"))
    codec = codec.to(device).eval()

    model = MusicGenModel(codec_config, transformer_config, text_embed_dim=text_embed_dim)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model = model.to(device).eval()

    text_encoder = TextEncoder(get_text_encoder_name(args.preset)).to(device).eval()

    try:
        if args.agent:
            run_agent_workflow(
                args.prompt,
                args.codec,
                args.model,
                str(Path(args.output).parent),
                max_iterations=args.agent_iterations,
                feedback_mode=args.feedback,
                preset=args.preset,
            )
            _clear_running_and_write({"status": "done", "output_path": args.output})
        else:
            result = generate_music(
                args.prompt,
                codec,
                model,
                text_encoder,
                device,
                duration_seconds=args.duration,
                temperature=args.temperature,
                progress_callback=_write_progress,
            )
            import soundfile as sf

            wav = result.waveform.cpu().numpy()
            if wav.ndim > 1:
                wav = wav[0]
            sf.write(args.output, wav, result.sample_rate)
            print(f"Saved to {args.output}")
            _clear_running_and_write({"status": "done", "output_path": args.output})
    except Exception as e:  # noqa: BLE001
        _clear_running_and_write({"status": "error", "message": str(e)})
        raise


if __name__ == "__main__":
    main()
