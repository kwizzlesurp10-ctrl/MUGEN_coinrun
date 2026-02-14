#!/usr/bin/env python3
"""Train the MusicGen transformer (text-to-audio) from scratch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from musicgen.codec import AudioCodec
from musicgen.model import MusicGenModel
from musicgen.text_encoder import TextEncoder
from musicgen.types import CodecConfig, TransformerConfig
from musicgen.model.transformer import _interleave_codebook_tokens


def train_model(
    manifest_path: str,
    codec_checkpoint: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_frames: int = 150,
    preset: str = "small",
) -> None:
    from musicgen.config_loader import get_text_embed_dim, get_text_encoder_name, load_preset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_config, transformer_config = load_preset(preset)
    text_embed_dim = get_text_embed_dim(preset)
    text_encoder_name = get_text_encoder_name(preset)

    codec = AudioCodec(codec_config)
    codec.load_state_dict(torch.load(codec_checkpoint, map_location="cpu"))
    codec = codec.to(device).eval()

    text_encoder = TextEncoder(text_encoder_name).to(device).eval()

    model = MusicGenModel(
        codec_config,
        transformer_config,
        text_embed_dim=text_embed_dim,
    ).to(device)

    from musicgen.data import MusicDataset

    dataset = MusicDataset(
        manifest_path,
        codec,
        sample_rate=codec_config.sample_rate,
        max_duration_seconds=10.0,
    )

    def collate(batch: list):
        indices_list = [b[0] for b in batch]
        lengths = [b[0].shape[0] for b in indices_list]
        indices = torch.nn.utils.rnn.pad_sequence(
            indices_list, batch_first=True, padding_value=0
        )
        texts = [b[2] for b in batch]
        return indices, texts, lengths

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_cb = codec_config.n_codebooks
        for indices, texts, lengths in loader:
            max_len = min(max(lengths), max_frames)
            indices = indices[:, : max_len * n_cb, :]
            seq_len = indices.shape[1] // n_cb
            indices = indices[:, : seq_len * n_cb, :].to(device)

            with torch.no_grad():
                text_emb, text_mask = text_encoder.encode(texts, device)

            logits = model(indices, text_emb, text_mask)
            flat_tokens = _interleave_codebook_tokens(indices)
            targets = flat_tokens[:, 1:]
            logits_for_loss = logits[:, :-1, :]

            mask = torch.zeros_like(targets, dtype=torch.bool)
            for b, l in enumerate(lengths):
                valid = min(l * n_cb - 1, seq_len * n_cb - 1)
                if valid > 0:
                    mask[b, :valid] = True

            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            loss = (loss.reshape_as(targets) * mask).sum() / mask.sum().clamp(min=1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss / len(loader):.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), output_path / f"model_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), output_path / "model_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--codec", required=True, help="Path to trained codec checkpoint")
    parser.add_argument("--output", default="checkpoints/model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "medium", "large"])
    args = parser.parse_args()
    train_model(
        args.manifest,
        args.codec,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        args.max_frames,
        args.preset,
    )
