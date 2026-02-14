from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from musicgen.types import CodecConfig, TransformerConfig
from musicgen.codec import AudioCodec


def _interleave_codebook_tokens(indices: torch.Tensor) -> torch.Tensor:
    batch, seq, n_codebooks = indices.shape
    interleaved = indices.permute(0, 2, 1).reshape(batch, seq * n_codebooks)
    return interleaved


def _deinterleave_codebook_tokens(
    interleaved: torch.Tensor, n_codebooks: int
) -> torch.Tensor:
    batch, total = interleaved.shape
    seq = total // n_codebooks
    return interleaved.reshape(batch, n_codebooks, seq).permute(0, 2, 1)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, use_cross_attn: bool = True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = (
            nn.MultiheadAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            if use_cross_attn
            else None
        )
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm_cross = nn.LayerNorm(config.d_model) if use_cross_attn else None

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = residual + attn_out

        if self.cross_attn is not None and text_emb is not None:
            residual = x
            x = self.norm_cross(x)
            cross_out, _ = self.cross_attn(x, text_emb, text_emb, need_weights=False)
            x = residual + cross_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class MusicGenModel(nn.Module):
    def __init__(
        self,
        codec_config: CodecConfig,
        transformer_config: TransformerConfig,
        text_vocab_size: int = 32128,
        text_embed_dim: int = 768,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.codec_config = codec_config
        self.transformer_config = transformer_config

        vocab_size = codec_config.codebook_size * codec_config.n_codebooks
        self.audio_embed = nn.Embedding(vocab_size, transformer_config.d_model)
        self.text_proj = nn.Linear(text_embed_dim, transformer_config.d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(transformer_config, use_cross_attn=use_cross_attn)
                for _ in range(transformer_config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(transformer_config.d_model)
        self.lm_head = nn.Linear(
            transformer_config.d_model,
            codec_config.codebook_size,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, n_codebooks = audio_tokens.shape
        flat_tokens = _interleave_codebook_tokens(audio_tokens)
        offset = (
            torch.arange(seq_len * n_codebooks, device=flat_tokens.device) % n_codebooks
            * self.codec_config.codebook_size
        )
        token_ids = flat_tokens + offset.unsqueeze(0)

        x = self.audio_embed(token_ids)
        text_proj = self.text_proj(text_embeddings)
        if text_mask is not None:
            text_proj = text_proj * text_mask.unsqueeze(-1)

        text_sum = text_proj.sum(dim=1, keepdim=True)
        x = x + text_sum

        causal_mask = self._build_causal_mask(seq_len * n_codebooks, x.device)

        for block in self.blocks:
            x = block(x, text_proj, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        text_embeddings: torch.Tensor,
        max_frames: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        batch = text_embeddings.shape[0]
        n_codebooks = self.codec_config.n_codebooks
        device = text_embeddings.device
        max_seq = self.transformer_config.max_seq_len
        total_tokens = max_frames * n_codebooks

        flat_tokens: list[torch.Tensor] = []

        for step in range(total_tokens):
            if progress_callback is not None:
                progress_callback(step + 1, total_tokens)
            if not flat_tokens:
                prompt = torch.zeros(
                    batch, 1, n_codebooks, dtype=torch.long, device=device
                )
            else:
                n_complete = len(flat_tokens) // n_codebooks
                if n_complete == 0:
                    padded = flat_tokens + [
                        torch.zeros(batch, dtype=torch.long, device=device)
                        for _ in range(n_codebooks - len(flat_tokens))
                    ]
                    prompt = torch.stack(padded, dim=1).unsqueeze(1)
                else:
                    stacked = torch.stack(flat_tokens[: n_complete * n_codebooks], dim=1)
                    prompt = stacked.reshape(batch, n_complete, n_codebooks)[:, -max_seq:, :]

            logits = self.forward(prompt, text_embeddings, None)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            flat_tokens.append(next_token)

        stacked = torch.stack(flat_tokens, dim=1)
        return stacked.reshape(batch, max_frames, n_codebooks)
