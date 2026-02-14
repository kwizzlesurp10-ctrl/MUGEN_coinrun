from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from musicgen.types import CodecConfig


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.latent_dim)
                for _ in range(config.n_codebooks)
            ]
        )
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        residual = z
        quantized = torch.zeros_like(z)
        indices_list: list[torch.Tensor] = []
        commitment_loss = 0.0

        for codebook in self.codebooks:
            residual_flat = residual.reshape(-1, residual.shape[-1])
            distances = torch.cdist(residual_flat, codebook.weight, p=2)
            indices = distances.argmin(dim=-1)
            indices_list.append(indices)

            quantized_i = codebook(indices)
            quantized_i = quantized_i.reshape(residual.shape)
            quantized = quantized + quantized_i
            residual = residual - quantized_i

            commitment_loss = commitment_loss + F.mse_loss(residual, torch.zeros_like(residual))

        indices_stack = torch.stack(indices_list, dim=-1)
        return quantized, indices_stack, [commitment_loss]

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros(
            indices.shape[0],
            indices.shape[1],
            self.config.latent_dim,
            device=indices.device,
            dtype=torch.float32,
        )
        for i, codebook in enumerate(self.codebooks):
            quantized = quantized + codebook(indices[..., i])
        return quantized
