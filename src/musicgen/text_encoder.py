from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "t5-base", max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.max_length = max_length
        self.embed_dim = self.model.config.d_model

    def encode(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state, attention_mask
