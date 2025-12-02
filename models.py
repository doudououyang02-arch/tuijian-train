import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


@dataclass
class DualEncoderConfig:
    base_model_name: str
    projection_dim: int
    dropout: float = 0.1


class DualEncoderModel(nn.Module):
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.encoder.config.hidden_size
        projection_dim = config.projection_dim or hidden_size
        self.head_a = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.head_b = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, inputs_a: Dict[str, torch.Tensor], inputs_b: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device
        inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
        inputs_b = {k: v.to(device) for k, v in inputs_b.items()}

        output_a = self.encoder(**inputs_a, return_dict=True)
        output_b = self.encoder(**inputs_b, return_dict=True)

        pooled_a = mean_pooling(output_a, inputs_a["attention_mask"])
        pooled_b = mean_pooling(output_b, inputs_b["attention_mask"])

        emb_a = nn.functional.normalize(self.head_a(pooled_a), p=2, dim=-1)
        emb_b = nn.functional.normalize(self.head_b(pooled_b), p=2, dim=-1)
        return emb_a, emb_b

    @torch.no_grad()
    def encode(self, tokenizer: AutoTokenizer, texts, tower: str = "a", max_length: int = 512):
        if tower not in {"a", "b"}:
            raise ValueError("tower must be 'a' or 'b'")
        device = next(self.parameters()).device
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.encoder(**encoded, return_dict=True)
        pooled = mean_pooling(outputs, encoded["attention_mask"])
        head = self.head_a if tower == "a" else self.head_b
        embeddings = nn.functional.normalize(head(pooled), p=2, dim=-1)
        return embeddings.cpu()


def save_model(model: DualEncoderModel, tokenizer: AutoTokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(model.config)}, os.path.join(save_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_dir)


def load_model(save_dir: str, device: torch.device) -> Tuple[DualEncoderModel, AutoTokenizer]:
    checkpoint = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=device)
    config = DualEncoderConfig(**checkpoint["config"])
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = DualEncoderModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, tokenizer
