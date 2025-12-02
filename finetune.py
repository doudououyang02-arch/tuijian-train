import argparse
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models import DualEncoderConfig, DualEncoderModel, save_model


SEMANTIC_FIELDS = [
    "design_semantics",
    "color_semantics",
    "layout_semantics",
    "content_semantics",
    "visual_features",
    "font",
    "layout",
    "color",
    "design_style",
    "content_understanding",
    "content",
    "main_component",
    "customized_component",
    "image_component",
    "icon_component",
]


class TextPairDataset(Dataset):
    def __init__(self, json_dir: str, tokenizer: AutoTokenizer, min_fields: int, max_fields: int, mask_ratio: Tuple[float, float], max_length: int):
        self.samples = glob.glob(os.path.join(json_dir, "*.json"))
        self.tokenizer = tokenizer
        self.min_fields = min_fields
        self.max_fields = max_fields
        self.mask_ratio = mask_ratio
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _compose_text(self, data: Dict) -> str:
        parts: List[str] = []
        available_fields = [f for f in SEMANTIC_FIELDS if f in data and data[f]]
        if not available_fields:
            return ""
        k = random.randint(self.min_fields, min(self.max_fields, len(available_fields)))
        chosen = random.sample(available_fields, k=k)
        for field in chosen:
            value = data.get(field, "")
            if isinstance(value, list):
                value = " ".join(map(str, value))
            parts.append(f"{field}: {value}")
        return " \n ".join(parts)

    def _apply_mask(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return text
        ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        mask_count = max(1, int(len(tokens) * ratio))
        mask_indices = random.sample(range(len(tokens)), k=mask_count)
        for idx in mask_indices:
            tokens[idx] = self.tokenizer.mask_token or "[MASK]"
        return self.tokenizer.convert_tokens_to_string(tokens)

    def __getitem__(self, idx):
        json_path = self.samples[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text_a = self._compose_text(data)
        text_b = self._apply_mask(text_a)
        return text_a, text_b


class Collator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str]]):
        texts_a, texts_b = zip(*batch)
        inputs_a = self.tokenizer(
            list(texts_a),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs_b = self.tokenizer(
            list(texts_b),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return inputs_a, inputs_b


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = TextPairDataset(
        json_dir=args.json_dir,
        tokenizer=tokenizer,
        min_fields=args.min_fields,
        max_fields=args.max_fields,
        mask_ratio=(args.min_mask, args.max_mask),
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer, args.max_length),
    )

    config = DualEncoderConfig(
        base_model_name=args.base_model,
        projection_dim=args.projection_dim or tokenizer.model_max_length,
        dropout=args.dropout,
    )
    model = DualEncoderModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            inputs_a, inputs_b = batch
            optimizer.zero_grad()
            emb_a, emb_b = model(inputs_a, inputs_b)
            target = torch.ones(emb_a.size(0), device=device)
            loss = F.cosine_embedding_loss(emb_a, emb_b, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "best")
            save_model(model, tokenizer, save_path)
            print(f"Saved new best model to {save_path}")

    final_path = os.path.join(args.output_dir, "last")
    save_model(model, tokenizer, final_path)
    print(f"Training completed. Final model saved to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a shared encoder dual-tower model on JSON text pairs.")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing JSON files for training.")
    parser.add_argument("--base_model", type=str, default="BAAI/bge-base-zh", help="Base model path or name for initialization.")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model", help="Where to save checkpoints.")
    parser.add_argument("--projection_dim", type=int, default=None, help="Projection dimension for tower heads.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_fields", type=int, default=3, help="Minimum number of JSON fields to sample per example.")
    parser.add_argument("--max_fields", type=int, default=8, help="Maximum number of JSON fields to sample per example.")
    parser.add_argument("--min_mask", type=float, default=0.1, help="Minimum masking ratio.")
    parser.add_argument("--max_mask", type=float, default=0.95, help="Maximum masking ratio.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
