import argparse
import copy
import json
import os
from typing import Dict

import numpy as np
import torch

from models import load_model


DEFAULT_WEIGHTS = {
    "design_semantics": 0.3,
    "color_semantics": 0.2,
    "layout_semantics": 0.2,
    "content_semantics": 0.3,
    "visual_features": 0.3,
    "font": 0.2,
    "layout": 0.2,
    "color": 0.2,
    "design_style": 0.3,
    "content_understanding": 0.4,
    "content": 0.3,
    "main_component": 0.2,
    "customized_component": 0.1,
    "image_component": 0.1,
    "icon_component": 0.1,
}


def extract_fields(data: Dict) -> Dict[str, str]:
    try:
        semantic = data.get("semantic_pattern", {})
        design = semantic.get("Design_semantics", [{}])[0]
        visual = semantic.get("Visual_features", [{}])[0]
        content_block = semantic.get("Content_understanding", [{}])[0]
        component = content_block.get("component", [{}])[0]

        return {
            "design_semantics": design.get("design_semantics", ""),
            "color_semantics": design.get("color_semantics", ""),
            "layout_semantics": design.get("layout_semantics", ""),
            "content_semantics": design.get("content_semantics", ""),
            "visual_features": visual.get("visual_features", ""),
            "font": visual.get("font", ""),
            "layout": visual.get("layout", ""),
            "color": visual.get("color", ""),
            "design_style": visual.get("design_style", ""),
            "content_understanding": content_block.get("content_understanding", ""),
            "content": content_block.get("content", ""),
            "main_component": component.get("main_component", ""),
            "customized_component": component.get("customized_component", ""),
            "image_component": component.get("image_component", ""),
            "icon_component": component.get("icon_component", ""),
        }
    except Exception:
        # Fallback to flat schema if structure differs
        return {key: str(data.get(key, "")) for key in DEFAULT_WEIGHTS}


def encode_weighted_embedding(model, tokenizer, fields: Dict[str, str], weights: Dict[str, float], max_length: int) -> np.ndarray:
    weights_copy = copy.deepcopy(weights)
    if not fields.get("image_component") or fields.get("image_component") in {"", "无", None}:
        weights_copy["image_component"] = 0.0
    if not fields.get("icon_component") or fields.get("icon_component") in {"", "无", None}:
        weights_copy["icon_component"] = 0.0

    embeddings = []
    for key, weight in weights_copy.items():
        text = fields.get(key, "")
        if not text or weight == 0:
            continue
        emb = model.encode(tokenizer, [text], tower="a", max_length=max_length)[0].numpy()
        embeddings.append(weight * emb)

    if not embeddings:
        raise ValueError("No valid text fields found for encoding.")

    return np.sum(embeddings, axis=0)


def process_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, tokenizer = load_model(args.model_dir, device)

    image_files = [f for f in os.listdir(args.image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    json_files = [f.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json") for f in image_files]

    image_paths = []
    image_descriptions = []

    for idx, (json_file, image_file) in enumerate(zip(json_files, image_files), start=1):
        print(f"Processing {idx}/{len(json_files)}: {json_file}")
        json_path = os.path.join(args.json_folder, json_file)
        image_path = os.path.join(args.image_folder, image_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fields = extract_fields(data)
        embedding = encode_weighted_embedding(model, tokenizer, fields, DEFAULT_WEIGHTS, args.max_length)

        image_paths.append(image_path)
        image_descriptions.append(embedding)

    embeddings_matrix = np.stack(image_descriptions, axis=0)
    np.save(args.output_embeddings, embeddings_matrix)
    np.save(args.output_paths, np.array(image_paths))
    print(f"Saved embeddings to {args.output_embeddings} and paths to {args.output_paths}")
    print(embeddings_matrix.shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate weighted text embeddings using a finetuned dual-tower encoder.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to finetuned model directory.")
    parser.add_argument("--image_folder", type=str, default="/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good")
    parser.add_argument("--json_folder", type=str, default="/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good_json")
    parser.add_argument("--output_embeddings", type=str, default="text_embeddings_finetuned.npy")
    parser.add_argument("--output_paths", type=str, default="path_matrix_finetuned.npy")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
