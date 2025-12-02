import argparse
import os
import shutil

import numpy as np
import torch
from sentence_transformers import util

from models import load_model


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, tokenizer = load_model(args.model_dir, device)

    text_embeddings_matrix = np.load(args.embeddings_path)
    path_matrix = np.load(args.paths_path)

    query_out = model.encode(tokenizer, [args.query], tower="a", max_length=args.max_length)
    q_dense = query_out.numpy()

    dense_sim = util.cos_sim(torch.from_numpy(q_dense), torch.from_numpy(text_embeddings_matrix)).reshape(-1)

    topk_indices = torch.argsort(dense_sim, descending=True)[: args.top_k]
    candidates = [(dense_sim[i].item(), path_matrix[i]) for i in topk_indices]

    os.makedirs(args.output_dir, exist_ok=True)
    for score, img_path in candidates:
        img_name = os.path.basename(img_path)
        json_path = os.path.join("/".join(img_path.split("/")[:-1]).replace("final_good", "final_good_json"), img_name.replace(".png", ".json").replace(".jpg", ".json"))
        output_path = os.path.join(args.output_dir, img_name)
        output_json_path = os.path.join(args.output_dir, os.path.basename(json_path))
        shutil.copy(img_path, output_path)
        if os.path.exists(json_path):
            shutil.copy(json_path, output_json_path)
        print(f"Score: {score:.4f} | Image: {img_path}")
    print(f"Results saved to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with finetuned dual-tower embeddings.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to finetuned model directory.")
    parser.add_argument("--embeddings_path", type=str, default="text_embeddings_finetuned.npy")
    parser.add_argument("--paths_path", type=str, default="path_matrix_finetuned.npy")
    parser.add_argument("--query", type=str, default="4行5列的表格")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./text_results")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
