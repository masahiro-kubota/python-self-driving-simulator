#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


def convert_model(ckpt_path: str, output_path: str):
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    numpy_weights = {}
    for key, value in state_dict.items():
        # Replace dots with underscores for compatibility if needed,
        # or keep consistent with Trainer's saving logic
        new_key = key.replace(".", "_")
        numpy_weights[new_key] = value.numpy()

    print(f"Saving numpy weights to {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, numpy_weights)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to NumPy weights")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy file")

    args = parser.parse_args()
    convert_model(args.ckpt, args.output)
