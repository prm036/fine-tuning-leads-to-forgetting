"""
Evaluate our fine-tuned qwen model on the GSM8K and Ailuminate test datasets.
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace):
    model_path = Path(args.model_path)
    model_path.resolve()
    data_path = Path(args.data_path)
    data_path.resolve()

    print(f"Loading model from {model_path}...")
    print(f"Loading data from {data_path}...")

    print("\n--- EVALUATING MODEL ---")
    # TODO run inference on datasets

    # Evaluation
    print("\n--- FINAL METRICS ---")
    # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    main(parser.parse_args())
