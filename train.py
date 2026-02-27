# ─── Entry point for Training ─────────────────────────────────────────────────────────────

import os
import argparse
from pathlib import Path
from config import TRAINING_CONFIGS as CONFIGS
from finetune import run_training, get_hf_token

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument(
        "--config_id",
        type=int,
        required=True,
        choices=range(10),
        help="Index into CONFIGS list (0-9)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing training data files",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()

    cfg = CONFIGS[args.config_id]
    print(f"\n{'='*60}")
    print(f"Config {args.config_id}: {cfg}")
    print(f"{'='*60}\n")

    # Set GPU for this job (SLURM_LOCALID is 0 on single-GPU jobs)
    gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Authenticate with HuggingFace
    get_hf_token()

    sft_model, sft_tokenizer, adapter_path = run_training(cfg, data_path)

    print(f"\n[{cfg['version']}] Training complete.")

if __name__ == "__main__":
    main()