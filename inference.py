# ─── Entry point for Inference ─────────────────────────────────────────────────────────────

"""
Evaluate our fine-tuned qwen model on the GSM8K and Ailuminate test datasets.
"""
import os
import argparse
import torch
from pathlib import Path
from finetune import run_inference, get_hf_token
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def main(args: argparse.Namespace):

    # Set GPU for this job (SLURM_LOCALID is 0 on single-GPU jobs)
    gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Authenticate with HuggingFace
    get_hf_token()

    pretrained_model = args.base_model
    adapter_model = args.adapter_model
    data_path = Path(args.data_path)
    data_path.resolve()

    print(f"Loading data from {data_path}...")

    print("\n--- EVALUATING MODEL ---")
    sft_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model, 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
    )
    sft_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    sft_tokenizer.model_max_length = 10000
    sft_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    kwargs = {"max_new_tokens": args.max_new_tokens, "test_and_shot": args.test_and_shot, "do_sample": args.do_sample}
    run_inference(sft_model, sft_tokenizer, adapter_model, data_path, **kwargs)
    print(f"\n Job complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for LoRA fine-tuned models")

    # Model args
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--adapter_model",
        type=str,
        required=True,
        help="Adapter repo ID or local path (e.g. tutor369/Qwen2.5-7B-Instruct-lora-v1)",
    )

    # Data args
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing dataset files",
    )

    # Inference args
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--test_and_shot",
        type=int,
        default=8,
        help="Number of few-shot examples at test time (default: 8)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for generation; omit for greedy decoding",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Base model:      {args.base_model}")
    print(f"Adapter model:   {args.adapter_model}")
    print(f"Data path:       {args.data_path}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Test & shot:     {args.test_and_shot}")
    print(f"Do sample:       {args.do_sample}")
    main(args)
