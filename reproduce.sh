#!/bin/bash

# 1. Install dependencies
echo "--- Installing dependencies ---"
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
uv sync

# 2. Load environment variables (HF_TOKEN)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 3. Authenticate with Hugging Face
echo "--- Authenticating with HF ---"
hf auth login --token $HF_TOKEN --add-to-git-credential

# 4. Download Data and Model
echo "--- Downloading Assets ---"
# download datasets
mkdir -p dataset/
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl -O dataset/gsm8k_train.jsonl # original dataset for fine-tuning
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl -O dataset/gsm8k_train_self-instruct.jsonl # part of fine-tuning dataset refined by llama-3.2-1b-instruct
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl -O dataset/gsm8k_test_public.jsonl # gsm8k public test dataset
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl -O dataset/gsm8k_test_private.jsonl # gsm8k private test dataset
wget -nc https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv -O dataset/ailuminate_test.csv # ailuminate test dataset (public + private)

# 5.1 Run Training
# uv run train.py --config_id 0 --data_path dataset/
# uv run train.py --config_id 1 --data_path dataset/
# uv run train.py --config_id 2 --data_path dataset/
# uv run train.py --config_id 3 --data_path dataset/
# uv run train.py --config_id 4 --data_path dataset/
# uv run train.py --config_id 5 --data_path dataset/
# uv run train.py --config_id 6 --data_path dataset/
# uv run train.py --config_id 7 --data_path dataset/
# uv run train.py --config_id 8 --data_path dataset/
# uv run train.py --config_id 9 --data_path dataset/
# uv run train.py --config_id 10 --data_path dataset/

# 5.2 Run Inference and Evaluation
echo "--- Running Evaluation ---"
uv run inference.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_model tutor369/Qwen2.5-7B-Instruct-lora-v1 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_model tutor369/Qwen2.5-7B-Instruct-lora-v2 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_model tutor369/Qwen2.5-7B-Instruct-lora-v3 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v4 --data_path dataset/ --max_new_tokens 1024 --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v5 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v6 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v7 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v8 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v9 --data_path dataset/ --do_sample
uv run inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_model tutor369/Qwen2.5-1.5B-Instruct-lora-v10 --data_path dataset/ --do_sample

echo "--- Complete ---"