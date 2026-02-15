#!/bin/bash

# 1. Install dependencies
echo "--- Installing dependencies ---"
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
uv sync

# 2. Load environment variables (HUGGINGFACE_HUB_TOKEN)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 3. Authenticate with Hugging Face
echo "--- Authenticating with HF ---"
hf auth login --token $HUGGINGFACE_HUB_TOKEN --add-to-git-credential

# 4. Download Data and Model
echo "--- Downloading Assets ---"
# download datasets
mkdir -p dataset/
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl -O dataset/gsm8k_train.jsonl # original dataset for fine-tuning
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl -O dataset/gsm8k_train_self-instruct.jsonl # part of fine-tuning dataset refined by llama-3.2-1b-instruct
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl -O dataset/gsm8k_test_public.jsonl # gsm8k public test dataset
wget -nc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl -O dataset/gsm8k_test_private.jsonl # gsm8k private test dataset
wget -nc https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv -O dataset/ailuminate_test.csv # ailuminate test dataset (public + private)

# download our model
HF_UNAME="andnet-deboer"
MODEL_NAME="qwen2.5-1.5b-gsm8k-lora"
hf download $HF_UNAME/$MODEL_NAME --local-dir ./model

# 5. Run Inference and Evaluation
echo "--- Running Evaluation ---"
uv run eval.py --model_path ./model --data_path dataset/

echo "--- Reproduction Complete ---"