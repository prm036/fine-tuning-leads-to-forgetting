#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_inference_job.sh  –  Submit 10 inference jobs as a SLURM array
#
# Usage:
#   bash slurm_inference_job.sh                        # infer all 10 configs
#   SLURM_ARRAY_RANGE=0-2 bash slurm_inference_job.sh  # infer configs 0-2 only
# ─────────────────────────────────────────────────────────────────────────────

ARRAY_RANGE="${SLURM_ARRAY_RANGE:-0-9}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lora_inference
#SBATCH --array=${ARRAY_RANGE}%10
#SBATCH --output=logs/infer_%A_cfg%a.out
#SBATCH --error=logs/infer_%A_cfg%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gengpu
#SBATCH --account=e33188

# ── Environment ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "SLURM Job ID   : \$SLURM_JOB_ID"
echo "Array Task ID  : \$SLURM_ARRAY_TASK_ID"
echo "Node           : \$SLURMD_NODENAME"
echo "Started at     : \$(date)"
echo "=========================================="

# ── Cache dirs ────────────────────────────────────────────────────────────────
export HF_HOME=/gpfs/projects/e33188/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/e33188/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/e33188/hf_cache
source .env
mkdir -p /gpfs/projects/e33188/hf_cache

# ── Activate environment ──────────────────────────────────────────────────────
source /projects/e33188/myenv/venv/bin/activate

# ── Move to working directory ─────────────────────────────────────────────────
cd "\${SLURM_SUBMIT_DIR}"
mkdir -p logs

# ── Resolve base model + adapter from task ID ────────────────────────────────
# Task 0 → v1  (7B)   tutor369/Qwen2.5-7B-Instruct-lora-v1
# Task 1 → v2  (7B)   tutor369/Qwen2.5-7B-Instruct-lora-v2
# Task 2 → v3  (7B)   tutor369/Qwen2.5-7B-Instruct-lora-v3
# Task 3 → v4  (1.5B) tutor369/Qwen2.5-1.5B-Instruct-lora-v4  [max_new_tokens=1024]
# Task 4 → v5  (1.5B) tutor369/Qwen2.5-1.5B-Instruct-lora-v5
# ...
# Task 9 → v10 (1.5B) tutor369/Qwen2.5-1.5B-Instruct-lora-v10
TASK_ID=\${SLURM_ARRAY_TASK_ID}
VERSION=\$((TASK_ID + 1))

if [ \$TASK_ID -le 2 ]; then
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
else
    BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
fi

MODEL_SHORTNAME=\$(echo \$BASE_MODEL | cut -d'/' -f2)
ADAPTER_MODEL="\${HF_UNAME}/\${MODEL_SHORTNAME}-lora-v\${VERSION}"

# config_id 3 (task ID 3) uses 1024 tokens, all others use 512
if [ \$TASK_ID -eq 3 ]; then
    MAX_NEW_TOKENS=1024
else
    MAX_NEW_TOKENS=512
fi

echo "Task ID:         \$TASK_ID"
echo "Version:         v\${VERSION}"
echo "Base model:      \$BASE_MODEL"
echo "Adapter model:   \$ADAPTER_MODEL"
echo "Max new tokens:  \$MAX_NEW_TOKENS"

# ── Run ───────────────────────────────────────────────────────────────────────
python inference.py \\
    --base_model    \$BASE_MODEL \\
    --adapter_model \$ADAPTER_MODEL \\
    --data_path     dataset/ \\
    --max_new_tokens \$MAX_NEW_TOKENS \\
    --do_sample

echo "Finished at: \$(date)"
EOF

echo "Submitted inference array job for configs ${ARRAY_RANGE}."