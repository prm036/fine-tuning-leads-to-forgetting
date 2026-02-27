#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_train.sh  –  Submit 10 training jobs as a SLURM array
#
# Usage:
#   bash submit_train.sh                        # train all 10 configs
#   SLURM_ARRAY_RANGE=0-2 bash submit_train.sh  # train only configs 0-2
# ─────────────────────────────────────────────────────────────────────────────

ARRAY_RANGE="${SLURM_ARRAY_RANGE:-0-9}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --array=${ARRAY_RANGE}%10
#SBATCH --output=logs/train_%A_cfg%a.out
#SBATCH --error=logs/train_%A_cfg%a.err
#SBATCH --time=12:00:00
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

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Training config \${SLURM_ARRAY_TASK_ID}..."
python train.py --config_id \${SLURM_ARRAY_TASK_ID} --data_path dataset/ 

echo "Finished at: \$(date)"
EOF

echo "Submitted training array job for configs ${ARRAY_RANGE}."