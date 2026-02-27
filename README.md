# DontForgetAboutSafety
[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-FFD21E?logo=huggingface&logoColor=black)](https://hf.co/collections/cwoodhayes/dontforgetaboutsafety)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

**Authors**: Andnet DeBoer, Conor Hayes, Praneeth Reddy Mallupalli, Robert Zhu (equal contribution)

Fine-tuning LLM's to improve math word problem performance without losing prior alignment on AI safety.
Fine-tuned using [GSM8K](https://huggingface.co/datasets/openai/gsm8k), evaluated using [AILuminate](https://mlcommons.org/benchmarks/ailuminate/).

HuggingFace Model Collection: https://hf.co/collections/tutor369

## 🏃‍♂️ Run Instructions
Run the following to reproduce the results shown in the paper:
```bash
./reproduce.sh
```

- Complete hyperparameter configurations, and evaluation results are documented in the [project report](report.pdf). 

- Parallel training and inference across all 10 configurations can be launched on a SLURM-based GPU cluster using the provided [`slurm_train_job.sh`](slurm_train_job.sh) and [`slurm_inference_job.sh`](slurm_inference_job.sh) scripts.