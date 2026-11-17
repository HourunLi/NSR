# 环境配置
conda env create \
  -n nsr \
  -f "/data/lihourun/NSR/manifests/environment-no-builds.yml"

conda activate nsr


# 模型下载
mkdir -p "/data/lihourun/NSR/cache/huggingface"
mkdir -p "/data/lihourun/NSR/models/Qwen2.5-Math-7B"

HF_HOME="/data/lihourun/NSR/cache/huggingface" \
hf download \
  Qwen/Qwen2.5-Math-7B \
  --local-dir "/data/lihourun/NSR/models/Qwen2.5-Math-7B"


# 提交任务
cd "/data/lihourun/NSR/NSR-baseline"
unset PYTORCH_CUDA_ALLOC_CONF

RAY_DATA_HOME="/data/lihourun/NSR" \
MODEL_PATH="/data/lihourun/NSR/models/Qwen2.5-Math-7B" \
RAY_ADDRESS=auto \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=4 \
bash recipe/nsr/DAPO_baseline_7b.sh




# Environment
conda env create -n nsr -f "/data/lihourun/NSR/manifests/environment-no-builds.yml"
conda activate nsr

# Validation data
python3 - <<'PY'
from pathlib import Path
import json
import pandas as pd

source = Path("/data/lihourun/NSR/data/AIME-2024/data/aime-2024.parquet")
data = pd.read_parquet(source)
keys = data["prompt"].map(
    lambda messages: json.dumps(list(messages), sort_keys=True, ensure_ascii=False)
)
data.loc[~keys.duplicated()].to_parquet(
    source.with_name("aime-2024-unique30.parquet"),
    index=False,
)
PY

# Training
cd "/data/lihourun/NSR/NSR-baseline"
unset PYTORCH_CUDA_ALLOC_CONF

RAY_DATA_HOME="/data/lihourun/NSR" \
MODEL_PATH="/data/lihourun/NSR/models/Qwen2.5-Math-7B" \
TRAIN_FILE="/data/lihourun/NSR/data/DAPO-Math-17k/data/dapo-math-17k.parquet" \
TEST_FILE="/data/lihourun/NSR/data/AIME-2024/data/aime-2024-unique30.parquet" \
RAY_ADDRESS=local    \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=1 \
N_GPUS_PER_NODE=8 \
EXP_NAME="DAPO_baseline_7b" \
WANDB_MODE=offline \
bash recipe/nsr/DAPO_baseline_7b_4096.sh
