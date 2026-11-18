# ModelArts 1.5B 训练（单机八卡）
# 将仓库作为代码目录提交，以下命令假设任务中的目录名为 NSR。
# 启动前检查代码、缓存、模型配置、Parquet 文件和 Conda 初始化脚本。
cd "${MA_JOB_DIR}/NSR" && bash majob.sh check

# ModelArts 启动命令，先跑 baseline 300 步，成功后再跑 NSR 300 步。
cd "${MA_JOB_DIR}/NSR" && bash majob.sh both

# 也可以用 bash majob.sh baseline 或 bash majob.sh nsr 单独提交。
# 默认激活 nsr，Conda 安装位置不同可设置 CONDA_SH。
# 共享盘 NSR 根目录为 /home/ma-user/work/bucket-pangu-green/lihourun/NSR，
# 对应开发机的 /data/lihourun/NSR。路径不同可设置 RAY_DATA_HOME。
# HF 缓存沿用 niahLogKV 的 /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache。
# 离线运行前准备本地模型权重及训练和验证 Parquet，AIME 文件须有 960 行。
# 每次启动会在 RAY_DATA_HOME/paper-runs 下新建 run-XXXXXXXX，两个实验各有一个子目录。
# 可用 RUNS_DIR 指定输出根目录。重跑会从初始模型开始训练。

# 开发机也可以用同一个启动脚本。未设置 MA_JOB_DIR 时，
# 数据根目录默认是脚本所在的仓库目录，HF 缓存默认是该数据根目录下的 cache/huggingface。
# 已激活的 nsr 环境会直接复用。
cd /data/lihourun/NSR
conda activate nsr
bash majob.sh check
bash majob.sh both


# 1.5B 模型下载
mkdir -p "/data/lihourun/NSR/models/Qwen2.5-Math-1.5B"

HF_HOME="/data/lihourun/NSR/cache/huggingface" \
hf download \
  Qwen/Qwen2.5-Math-1.5B \
  --local-dir "/data/lihourun/NSR/models/Qwen2.5-Math-1.5B"


# 1.5B 训练配置（复用 nsr 环境，单机八卡）
conda activate nsr
cd "/data/lihourun/NSR/NSR-baseline"
unset PYTORCH_CUDA_ALLOC_CONF

export RAY_DATA_HOME="/data/lihourun/NSR"
export MODEL_PATH="${RAY_DATA_HOME}/models/Qwen2.5-Math-1.5B"
export TRAIN_FILE="${RAY_DATA_HOME}/data/DAPO-Math-17k/data/dapo-math-17k.parquet"
export TEST_FILE="${RAY_DATA_HOME}/data/AIME-2024/data/aime-2024.parquet"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS_PER_NODE=8
export NNODES=1

# baseline（300 step）
NSR=false STEPS=300 EXP_NAME=small-baseline-300 \
bash recipe/nsr/DAPO_baseline_1p5b.sh

# NSR（300 step，baseline 完成后运行）
NSR=true STEPS=300 EXP_NAME=small-nsr-300 \
bash recipe/nsr/DAPO_baseline_1p5b.sh


# 以下为原有 7B 配置
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
