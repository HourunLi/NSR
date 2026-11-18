# ModelArts 1.5B / 7B 训练（单机或多机，自动检测每节点 GPU 数）
# 将仓库作为代码目录提交，以下命令假设任务中的目录名为 NSR。
# 启动前用 check 检查 YAML、模型配置、数据路径和 AIME 行数，无需 GPU。
cd "${MA_JOB_DIR}/NSR" && bash majob.sh check

# 不传第二个参数时使用 Qwen/Qwen2.5-Math-1.5B。both 先跑 baseline，再跑 NSR。
cd "${MA_JOB_DIR}/NSR" && bash majob.sh both

# 使用 Qwen/Qwen2.5-Math-7B。请在下方下载步骤完成后执行。
cd "${MA_JOB_DIR}/NSR" && bash majob.sh check 7b
cd "${MA_JOB_DIR}/NSR" && bash majob.sh both 7b

# 也可以用 bash majob.sh baseline 7b 或 bash majob.sh nsr 7b 单独提交。
# 两节点共 16 卡时，选择 2 个实例、每实例 8 张 GPU，使用同一条自定义启动命令。
# majob.sh 根据 MA_NUM_HOSTS、VC_TASK_INDEX 和 VC_WORKER_HOSTS 组建 Ray 集群。
# 0 号节点在两台机器各有 8 张 GPU 加入 Ray 后提交训练。其他节点加入并等待。
# 确保各节点可以访问同路径的模型、数据和共享 RUNS_DIR（FSDP checkpoint 由多节点共同写入）。
# N_GPUS_PER_NODE 表示每节点卡数，可用 N_GPUS_PER_NODE=8 显式要求两节点各 8 卡。
# Ray 默认使用 6379 端口，等待节点加入的上限为 300 秒，可通过 NSR_RAY_PORT 和 NSR_RAY_START_TIMEOUT 调整。
# Hydra 配置位于 NSR-baseline/recipe/dapo/config/majob/。
# 1p5b.yaml 和 7b.yaml 指定模型目录、长度、batch size 和并行度。
# common.yaml 指定数据默认路径、训练步数（300）、优化器、评估及保存频率。
# MODEL_PATH、TRAIN_FILE、TEST_FILE 可覆盖本地路径。配置中的训练步数可由各 YAML 覆盖。
# 默认激活 nsr，Conda 安装位置不同可设置 CONDA_SH。
# 作业日志中的 [NSR CUDA launcher ...] 记录实际 Python、Torch、NVIDIA 驱动和 CUDA 初始化结果。
# Ray worker 无法选择 GPU 后端时，[NSR CUDA worker ...] 还会记录该进程分配到的 GPU ID。
# 启动脚本优先查找系统驱动，并从运行时搜索路径中移除 ModelArts 注入的 CUDA 11.4 compat 目录。
# majob.sh 保留 CUDA_VISIBLE_DEVICES，按实际可用卡数、batch size 和并行度选择 GPU 数量。
# 例如单节点有 6 张可用卡时两个预设都会选 4 张。可用 N_GPUS_PER_NODE=4 显式指定卡数。
# 7B 复用 DAPO_baseline_7b.sh 的 4096-token 训练参数，使用 2 路序列并行。
# 卡数通过检查后仍需满足显存要求。
# 共享盘 NSR 根目录为 /home/ma-user/work/bucket-pangu-green/lihourun/NSR，
# 对应开发机的 /data/lihourun/NSR。路径不同可设置 RAY_DATA_HOME。
# ModelArts 的 HF 缓存默认位于 /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache。
# 如需使用其他缓存目录，可设置 HF_HOME。
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
bash majob.sh check 7b
bash majob.sh both 7b


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
