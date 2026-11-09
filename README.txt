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
