# 环境配置
conda env create \
  -n nsr \
  -f ".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/manifests/environment-no-builds.yml"

conda activate nsr


# 模型下载
mkdir -p ".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/cache/huggingface"
mkdir -p ".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/models/Qwen2.5-Math-7B"

HF_HOME=".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/cache/huggingface" \
hf download \
  Qwen/Qwen2.5-Math-7B \
  --local-dir ".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/models/Qwen2.5-Math-7B"


# 提交任务
cd ".../NSR/NSR-baseline"
unset PYTORCH_CUDA_ALLOC_CONF

RAY_DATA_HOME=".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733" \
MODEL_PATH=".../NSR-7B-baseline-32gpu-COMPLETE-20260916-220733/models/Qwen2.5-Math-7B" \
RAY_ADDRESS=auto \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=4 \
bash recipe/nsr/DAPO_baseline_7b.sh
