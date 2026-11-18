#!/usr/bin/env bash
set -eo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

mode=${1:-both}
case "$mode" in
    both|baseline|nsr|check) ;;
    *) die "Usage: bash majob.sh [both|baseline|nsr|check]" ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$repo_root/NSR-baseline" || die "Cannot find NSR-baseline in $repo_root"
recipe=recipe/nsr/DAPO_baseline_1p5b.sh
[[ -r "$recipe" ]] || die "Cannot read $PWD/$recipe"

if [[ -n "${MA_JOB_DIR:-}" ]]; then
    shared=/home/ma-user/work/bucket-pangu-green/lihourun
    ray_data_home=${RAY_DATA_HOME:-$shared/NSR}
    hf_home=${HF_HOME:-$shared/data/hf_cache}
else
    ray_data_home=${RAY_DATA_HOME:-$repo_root}
    hf_home=${HF_HOME:-$ray_data_home/cache/huggingface}
fi
model_path=${MODEL_PATH:-$ray_data_home/models/Qwen2.5-Math-1.5B}
train_file=${TRAIN_FILE:-$ray_data_home/data/DAPO-Math-17k/data/dapo-math-17k.parquet}
test_file=${TEST_FILE:-$ray_data_home/data/AIME-2024/data/aime-2024.parquet}
runs_dir=${RUNS_DIR:-$ray_data_home/paper-runs}
conda_env=${CONDA_ENV:-nsr}
conda_sh=${CONDA_SH:-}
if [[ -z "$conda_sh" && "${CONDA_DEFAULT_ENV:-}" != "$conda_env" ]]; then
    if [[ -n "${MA_JOB_DIR:-}" ]]; then
        conda_sh=/home/ma-user/anaconda3/etc/profile.d/conda.sh
    else
        conda_base=$(conda info --base) || die "Cannot locate Conda. Activate $conda_env or set CONDA_SH."
        conda_sh="$conda_base/etc/profile.d/conda.sh"
    fi
fi

printf 'Repo: %s\nHF cache: %s\nNSR data: %s\nModel: %s\nTrain data: %s\nAIME data: %s\nRuns: %s\nConda: %s (%s)\n' \
    "$repo_root" "$hf_home" "$ray_data_home" "$model_path" "$train_file" "$test_file" \
    "$runs_dir" "${conda_sh:-active environment}" "$conda_env"

[[ -r "$model_path/config.json" ]] || die "Cannot read model config: $model_path/config.json"
[[ -r "$train_file" ]] || die "Cannot read training data: $train_file"
[[ -r "$test_file" ]] || die "Cannot read AIME data: $test_file"
if [[ -n "${MA_JOB_DIR:-}" ]]; then
    [[ -d "$hf_home" && -x "$hf_home" ]] || die "Cannot access HF cache: $hf_home"
fi
if [[ -n "$conda_sh" ]]; then
    [[ -r "$conda_sh" ]] || die "Cannot read conda.sh: $conda_sh (set CONDA_SH)"
fi
if [[ "$mode" == check ]]; then
    exit 0
fi

if [[ -n "$conda_sh" ]]; then
    source "$conda_sh"
    conda activate "$conda_env" || die "Cannot activate the NSR conda environment"
fi
set -u

export HF_HOME="$hf_home"
export HF_DATASETS_CACHE="$HF_HOME/hf_cache"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export NLTK_DATA="$HF_HOME/nltk_data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
# 复用仓库的导入路径隔离，避免 ModelArts 的 .pth 注入其他环境的包。
export PYTHONPATH="$PWD:$repo_root/NSR-main/baseline_4gpu/pathfix"
export BASELINE_ENV_ONLY=1

# Prefer the host driver as in LogKV. ModelArts can prepend a CUDA 11.4
# forward-compatibility driver that shadows the host libcuda for cu124 Torch.
cuda_library_dirs=(/usr/lib/x86_64-linux-gnu)
IFS=: read -r -a inherited_library_dirs <<< "${LD_LIBRARY_PATH:-}"
for lib_dir in "${inherited_library_dirs[@]}"; do
    case "${lib_dir%/}" in
        ""|/usr/lib/x86_64-linux-gnu|/usr/local/cuda-11.4/compat) ;;
        *) cuda_library_dirs+=("$lib_dir") ;;
    esac
done
printf -v LD_LIBRARY_PATH '%s:' "${cuda_library_dirs[@]}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"

export RAY_DATA_HOME="$ray_data_home" MODEL_PATH="$model_path"
export TRAIN_FILE="$train_file" TEST_FILE="$test_file"
export NNODES=1
export NCCL_NVLS_ENABLE=0 NCCL_NET_PLUGIN=none NCCL_IB_TIMEOUT=22
export NSR_LOG_CUDA_STARTUP=1
unset PYTORCH_CUDA_ALLOC_CONF

N_GPUS_PER_NODE=$(python3 verl/utils/cuda_diagnostics.py)
[[ "$N_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "GPU detection did not return a valid GPU count"
export N_GPUS_PER_NODE

local_tmp=$(mktemp -d /tmp/nsr.XXXXXXXX)
export TMPDIR="$local_tmp" RAY_TMPDIR="$local_tmp"
mkdir -p -- "$runs_dir"
batch_dir=$(mktemp -d -- "$runs_dir/run-XXXXXXXX") ||
    die "Cannot create a run directory under $runs_dir"
printf 'Ray temp: %s\nRun output: %s\n' "$RAY_TMPDIR" "$batch_dir"

run_training() {
    local label=$1 enable_nsr=$2
    local name="small-${label}-300"
    local run_dir="$batch_dir/$name"

    mkdir -- "$run_dir"
    printf 'Starting %s | output: %s\n' "$name" "$run_dir"
    NSR="$enable_nsr" STEPS=300 EXP_NAME="$name" RUN_DIR="$run_dir" VAL_ONLY=false \
        bash "$recipe" 2>&1 | tee "$run_dir/train.log"
}

case "$mode" in
    both) run_training baseline false; run_training nsr true ;;
    baseline) run_training baseline false ;;
    nsr) run_training nsr true ;;
esac
