#!/usr/bin/env bash
set -eo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage="Usage: bash majob.sh [both|baseline|nsr|check] [1.5b|7b]"
(( $# <= 2 )) || die "$usage"
mode=${1:-both}
case "$mode" in
    both|baseline|nsr|check) ;;
    *) die "$usage" ;;
esac
case "${2:-1.5b}" in
    1.5b) profile=1p5b ;;
    7b) profile=7b ;;
    *) die "$usage" ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$repo_root/NSR-baseline" || die "Cannot find NSR-baseline in $repo_root"

nnodes=${MA_NUM_HOSTS:-${VC_WORKER_NUM:-1}}
[[ "$nnodes" =~ ^[1-9][0-9]*$ ]] || die "Invalid ModelArts node count: $nnodes"

if [[ -n "${MA_JOB_DIR:-}" ]]; then
    shared=/home/ma-user/work/bucket-pangu-green/lihourun
    ray_data_home=${RAY_DATA_HOME:-$shared/NSR}
    hf_home=${HF_HOME:-$shared/data/hf_cache}
else
    ray_data_home=${RAY_DATA_HOME:-$repo_root}
    hf_home=${HF_HOME:-$ray_data_home/cache/huggingface}
fi
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

printf 'Repo: %s\nHF cache: %s\nNSR data: %s\nRuns: %s\nConda: %s (%s)\n' \
    "$repo_root" "$hf_home" "$ray_data_home" \
    "$runs_dir" "${conda_sh:-active environment}" "$conda_env"

if [[ -n "${MA_JOB_DIR:-}" ]]; then
    [[ -d "$hf_home" && -x "$hf_home" ]] || die "Cannot access HF cache: $hf_home"
fi
if [[ -n "$conda_sh" ]]; then
    [[ -r "$conda_sh" ]] || die "Cannot read conda.sh: $conda_sh (set CONDA_SH)"
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
export RAY_DATA_HOME="$ray_data_home"
for override in MODEL_PATH TRAIN_FILE TEST_FILE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH; do
    # Keep empty path/length overrides equivalent to an unset environment variable.
    if [[ -z "${!override:-}" ]]; then
        unset "$override"
    fi
done

# Prefer the host CUDA driver. ModelArts can prepend a CUDA 11.4
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

if (( nnodes == 1 )); then
    export NCCL_NVLS_ENABLE=0 NCCL_NET_PLUGIN=none NCCL_IB_TIMEOUT=22
fi
export NSR_LOG_CUDA_STARTUP=1
export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF

if [[ "$mode" == check ]]; then
    python3 recipe/nsr/majob_preflight.py "$profile" --check
    exit 0
fi
launch_settings=$(python3 recipe/nsr/majob_preflight.py "$profile")
read -r run_prefix steps max_batched_tokens gpu_count <<< "$launch_settings"

local_tmp=$(mktemp -d /tmp/nsr.XXXXXXXX)
export TMPDIR="$local_tmp" RAY_TMPDIR="$local_tmp"

ray_init_args=()
if (( nnodes == 1 )); then
    export RAY_ADDRESS=local
    ray_init_args+=("ray_init.num_cpus=$((gpu_count + 2))")
else
    rank=${VC_TASK_INDEX:-}
    [[ "$rank" =~ ^(0|[1-9][0-9]*)$ ]] && (( rank < nnodes )) ||
        die "Multi-node training needs a VC_TASK_INDEX between 0 and $((nnodes - 1))"
    IFS=, read -r -a node_hosts <<< "${VC_WORKER_HOSTS:-}"
    (( ${#node_hosts[@]} == nnodes )) && [[ -n "${node_hosts[0]}" && -n "${node_hosts[rank]}" ]] ||
        die "VC_WORKER_HOSTS must list the $nnodes ModelArts nodes in rank order"
    ray_port=${NSR_RAY_PORT:-6379}
    ray_timeout=${NSR_RAY_START_TIMEOUT:-300}
    [[ "$ray_port" =~ ^[1-9][0-9]{0,4}$ ]] && (( ray_port <= 65535 )) ||
        die "NSR_RAY_PORT must be an integer between 1 and 65535"
    [[ "$ray_timeout" =~ ^[1-9][0-9]*$ ]] || die "NSR_RAY_START_TIMEOUT must be a positive number of seconds"
    node_host=${MA_CURRENT_IP:-${node_hosts[rank]}}
    node_ip=$(python3 -c 'import socket, sys; print(socket.gethostbyname(sys.argv[1]))' "$node_host") ||
        die "Cannot resolve this node's Ray IP from $node_host"
    export NSR_RAY_NODE_IP="$node_ip" RAY_USAGE_STATS_ENABLED=0
    if (( rank == 0 )); then
        ray_head_ip=$node_ip
    else
        export RAY_ADDRESS="${node_hosts[0]}:$ray_port"
        ray_head_ip=$(python3 recipe/nsr/majob_ray.py wait-head "$ray_timeout")
    fi
    export RAY_ADDRESS="$ray_head_ip:$ray_port"
    printf 'Ray cluster: node %s/%s at %s, %s GPUs per node, head %s\n' \
        "$rank" "$nnodes" "$NSR_RAY_NODE_IP" "$gpu_count" "$RAY_ADDRESS"

    ray_args=(--block "--node-ip-address=$NSR_RAY_NODE_IP" "--num-gpus=$gpu_count" "--num-cpus=$((gpu_count + 2))")
    if (( rank == 0 )); then
        ray_args+=(--head "--port=$ray_port" --include-dashboard=false)
    else
        ray_args+=("--address=$RAY_ADDRESS")
    fi
    ray start "${ray_args[@]}" &
    ray_pid=$!
    # Ray's --block process owns this node's Ray services and reaps them on SIGTERM.
    trap 'kill -TERM "$ray_pid" 2>/dev/null || true; wait "$ray_pid" 2>/dev/null || true' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if (( rank != 0 )); then
        python3 recipe/nsr/majob_ray.py wait-stop "$ray_pid"
        exit 0
    fi
    timeout --kill-after=5s "$((ray_timeout + 10))" \
        python3 recipe/nsr/majob_ray.py wait-workers "$ray_timeout" "$ray_pid" "$nnodes" "$gpu_count" ||
        die "Ray cluster startup failed. See the Ray logs on all ModelArts nodes."
fi

mkdir -p -- "$runs_dir"
batch_dir=$(mktemp -d -- "$runs_dir/run-XXXXXXXX") ||
    die "Cannot create a run directory under $runs_dir"
printf 'Ray temp: %s\nRun output: %s\n' "$RAY_TMPDIR" "$batch_dir"

run_training() {
    local label=$1 enable_nsr=$2
    local name="${run_prefix}-${label}-${steps}"
    local run_dir="$batch_dir/$name"

    mkdir -p -- "$run_dir/validation" "$run_dir/checkpoints"
    printf 'Starting %s | output: %s\n' "$name" "$run_dir"
    EXP_NAME="$name" RUN_DIR="$run_dir" \
        python3 -m recipe.dapo.main_dapo "majob=$profile" \
        "actor_rollout_ref.actor.NSR=$enable_nsr" \
        "actor_rollout_ref.rollout.max_num_batched_tokens=$max_batched_tokens" \
        "trainer.nnodes=$nnodes" "trainer.n_gpus_per_node=$gpu_count" "${ray_init_args[@]}" \
        2>&1 | tee "$run_dir/train.log"
}

case "$mode" in
    both) run_training baseline false; run_training nsr true ;;
    baseline) run_training baseline false ;;
    nsr) run_training nsr true ;;
esac
