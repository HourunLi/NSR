#!/usr/bin/env bash

set -uo pipefail
umask 077

# =========================
# 基本配置
# =========================
NSR_ROOT="/home/lihourun/hsc/NSR-work"
REPO="$NSR_ROOT/NSR-main"
CONDA_SH="/home/lihourun/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="nsr"

# 每次检查间隔
CHECK_INTERVAL=60

# GPU空闲标准
MAX_MEMORY_USED_MIB=512
MAX_GPU_UTIL=5

# 同一组GPU连续空闲多少次才启动
REQUIRED_CONSECUTIVE=3

WATCH_DIR="$NSR_ROOT/gpu-watcher"
WATCH_LOG="$WATCH_DIR/watcher.log"
WATCH_PID="$WATCH_DIR/watcher.pid"
LOCK_FILE="$WATCH_DIR/watcher.lock"

mkdir -p "$WATCH_DIR"

# 所有输出写入持久日志
exec >>"$WATCH_LOG" 2>&1

echo
echo "=================================================="
echo "Watcher started: $(date '+%F %T')"
echo "PID: $$"
echo "Repository: $REPO"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Required consecutive checks: $REQUIRED_CONSECUTIVE"
echo "=================================================="

# 防止重复启动多个监控器
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock command not found"
    exit 1
fi

exec 9>"$LOCK_FILE"

if ! flock -n 9; then
    echo "ERROR: another watcher is already running"
    exit 1
fi

echo "$$" > "$WATCH_PID"

cleanup() {
    echo "Watcher stopping: $(date '+%F %T')"
    rm -f "$WATCH_PID"
}

trap cleanup EXIT
trap 'exit 143' TERM INT

# 激活conda环境
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda activation script not found: $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"

if ! conda activate "$CONDA_ENV"; then
    echo "ERROR: cannot activate conda environment: $CONDA_ENV"
    exit 1
fi

cd "$REPO" || exit 1

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# 检查启动器
if ! python -m baseline_4gpu.launch --help >/dev/null 2>&1; then
    echo "ERROR: baseline_4gpu launcher is unavailable"
    exit 1
fi

previous_selection=""
consecutive_count=0

while true; do
    now=$(date '+%F %T')

    # 当前账号已有训练时不再提交
    existing_training=$(
        pgrep -u "$(id -u)" -af \
          'recipe[.]dapo[.]main_dapo|baseline_4gpu[.]launch run|ray::TaskRunner|ray::WorkerDict' \
          2>/dev/null || true
    )

    if [ -n "$existing_training" ]; then
        echo "[$now] Existing training process found; waiting:"
        printf '%s\n' "$existing_training"

        previous_selection=""
        consecutive_count=0
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # 获取GPU状态
    if ! gpu_info=$(
        nvidia-smi \
          --query-gpu=index,uuid,memory.used,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null
    ); then
        echo "[$now] nvidia-smi query failed; retrying"
        previous_selection=""
        consecutive_count=0
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # 获取GPU计算进程
    process_info=$(
        nvidia-smi \
          --query-compute-apps=gpu_uuid,pid \
          --format=csv,noheader,nounits 2>/dev/null || true
    )

    unset gpu_has_process
    declare -A gpu_has_process=()

    while IFS=',' read -r process_uuid process_pid; do
        process_uuid="${process_uuid//[[:space:]]/}"
        process_pid="${process_pid//[[:space:]]/}"

        if [ -n "$process_uuid" ] && [ -n "$process_pid" ]; then
            gpu_has_process["$process_uuid"]=1
        fi
    done <<< "$process_info"

    free_gpus=()

    while IFS=',' read -r gpu_index gpu_uuid memory_used gpu_util; do
        gpu_index="${gpu_index//[[:space:]]/}"
        gpu_uuid="${gpu_uuid//[[:space:]]/}"
        memory_used="${memory_used//[[:space:]]/}"
        gpu_util="${gpu_util//[[:space:]]/}"

        if ! [[ "$memory_used" =~ ^[0-9]+$ ]] ||
           ! [[ "$gpu_util" =~ ^[0-9]+$ ]]; then
            continue
        fi

        has_process=0
        if [ -n "${gpu_has_process[$gpu_uuid]+x}" ]; then
            has_process=1
        fi

        if [ "$memory_used" -le "$MAX_MEMORY_USED_MIB" ] &&
           [ "$gpu_util" -le "$MAX_GPU_UTIL" ] &&
           [ "$has_process" -eq 0 ]; then
            free_gpus+=("$gpu_index")
        fi
    done <<< "$gpu_info"

    echo "[$now] Free GPUs: ${free_gpus[*]:-none}"

    if [ "${#free_gpus[@]}" -lt 4 ]; then
        previous_selection=""
        consecutive_count=0
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # 选择前4张空闲卡
    current_selection="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]},${free_gpus[3]}"

    if [ "$current_selection" = "$previous_selection" ]; then
        consecutive_count=$((consecutive_count + 1))
    else
        previous_selection="$current_selection"
        consecutive_count=1
    fi

    echo "[$now] Candidate GPUs: $current_selection"
    echo "[$now] Consecutive free checks: $consecutive_count/$REQUIRED_CONSECUTIVE"

    if [ "$consecutive_count" -lt "$REQUIRED_CONSECUTIVE" ]; then
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # 创建唯一实验名
    timestamp=$(date '+%Y%m%d-%H%M%S')
    gpu_suffix="${current_selection//,/}"
    run_name="DAPO7B-paper4-auto-${timestamp}-g${gpu_suffix}"

    export CUDA_VISIBLE_DEVICES="$current_selection"

    launch_log="$WATCH_DIR/${run_name}.launcher.log"
    run_dir="$NSR_ROOT/paper-runs/$run_name"

    echo "=================================================="
    echo "[$(date '+%F %T')] Four stable free GPUs found"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Run name: $run_name"
    echo "Launcher log: $launch_log"
    echo "=================================================="

    # 不使用 --skip-gpu-check：
    # 启动器在真正启动前会再次确认，防止检查后被其他任务抢占。
    python -u -m baseline_4gpu.launch run \
      --name "$run_name" \
      --n-gpus 4 \
      --skip-disk-check \
      >"$launch_log" 2>&1 &

    launcher_pid=$!
    echo "Launcher PID: $launcher_pid"

    wait "$launcher_pid"
    exit_code=$?

    echo "[$(date '+%F %T')] Launcher exited with code $exit_code"

    if [ -d "$run_dir" ]; then
        # 实验目录已经创建，说明训练实际启动过。
        # 无论最后成功还是失败，都不自动无限重试。
        echo "Run directory exists: $run_dir"
        echo "Training log: $run_dir/train.log"
        exit "$exit_code"
    fi

    # 没有创建实验目录，通常表示启动瞬间GPU又被抢占。
    # 继续等待下一次机会。
    echo "Training was rejected before run directory creation; resume monitoring."

    previous_selection=""
    consecutive_count=0
    sleep "$CHECK_INTERVAL"
done