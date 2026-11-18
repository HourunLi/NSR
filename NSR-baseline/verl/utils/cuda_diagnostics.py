"""Report CUDA startup from the process that actually imports PyTorch."""

import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path


def report_cuda(role):
    prefix = f"[NSR CUDA {role} pid={os.getpid()}]"

    def log(message):
        print(f"{prefix} {message}", flush=True)

    log(f"python={sys.executable} version={sys.version.split()[0]} prefix={sys.prefix}")
    log(f"sitecustomize={getattr(sys.modules.get('sitecustomize'), '__file__', None)}")
    for name in ("CONDA_PREFIX", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        log(f"{name}={os.getenv(name)!r}")

    if role == "worker":
        try:
            import ray

            log(f"ray={ray.__file__} gpu_ids={ray.get_gpu_ids() if ray.is_initialized() else None}")
        except Exception as error:
            log(f"Ray GPU IDs unavailable: {type(error).__name__}: {error}")
    else:
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            versions = ", ".join(sorted(set(driver.stdout.splitlines())))
            if driver.returncode == 0 and versions:
                log(f"NVIDIA driver={versions}")
            else:
                log(f"nvidia-smi exit={driver.returncode}: {(driver.stderr or driver.stdout).strip()}")
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"nvidia-smi unavailable: {error}")

    try:
        import torch

        log(f"torch={torch.__file__} version={torch.__version__} CUDA build={torch.version.cuda}")
        visible = torch.cuda.device_count()
        log(f"visible GPUs={visible}")
        torch.cuda.init()
        log("CUDA init OK")
        return visible
    except Exception as error:
        log(f"CUDA init failed: {type(error).__name__}: {error}")
        return None
    finally:
        try:
            with Path("/proc/self/maps").open(encoding="utf-8", errors="replace") as maps:
                libs = {
                    line.split(maxsplit=5)[-1].strip()
                    for line in maps
                    if any(name in line for name in ("/libcuda.so", "/libcudart.so", "/libnccl.so"))
                }
        except OSError as error:
            log(f"Cannot read loaded CUDA libraries: {error}")
        else:
            log(f"loaded CUDA libraries: {', '.join(sorted(libs)) or '(none)'}")


def select_gpu_count(minibatch=32, train_batch=128, gen_batch=384, actor_sp=1, ref_sp=1, rollout_tp=1):
    with redirect_stdout(sys.stderr):
        visible = report_cuda("launcher")
    if visible is None:
        raise SystemExit("NSR CUDA preflight failed. See the launcher diagnostics above.")

    try:
        hosts = int(os.environ.get("MA_NUM_HOSTS") or os.environ.get("VC_WORKER_NUM") or 1)
        allocated = int(os.environ.get("MA_NUM_GPUS") or visible)
        requested = int(os.environ["N_GPUS_PER_NODE"]) if os.environ.get("N_GPUS_PER_NODE") else None
    except ValueError as error:
        raise SystemExit("The ModelArts node count, MA_NUM_GPUS and N_GPUS_PER_NODE must be integers.") from error
    if hosts < 1:
        raise SystemExit(f"The ModelArts node count must be positive, got {hosts}.")
    available = min(allocated, visible)
    if available < 1:
        raise SystemExit(f"No usable GPUs: {allocated} allocated, {visible} CUDA-visible.")

    # Ray splits generation and training batches across GPUs. FSDP normalizes
    # the actor's PPO minibatch across its data-parallel groups.
    compatible = [
        gpus for gpus in range(1, available + 1)
        if (world_size := gpus * hosts) % actor_sp == 0 and world_size % ref_sp == 0
        and world_size % rollout_tp == 0 and minibatch % (world_size // actor_sp) == 0
        and train_batch % world_size == 0 and gen_batch % world_size == 0
    ]
    if requested is not None and requested not in compatible:
        raise SystemExit(
            f"N_GPUS_PER_NODE={requested} is incompatible with this profile on {hosts} node(s) "
            f"with {available} usable GPUs per node. Valid counts: {compatible}."
        )
    if not compatible:
        raise SystemExit(
            f"No compatible GPU count for this profile on {hosts} node(s) with {available} usable GPUs per node "
            f"(actor sequence parallelism={actor_sp}, reference sequence parallelism={ref_sp}, "
            f"rollout tensor parallelism={rollout_tp})."
        )
    selected = requested if requested is not None else compatible[-1]
    source = "override" if requested is not None else "auto"
    print(
        f"[NSR CUDA launcher] selected GPUs={selected} per node ({source}), total={selected * hosts}; "
        f"CUDA-visible={visible}, allocated={allocated}, usable={available}",
        file=sys.stderr,
        flush=True,
    )
    return selected


if __name__ == "__main__":
    print(select_gpu_count(*map(int, sys.argv[1:])))
