"""Report CUDA startup from the process that actually imports PyTorch."""

import os
import subprocess
import sys
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


def main():
    visible = report_cuda("launcher")
    if visible is None:
        raise SystemExit("NSR CUDA preflight failed. See the launcher diagnostics above.")

    hosts = int(os.environ.get("MA_NUM_HOSTS") or 1)
    required = int(os.environ["N_GPUS_PER_NODE"])
    allocated = int(os.environ.get("MA_NUM_GPUS") or required)
    if hosts != 1 or min(allocated, visible) < required:
        raise SystemExit(
            f"Need one host with {required} GPUs. "
            f"Job has {hosts} host(s), {allocated} allocated GPUs, {visible} visible GPUs."
        )


if __name__ == "__main__":
    main()
