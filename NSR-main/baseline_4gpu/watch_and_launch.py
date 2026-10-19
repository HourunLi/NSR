"""Wait for four stably idle GPUs, then launch exactly one four-GPU baseline run."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .launch import owned_training_processes

try:
    import fcntl
except ModuleNotFoundError:  # Allows parser unit tests on non-Linux delivery hosts.
    fcntl = None


def stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def parse_gpu_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
        index, uuid, name, memory, util = parts
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory": int(memory), "util": int(util)})
    return rows


def query_gpus(memory_limit, util_limit):
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = parse_gpu_rows(raw)
    apps = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if apps.returncode:
        raise RuntimeError(f"Cannot query GPU processes: {apps.stderr.strip()}")
    app_uuids = {line.split(",", 1)[0].strip() for line in apps.stdout.splitlines() if line.strip()}
    idle = [row for row in rows if row["uuid"] not in app_uuids
            and row["memory"] <= memory_limit and row["util"] <= util_limit]
    return rows, sorted(idle, key=lambda row: row["index"])


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-limit-mib", type=int, default=512)
    parser.add_argument("--util-limit", type=int, default=5)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--name-prefix", default="DAPO7B-paper4-auto")
    parser.add_argument("--skip-disk-check", action="store_true")
    args = parser.parse_args()
    if fcntl is None:
        parser.error("GPU watcher requires Linux")
    if args.interval < 5 or args.stable_samples < 1:
        parser.error("interval must be >=5 seconds and stable-samples must be >=1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", args.name_prefix):
        parser.error("Use a simple name prefix of at most 40 characters")
    root = Path(os.environ.get("NSR_ROOT", "/home/lihourun/hsc/NSR-work")).resolve()
    repo = Path(__file__).resolve().parent.parent
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock_stream = (logs / "gpu-watcher.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{stamp()} STOPPED: another GPU watcher for this account is already running", flush=True)
        return 2
    print(f"{stamp()} watcher started: need 4 GPUs, memory<={args.memory_limit_mib} MiB, "
          f"util<={args.util_limit}%, stable {args.stable_samples} samples", flush=True)
    last_choice = None
    stable = 0
    while True:
        try:
            old = owned_training_processes()
            rows, idle = query_gpus(args.memory_limit_mib, args.util_limit)
            summary = " ".join(f"{r['index']}:{r['memory']}MiB/{r['util']}%" for r in rows)
            if old:
                stable, last_choice = 0, None
                print(f"{stamp()} waiting: this account has {len(old)} training process(es); {summary}", flush=True)
                time.sleep(args.interval)
                continue
            choice = tuple(row["index"] for row in idle[:4]) if len(idle) >= 4 else None
            if choice is None:
                stable, last_choice = 0, None
                print(f"{stamp()} waiting: {len(idle)}/4 idle; {summary}", flush=True)
                time.sleep(args.interval)
                continue
            if choice == last_choice:
                stable += 1
            else:
                last_choice, stable = choice, 1
            print(f"{stamp()} candidate={choice}, stable={stable}/{args.stable_samples}; {summary}", flush=True)
            if stable < args.stable_samples:
                time.sleep(args.interval)
                continue
            suffix = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            name = f"{args.name_prefix}-{suffix}"
            launch_log = logs / f"{name}-launcher.log"
            run_dir = root / "paper-runs" / name
            command = [sys.executable, "-u", "-m", "baseline_4gpu.launch", "run",
                       "--name", name, "--n-gpus", "4"]
            if args.skip_disk_check:
                command.append("--skip-disk-check")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, choice))
            print(f"{stamp()} launching {name} on physical GPUs {env['CUDA_VISIBLE_DEVICES']}", flush=True)
            print(f"{stamp()} launcher log: {launch_log}", flush=True)
            with launch_log.open("x", encoding="utf-8") as output:
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            (logs / "gpu-watcher-last-launch.pid").write_text(str(child.pid) + "\n", encoding="utf-8")
            # A race can make the launcher reject the GPUs. Retry only if training never started.
            for _ in range(180):
                driver_file = run_dir / "driver.pid"
                if driver_file.is_file():
                    driver = int(driver_file.read_text().strip())
                    if process_alive(driver):
                        print(f"{stamp()} submitted successfully: launcher PID={child.pid}, driver PID={driver}", flush=True)
                        print(f"{stamp()} training log: {run_dir / 'train.log'}", flush=True)
                        return 0
                code = child.poll()
                if code is not None:
                    if run_dir.exists():
                        print(f"{stamp()} launcher exited code={code} after creating {run_dir}; not retrying automatically", flush=True)
                        return code or 1
                    print(f"{stamp()} launch race/config rejection (code={code}); resuming GPU wait", flush=True)
                    stable, last_choice = 0, None
                    break
                time.sleep(2)
            else:
                print(f"{stamp()} launcher did not expose a driver within 6 minutes; inspect {launch_log}", flush=True)
                return 1
        except KeyboardInterrupt:
            print(f"{stamp()} watcher interrupted before submission", flush=True)
            return 130
        except Exception as exc:
            print(f"{stamp()} check error: {type(exc).__name__}: {exc}; retrying", flush=True)
            stable, last_choice = 0, None
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
