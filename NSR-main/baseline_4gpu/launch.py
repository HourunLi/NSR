"""Read-only preflight and fresh-run launcher. Never stops an existing job."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

from .protocol import SYSTEM_PROMPT, hydra_args, settings, validate_aime_rows, with_answer_system

GIB = 1024 ** 3
MIN_FREE_GIB = 300  # Conservative budget for two full checkpoints, exports and working files.


def existing_parent(path):
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(root, repo, name, n_gpus):
    env = os.environ.copy()
    if n_gpus != 8:
        env.pop("RAY_ADDRESS", None)  # Local 2/4-GPU mode starts its own Ray.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.update({
        "NSR_ROOT": str(root), "VERL_HOME": str(root), "N_GPUS": str(n_gpus), "NNODES": "4" if n_gpus == 8 else "1",
        "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1",
        "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "cache/huggingface"),
        "HF_DATASETS_CACHE": str(root / "cache/huggingface/datasets"),
        "TORCH_HOME": str(root / "cache/torch"),
        "VLLM_CACHE_ROOT": str(root / "cache/vllm"),
        "CUDA_CACHE_PATH": str(root / "cache/cuda"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "XDG_CACHE_HOME": str(root / "cache/xdg"),
        "TMPDIR": str(root.parent / "tp"),
        "RAY_TMPDIR": str(root.parent / "rp" / hashlib.sha256(name.encode()).hexdigest()[:8]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
    })
    return env


def check_data(root):
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer
    from .dataset import AnswerFormatDataset

    model = root / "models/Qwen2.5-Math-7B"
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type != "qwen2" or config.max_position_embeddings < 10240:
        raise ValueError("Need Qwen2 and model context >= 10240. No model config is changed by this launcher.")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    for filename in weights:
        if not (model / filename).is_file() or (model / filename).stat().st_size == 0:
            raise ValueError(f"Missing/empty weight shard: {filename}")
    total = sum((model / filename).stat().st_size for filename in weights)
    print(f"Model: {len(weights)} shards, {total/GIB:.2f} GiB, context={config.max_position_embeddings}")
    # Check the actual custom dataset method, without loading the full training dataset.
    adapter = object.__new__(AnswerFormatDataset)
    adapter.prompt_key, adapter.image_key, adapter.video_key = "prompt", "images", "videos"
    examples = {}
    for filename in ("dapo-math-17k.parquet", "aime-2024.parquet"):
        file = pq.ParquetFile(root / "data" / filename)
        row = next(file.iter_batches(batch_size=1)).to_pylist()[0]
        if "Answer:" not in str(row["prompt"]):
            raise ValueError(f"Unexpected answer instruction in {filename}")
        if row["data_source"] != "math_dapo" and not row["data_source"].startswith("aime"):
            raise ValueError(f"Unexpected scoring route in {filename}: {row['data_source']}")
        built = adapter._build_messages(dict(row))
        assert built == with_answer_system(row["prompt"])
        rendered = tokenizer.apply_chat_template(built, tokenize=False, add_generation_prompt=True)
        assert SYSTEM_PROMPT in rendered
        if r"put your final answer within \boxed{}" in rendered:
            raise ValueError("Conflicting automatic system message is still present")
        examples[filename] = rendered
        print(f"Dataset {filename}: {file.metadata.num_rows} rows")
        if filename.startswith("dapo") and (file.metadata.num_rows // 1536) * 3 < 300 * 10:
            raise ValueError("Three epochs may not provide the configured 300-step generation budget")
    val = pq.read_table(root / "data/aime-2024.parquet", columns=["prompt", "data_source"]).to_pylist()
    validate_aime_rows(val)
    print("AIME: 30 distinct questions x 32 rows x n=1 = 960 samples per evaluation")
    # Verify every distinct validation prompt, not just its first row.
    for row in val:
        with_answer_system(row["prompt"])
    return examples


def gpu_status(env, n_gpus):
    ids = env["CUDA_VISIBLE_DEVICES"].split(",")
    if len(ids) != n_gpus or len(set(ids)) != n_gpus or not all(i.isdigit() for i in ids):
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain {n_gpus} distinct numeric GPU IDs")
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    devices = {}
    for line in raw.splitlines():
        index, uuid, name, used = [part.strip() for part in line.split(",")]
        devices[index] = (uuid, name, used)
    if not set(ids) <= devices.keys():
        raise ValueError("A selected GPU does not exist")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    busy = []
    for index in ids:
        uuid, name, used = devices[index]
        print(f"Selected GPU {index}: {name}, memory used {used} MiB")
        if uuid in apps or float(used) > 512:
            busy.append(index)
    return busy


def owned_training_processes():
    """Return this user's other active DAPO/Ray training processes, excluding this launcher's ancestry."""
    raw = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,args="], text=True
    )
    rows = []
    parents = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        rows.append((pid, ppid, parts[2]))
        parents[pid] = ppid
    excluded = set()
    pid = os.getpid()
    while pid and pid not in excluded:
        excluded.add(pid)
        pid = parents.get(pid, 0)
    pattern = re.compile(
        r"recipe\.dapo\.main_dapo|baseline_4gpu\.launch\s+run|"
        r"ray::(?:TaskRunner|WorkerDict)|actor_rollout_(?:generate|update)"
    )
    return [(pid, command) for pid, _, command in rows if pid not in excluded and pattern.search(command)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run"])
    parser.add_argument("--name", default="DAPO7B-paper4-v2")
    parser.add_argument(
        "--n-gpus", type=int, choices=(2, 4, 8), default=4,
        help="Number of GPUs on this one node. Global/mini batch and rollout count are unchanged.",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Explicitly bypass launcher disk guards. Does not disable Ray's own disk monitor or checkpoint writes.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Explicitly permit selected GPUs with existing allocations. Does not free or isolate GPU memory.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.name):
        parser.error("Use a simple experiment name of at most 64 characters")
    repo = Path(__file__).resolve().parent.parent
    if not (repo / "recipe/dapo/main_dapo.py").is_file():
        raise ValueError("Place the baseline_4gpu folder directly inside NSR-main before running")
    root = Path(os.environ.get("NSR_ROOT", "/home/lihourun/hsc/NSR-work")).resolve()
    runs = Path(os.environ.get("BASELINE_RUNS", str(root / "paper-runs"))).resolve()
    run_dir = runs / args.name
    env = environment(root, repo, args.name, args.n_gpus)
    # The preflight itself does not allocate model weights on GPUs or start Ray.
    # Existing TMPDIR may be missing, so use the inherited temp configuration for this stage.
    examples = check_data(root)
    score = runpy.run_path(str(repo / "verl/utils/reward_score/math_dapo.py"))["compute_score"]
    if not score("Answer: 540", "540")["acc"]:
        raise ValueError("Installed reward scorer does not accept the expected Answer: format")
    versions = {}
    for package in ("torch", "transformers", "vllm", "ray", "datasets", "flash-attn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    print("Installed versions:", json.dumps(versions))
    cmd = [
        sys.executable, "-u", "-m", "recipe.dapo.main_dapo",
        *hydra_args(root, run_dir, n_gpus=args.n_gpus),
    ]
    # Validate against the user's installed Hydra/config version without starting training.
    compose_env = dict(env)
    compose_env.pop("TMPDIR", None)
    result = subprocess.run(cmd + ["--cfg", "job", "--resolve"], cwd=repo, env=compose_env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Hydra configuration check failed; nothing has been launched")
    print("Hydra configuration: OK")
    old_tasks = owned_training_processes()
    print("Existing training processes owned by this account:")
    if old_tasks:
        for pid, command in old_tasks:
            print(f"  PID {pid}: {command}")
    else:
        print("  none")
    busy = gpu_status(env, args.n_gpus)
    free = shutil.disk_usage(existing_parent(runs)).free / GIB
    runtime_disk = shutil.disk_usage(existing_parent(root))
    runtime_ok = runtime_disk.free >= max(20 * GIB, runtime_disk.total * 0.06)
    print(f"Checkpoint filesystem: {free:.1f} GiB free; conservative launch budget: {MIN_FREE_GIB} GiB")
    print(f"Runtime filesystem: {runtime_disk.free/GIB:.1f} GiB free")
    print(f"New experiment: {run_dir}")
    print("Selected GPUs busy:", busy or "none")
    print(f"This is one fresh 300-step run on {args.n_gpus} GPUs. It is NOT three independent replications.")
    blockers = []
    if old_tasks:
        blockers.append("This account already has another DAPO/Ray training process")
    disk_warnings = []
    if free < MIN_FREE_GIB:
        disk_warnings.append("Not enough conservative headroom for two complete saved checkpoints")
    if not runtime_ok:
        disk_warnings.append("Runtime/cache filesystem needs at least 20 GiB and 6% available")
    if args.skip_disk_check:
        print("UNSAFE OVERRIDE: launcher disk checks are explicitly skipped.")
        print("UNSAFE OVERRIDE: Ray disk monitoring and checkpoint writes remain enabled.")
        for reason in disk_warnings:
            print("DISK WARNING (ignored by launcher):", reason)
    else:
        blockers.extend(disk_warnings)
    if args.skip_gpu_check:
        print("UNSAFE OVERRIDE: selected-GPU occupancy checks are explicitly skipped.")
        if busy:
            print("GPU WARNING (ignored by launcher): selected GPUs already have allocations:", busy)
    if run_dir.exists():
        blockers.append("Experiment directory already exists; this fresh-run launcher never overwrites or resumes it")
    if not os.access(existing_parent(runs), os.W_OK):
        blockers.append("Checkpoint parent directory is not writable")
    if args.action == "check":
        if busy and not args.skip_gpu_check:
            print("GPU occupancy is informational for check mode. Do not start a second overlapping job.")
        for reason in blockers:
            print("BLOCKED:", reason)
        return 2 if blockers else 0
    if busy and not args.skip_gpu_check:
        blockers.append("Selected GPUs have existing processes/memory allocations; no processes will be stopped")
    if blockers:
        raise RuntimeError("; ".join(blockers))
    run_dir.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "RAY_TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved.yaml").write_text(result.stdout, encoding="utf-8")
    source_files = ["recipe/dapo/main_dapo.py", "recipe/dapo/dapo_ray_trainer.py", "verl/workers/fsdp_workers.py",
                    "verl/workers/actor/dp_actor.py", "verl/utils/reward_score/math_dapo.py", "verl/trainer/ppo/core_algos.py"]
    manifest = {"settings": settings(root, run_dir, n_gpus=args.n_gpus), "versions": versions, "command": cmd,
                "n_gpus": args.n_gpus,
                "disk_check_skipped": args.skip_disk_check,
                "gpu_check_skipped": args.skip_gpu_check,
                "busy_selected_gpus_at_launch": busy,
                "disk_free_gib_at_launch": free,
                "runtime_disk_free_gib_at_launch": runtime_disk.free / GIB,
                "source_sha256": {f: sha256(repo / f) for f in source_files},
                "gpu_visibility": env["CUDA_VISIBLE_DEVICES"], "prompt_examples": examples,
                "model_config": json.loads((root / "models/Qwen2.5-Math-7B/config.json").read_text())}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copytree(Path(__file__).parent, run_dir / "launcher_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Starting training. Log: {run_dir / 'train.log'}", flush=True)
    with (run_dir / "train.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        (run_dir / "driver.pid").write_text(str(process.pid) + "\n")
        code = process.wait()
    print(f"Training process exited with code {code}")
    if code != 0:
        return code
    final_model = run_dir / "checkpoints/global_step_300/actor/huggingface"
    if not list(final_model.glob("*.safetensors")) or not (run_dir / "validation/300.jsonl").is_file():
        raise RuntimeError("Process ended without the expected final step-300 checkpoint/evaluation; inspect train.log")
    print(f"Step-300 outputs found: {final_model}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
