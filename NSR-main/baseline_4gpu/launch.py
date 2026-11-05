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
import traceback

from .protocol import (LAYOUTS, SYSTEM_PROMPT, dataset_path, default_nnodes, hydra_args, settings,
                       validate_aime_rows, with_answer_system)

GIB = 1024 ** 3
MIN_FREE_GIB = 300  # Conservative budget for two full checkpoints, exports and working files.
# Ray and vLLM talk over Unix sockets inside their temp dirs; these filesystems cannot host them reliably.
SOCKETLESS_FS = ("fuse", "nfs", "cifs", "smb", "9p", "ceph", "glusterfs")
# Ray appends "/ray/session_<date>_<time>_<usec>_<pid>/sockets/plasma_store" (about 69 characters)
# and Linux limits a Unix socket path to 107 bytes.
MAX_RAY_TMPDIR_CHARS = 38


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


def mount_fstype(path, mounts_text=None):
    """Filesystem type of the mount that holds path (longest matching mount point in /proc/mounts)."""
    if mounts_text is None:
        try:
            mounts_text = Path("/proc/mounts").read_text()
        except OSError:
            return "unknown"
    path = str(path)
    best, fstype = -1, "unknown"
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), parts[1])
        prefix = point.rstrip("/")
        if (path == point or path.startswith(prefix + "/")) and len(prefix) > best:
            best, fstype = len(prefix), parts[2]
    return fstype


def socket_capable(fstype):
    return not fstype.startswith(SOCKETLESS_FS)


def temp_dirs(root, name):
    """TMPDIR and RAY_TMPDIR next to NSR_ROOT, unless that filesystem cannot host Unix sockets."""
    key = hashlib.sha256(name.encode()).hexdigest()[:8]
    base, note = root.parent, None
    override = os.environ.get("BASELINE_LOCAL_TMP")
    if override:
        base = Path(override).resolve()
    else:
        fstype = mount_fstype(existing_parent(base))
        if not socket_capable(fstype):
            base = Path("/tmp")
            note = (f"{root.parent} is on {fstype}, which cannot host Ray/vLLM Unix sockets; "
                    f"temp dirs go under {base} (set BASELINE_LOCAL_TMP to choose another local disk)")
    return base / "tp", base / "rp" / key, note


def environment(root, repo, name, n_gpus, nnodes):
    env = os.environ.copy()
    tmpdir, ray_tmpdir, _ = temp_dirs(root, name)
    if nnodes == 1:
        env.pop("RAY_ADDRESS", None)  # Single-node mode starts its own local Ray.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.update({
        "NSR_ROOT": str(root), "VERL_HOME": str(root), "N_GPUS": str(n_gpus), "NNODES": str(nnodes),
        "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1",
        # Inherited entries can shadow installed packages (e.g. a vendored grpc), which kills Ray's agents.
        "PYTHONPATH": str(repo),
        # User-site *.pth files are searched before the env's site-packages and can shadow its packages too.
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "cache/huggingface"),
        "HF_DATASETS_CACHE": str(root / "cache/huggingface/datasets"),
        "TORCH_HOME": str(root / "cache/torch"),
        "VLLM_CACHE_ROOT": str(root / "cache/vllm"),
        "CUDA_CACHE_PATH": str(root / "cache/cuda"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "XDG_CACHE_HOME": str(root / "cache/xdg"),
        "TMPDIR": str(tmpdir),
        "RAY_TMPDIR": str(ray_tmpdir),
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
        path = dataset_path(root, filename)
        print(f"Dataset path: {path}")
        file = pq.ParquetFile(path)
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
    val = pq.read_table(dataset_path(root, "aime-2024.parquet"), columns=["prompt", "data_source"]).to_pylist()
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


ENV_PROBE = r"""
import glob, importlib.metadata, importlib.util, json, os, site, sys
spec = importlib.util.find_spec("grpc")
origin = spec.origin if spec else None
try:
    version = importlib.metadata.version("grpcio")
except importlib.metadata.PackageNotFoundError:
    version = None
entry = None
if origin:
    entry = max((p for p in sys.path if p and origin.startswith(os.path.join(os.path.abspath(p), ""))),
                key=len, default=None)
sources = []
if entry:
    files = []
    for d in list(site.getsitepackages()) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else []):
        files += sorted(glob.glob(os.path.join(d, "*.pth")))
    for name in ("sitecustomize", "usercustomize"):
        found = importlib.util.find_spec(name)
        if found and found.origin:
            files.append(found.origin)
    for path in files:
        try:
            if entry.rstrip("/") in open(path, encoding="utf-8", errors="replace").read():
                sources.append(path)
        except OSError:
            pass
print(json.dumps({"grpc": origin, "grpcio": version, "entry": entry, "sources": sources,
                  "prefix": sys.prefix, "python": sys.executable}), flush=True)
import grpc
from grpc._cython import cygrpc
"""


def grpc_diagnosis(info, import_ok):
    """Turn the probe result into an actionable blocker (empty when grpc imports)."""
    if import_ok:
        return []
    if not info:
        return ["grpc does not import in the training environment (see traceback above), so Ray's agent dies "
                "and the node never starts"]
    where = ", ".join(info["sources"]) or "an unknown startup file"
    if info["grpc"] is None or info["grpcio"] is None:
        found = f"; the only grpc on sys.path is {info['grpc']} (added by {where})" if info["grpc"] else ""
        return [f"grpcio is not installed in {info['prefix']}{found}. Ray's agent needs it, so the node never "
                f"starts. Install it into this env: {info['python']} -m pip install grpcio"]
    return [f"grpcio {info['grpcio']} is installed, but {info['entry']} (added by {where}) comes first on sys.path "
            f"and its broken grpc wins. Remove that entry from the file(s) named"]


def python_env_blockers(env, repo):
    """Ray's dashboard and per-node agent import grpc; if that fails the raylet exits and ray.init times out."""
    result = subprocess.run([sys.executable, "-c", ENV_PROBE], cwd=repo, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    lines = result.stdout.strip().splitlines()
    try:
        info = json.loads(lines[0]) if lines else None
    except ValueError:
        info = None
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr)
    elif info:
        print(f"grpc: {info['grpc']} (grpcio {info['grpcio']})")
    return grpc_diagnosis(info, result.returncode == 0)


def owned_ray_processes():
    """This account's running Ray head/node processes; a plain ray.init() would attach to them."""
    raw = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], text=True)
    found = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and Path(parts[1].split()[0]).name in ("gcs_server", "raylet"):
            found.append((int(parts[0]), parts[1]))
    return found


RAY_PROBE = """
import json, ray
ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
nodes = [{"host": n.get("NodeManagerAddress"), "alive": bool(n.get("Alive")),
          "gpus": float(n.get("Resources", {}).get("GPU", 0))} for n in ray.nodes()]
print(json.dumps({"nodes": nodes, "available_gpus": float(ray.available_resources().get("GPU", 0))}))
ray.shutdown()
"""


def cluster_blockers(info, n_gpus, nnodes):
    """Mirror verl's resource-pool check so a run that cannot get its GPUs is refused before mkdir."""
    alive = [node for node in info["nodes"] if node["alive"]]
    for node in alive:
        print(f"Ray node {node['host']}: {node['gpus']:g} GPUs")
    big_enough = [node for node in alive if node["gpus"] >= n_gpus]
    need = n_gpus * nnodes
    blockers = []
    if len(big_enough) < nnodes:
        blockers.append(f"Ray cluster has {len(big_enough)} alive node(s) with >= {n_gpus} GPUs; "
                        f"{nnodes}-node mode needs {nnodes} ({need} GPUs total). "
                        "For one 8-GPU machine use --n-gpus 8 --nnodes 1")
    if info["available_gpus"] < need:
        blockers.append(f"Ray cluster has {info['available_gpus']:g} free GPUs; this layout needs {need}")
    return blockers


def ray_cluster_blockers(env, n_gpus, nnodes):
    """Connect to the existing Ray cluster (never starts one) and check it can host the run."""
    print(f"Multi-node mode: {nnodes} nodes x {n_gpus} GPUs; RAY_ADDRESS={env.get('RAY_ADDRESS', '(unset)')}")
    try:
        result = subprocess.run([sys.executable, "-c", RAY_PROBE], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        return ["Timed out connecting to the Ray cluster for multi-node mode"]
    lines = result.stdout.strip().splitlines()
    if result.returncode or not lines:
        print(result.stderr[-2000:], file=sys.stderr)
        return [f"No reachable Ray cluster for {nnodes}-node mode (start it with `ray start` on every node "
                "first). For one 8-GPU machine use --n-gpus 8 --nnodes 1"]
    return cluster_blockers(json.loads(lines[-1]), n_gpus, nnodes)


def describe_existing_run(run_dir, tail_bytes=16 * 1024, tail_lines=30):
    """Show why a previous attempt in this directory stopped; read-only. Returns saved checkpoint names."""
    print(f"Existing experiment directory: {run_dir}")
    checkpoints = sorted(path.name for path in (run_dir / "checkpoints").glob("global_step_*"))
    print("  saved checkpoints:", ", ".join(checkpoints) or "none")
    log = run_dir / "train.log"
    if not log.is_file():
        print("  no train.log (the previous attempt stopped before training started)")
        return checkpoints
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - tail_bytes))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()[-tail_lines:]
    print(f"  last {len(lines)} lines of {log}:")
    for line in lines:
        print("  | " + line)
    return checkpoints


def existing_run_plan(run_dir, overwrite):
    """Return (replace, blocker). Failed attempts (no checkpoint) are replaced; saved checkpoints need --overwrite."""
    if not run_dir.exists() and not run_dir.is_symlink():
        return False, None
    if run_dir.is_symlink() or not run_dir.is_dir():
        return False, f"{run_dir} exists but is not a plain directory; remove it yourself"
    checkpoints = describe_existing_run(run_dir)
    if checkpoints and not overwrite:
        return False, (f"{run_dir} holds saved checkpoints ({', '.join(checkpoints)}); "
                       "add --overwrite to delete it and train again, or use a new --name")
    return True, None


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
        help="GPUs per node. Global/mini batch and rollout count are unchanged.",
    )
    parser.add_argument(
        "--nnodes", type=int, choices=(1, 4), default=None,
        help="Ray nodes. Default: 4 with --n-gpus 8 (author 32-GPU layout; needs a running multi-node Ray "
             "cluster reached via RAY_ADDRESS), otherwise 1. Use --n-gpus 8 --nnodes 1 for one 8-GPU machine.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Also replace an existing experiment directory that holds saved checkpoints. A directory from a "
             "failed attempt (logs, no checkpoint) is always replaced. Never resumes.",
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
    nnodes = default_nnodes(args.n_gpus) if args.nnodes is None else args.nnodes
    if (args.n_gpus, nnodes) not in LAYOUTS:
        parser.error(f"Unsupported layout --n-gpus {args.n_gpus} --nnodes {nnodes}; 4 nodes require --n-gpus 8")
    repo = Path(__file__).resolve().parent.parent
    if not (repo / "recipe/dapo/main_dapo.py").is_file():
        raise ValueError("Place the baseline_4gpu folder directly inside NSR-main before running")
    root = Path(os.environ.get("NSR_ROOT", "/home/lihourun/hsc/NSR-work")).resolve()
    runs = Path(os.environ.get("BASELINE_RUNS", str(root / "paper-runs"))).resolve()
    run_dir = runs / args.name
    env = environment(root, repo, args.name, args.n_gpus, nnodes)
    ignored_pythonpath = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if ignored_pythonpath:
        print("Not passing inherited PYTHONPATH to training:", os.pathsep.join(ignored_pythonpath))
    _, _, temp_note = temp_dirs(root, args.name)
    if temp_note:
        print("NOTE:", temp_note)
    print(f"TMPDIR={env['TMPDIR']}  RAY_TMPDIR={env['RAY_TMPDIR']}")
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
        *hydra_args(root, run_dir, n_gpus=args.n_gpus, nnodes=nnodes),
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
    env_blockers = python_env_blockers(compose_env, repo)
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
    temp_parent = existing_parent(Path(env["RAY_TMPDIR"]))
    temp_fstype = mount_fstype(temp_parent)
    temp_disk = shutil.disk_usage(temp_parent)
    temp_ok = temp_disk.free >= max(20 * GIB, temp_disk.total * 0.06)
    print(f"Checkpoint filesystem: {free:.1f} GiB free; conservative launch budget: {MIN_FREE_GIB} GiB")
    print(f"Runtime filesystem: {runtime_disk.free/GIB:.1f} GiB free")
    print(f"Ray/temp filesystem ({temp_fstype}): {temp_disk.free/GIB:.1f} GiB free under {temp_parent}")
    try:
        print(f"/dev/shm: {shutil.disk_usage('/dev/shm').free/GIB:.1f} GiB free (Ray object store, NCCL)")
    except OSError:
        print("/dev/shm: not available")
    print(f"New experiment: {run_dir}")
    print("Selected GPUs busy:", busy or "none")
    print(f"This is one fresh 300-step run on {nnodes} node(s) x {args.n_gpus} GPUs = {nnodes * args.n_gpus} GPUs. "
          "It is NOT three independent replications.")
    blockers = list(env_blockers)
    if not socket_capable(temp_fstype):
        blockers.append(f"Ray/temp dir {env['RAY_TMPDIR']} is on {temp_fstype}, which cannot host Unix sockets; "
                        "set BASELINE_LOCAL_TMP to a local-disk directory")
    if len(env["RAY_TMPDIR"]) > MAX_RAY_TMPDIR_CHARS:
        blockers.append(f"RAY_TMPDIR {env['RAY_TMPDIR']} is longer than {MAX_RAY_TMPDIR_CHARS} characters; Ray's "
                        "socket paths would exceed the 107-byte Unix limit. Use a shorter BASELINE_LOCAL_TMP")
    if nnodes > 1:
        blockers.extend(ray_cluster_blockers(compose_env, args.n_gpus, nnodes))
    else:
        stale_ray = owned_ray_processes()
        for pid, command in stale_ray:
            print(f"  Ray process PID {pid}: {command[:160]}")
        if stale_ray:
            blockers.append("A Ray instance is already running for this account; single-node training would attach "
                            "to it instead of starting a clean one. If it is left over, run `ray stop` first")
    if old_tasks:
        blockers.append("This account already has another DAPO/Ray training process")
    disk_warnings = []
    if free < MIN_FREE_GIB:
        disk_warnings.append("Not enough conservative headroom for two complete saved checkpoints")
    if not runtime_ok:
        disk_warnings.append("Runtime/cache filesystem needs at least 20 GiB and 6% available")
    if not temp_ok:
        disk_warnings.append(f"Ray/temp filesystem {temp_parent} needs at least 20 GiB and 6% available")
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
    replace_existing, existing_blocker = existing_run_plan(run_dir, args.overwrite)
    if existing_blocker:
        blockers.append(existing_blocker)
    elif replace_existing:
        print(f"OVERWRITE: {run_dir} will be deleted and replaced by a fresh run once all checks pass")
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
    if replace_existing:
        shutil.rmtree(run_dir)
        print(f"Deleted previous experiment directory: {run_dir}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "RAY_TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved.yaml").write_text(result.stdout, encoding="utf-8")
    source_files = ["recipe/dapo/main_dapo.py", "recipe/dapo/dapo_ray_trainer.py", "verl/workers/fsdp_workers.py",
                    "verl/workers/actor/dp_actor.py", "verl/utils/reward_score/math_dapo.py", "verl/trainer/ppo/core_algos.py"]
    manifest = {"settings": settings(root, run_dir, n_gpus=args.n_gpus, nnodes=nnodes), "versions": versions,
                "command": cmd, "n_gpus": args.n_gpus, "nnodes": nnodes,
                "disk_check_skipped": args.skip_disk_check,
                "gpu_check_skipped": args.skip_gpu_check,
                "busy_selected_gpus_at_launch": busy,
                "disk_free_gib_at_launch": free,
                "runtime_disk_free_gib_at_launch": runtime_disk.free / GIB,
                "tmpdir": env["TMPDIR"], "ray_tmpdir": env["RAY_TMPDIR"],
                "ray_temp_fstype": temp_fstype, "ray_temp_disk_free_gib_at_launch": temp_disk.free / GIB,
                "ignored_pythonpath": ignored_pythonpath,
                "source_sha256": {f: sha256(repo / f) for f in source_files},
                "gpu_visibility": env["CUDA_VISIBLE_DEVICES"], "prompt_examples": examples,
                "model_config": json.loads((root / "models/Qwen2.5-Math-7B/config.json").read_text())}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copytree(Path(__file__).parent, run_dir / "launcher_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Starting training. Output: terminal and {run_dir / 'train.log'}", flush=True)
    with (run_dir / "train.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
        (run_dir / "driver.pid").write_text(str(process.pid) + "\n")
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
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
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
