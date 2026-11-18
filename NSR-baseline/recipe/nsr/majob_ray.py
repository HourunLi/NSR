"""Wait for the ModelArts Ray cluster started by majob.sh."""

import os
import socket
import sys
import time


def head_reachable(address):
    host, port = address.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def check_ray_process(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("The local 'ray start --block' process exited. See its startup logs.") from None


def wait_for_head(address, deadline, pid=None):
    print(f"Waiting for Ray head at {address}...", flush=True)
    while time.monotonic() < deadline:
        if pid is not None:
            check_ray_process(pid)
        if head_reachable(address):
            return
        time.sleep(2)
    raise SystemExit(f"Timed out waiting for Ray head at {address}.")


def wait_for_workers(address, timeout, pid, nnodes, gpus_per_node):
    import ray

    deadline = time.monotonic() + timeout
    wait_for_head(address, deadline, pid)
    ray.init(address=address, logging_level="ERROR", log_to_driver=False)
    try:
        last_report = None
        while time.monotonic() < deadline:
            check_ray_process(pid)
            nodes = sorted(
                (node["NodeManagerAddress"], node["Resources"].get("GPU", 0))
                for node in ray.nodes()
                if node["Alive"]
            )
            if nodes != last_report:
                print(f"Ray live nodes (address, GPUs): {nodes}", flush=True)
                last_report = nodes
            if len(nodes) >= nnodes:
                if len(nodes) != nnodes or any(gpus != gpus_per_node for _, gpus in nodes):
                    raise SystemExit(f"Expected exactly {nnodes} Ray nodes with {gpus_per_node} GPUs each.")
                print(f"Ray cluster ready: {nnodes} nodes × {gpus_per_node} GPUs = {nnodes * gpus_per_node}", flush=True)
                return
            time.sleep(2)
        raise SystemExit(f"Timed out waiting for {nnodes} Ray nodes at {address}.")
    finally:
        ray.shutdown()


def wait_for_shutdown(address, pid):
    # The head's exit status reports the training result. Workers stay alive
    # across both stages and exit once the head shuts down the Ray cluster.
    missed = 0
    while missed < 3:
        if head_reachable(address):
            check_ray_process(pid)
            missed = 0
        else:
            missed += 1
        time.sleep(2)
    print(f"Ray head at {address} has stopped. Exiting worker.", flush=True)


if __name__ == "__main__":
    address = os.environ["RAY_ADDRESS"]
    action, *args = sys.argv[1:]
    if action == "wait-head":
        wait_for_head(address, time.monotonic() + int(args[0]))
    elif action == "wait-workers":
        wait_for_workers(address, *map(int, args))
    elif action == "wait-stop":
        wait_for_shutdown(address, int(args[0]))
    else:
        raise SystemExit(f"Unknown Ray preflight action: {action}")
