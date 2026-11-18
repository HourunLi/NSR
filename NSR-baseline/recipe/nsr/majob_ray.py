"""Wait for the ModelArts Ray cluster started by majob.sh."""

import os
import socket
import sys
import time


def reachable_head_ip(address):
    host, port = address.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=2) as connection:
            return connection.getpeername()[0]
    except OSError:
        return None


def check_ray_process(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("The local 'ray start --block' process exited. See its startup logs.") from None


def wait_for_head(address, deadline, pid=None):
    print(f"Waiting for Ray head at {address}...", file=sys.stderr, flush=True)
    while time.monotonic() < deadline:
        if pid is not None:
            check_ray_process(pid)
        ip = reachable_head_ip(address)
        if ip is not None:
            return ip
        time.sleep(2)
    raise SystemExit(f"Timed out waiting for Ray head at {address}.")


def wait_for_workers(address, timeout, pid, nnodes, gpus_per_node):
    from ray._raylet import GcsClient
    from ray.core.generated.gcs_pb2 import GcsNodeInfo
    from ray.exceptions import RpcError

    deadline = time.monotonic() + timeout
    wait_for_head(address, deadline, pid)
    # Query Ray 2.43's GCS directly. A driver requires a registered local raylet,
    # so ray.init() cannot report which raylets are missing during startup.
    client = None
    last_error = None
    last_report = None
    local_ip = os.environ["NSR_RAY_NODE_IP"]
    while time.monotonic() < deadline:
        check_ray_process(pid)
        try:
            if client is None:
                client = GcsClient(address=address, nums_reconnect_retry=0)
            node_info = client.get_all_node_info(timeout=5)
        except (RpcError, RuntimeError) as error:
            if str(error) != last_error:
                last_error = str(error)
                print(f"Waiting for Ray GCS at {address}: {last_error}", file=sys.stderr, flush=True)
            time.sleep(2)
            continue
        nodes = sorted(
            (node.node_manager_address, node.resources_total.get("GPU", 0))
            for node in node_info.values()
            if node.state == GcsNodeInfo.ALIVE
        )
        if nodes != last_report:
            print(f"Ray live nodes (address, GPUs): {nodes}", flush=True)
            last_report = nodes
        if len(nodes) >= nnodes:
            if len(nodes) != nnodes or any(gpus != gpus_per_node for _, gpus in nodes):
                raise SystemExit(f"Expected exactly {nnodes} Ray nodes with {gpus_per_node} GPUs each.")
            if not any(ip == local_ip for ip, _ in nodes):
                raise SystemExit(f"The Ray head's local IP {local_ip} is missing from the registered nodes: {nodes}")
            print(f"Ray cluster ready: {nnodes} nodes × {gpus_per_node} GPUs = {nnodes * gpus_per_node}", flush=True)
            return
        time.sleep(2)
    raise SystemExit(f"Timed out waiting for {nnodes} Ray nodes at {address}. Last live nodes: {last_report or []}")


def wait_for_shutdown(address, pid):
    # The head's exit status reports the training result. Workers stay alive
    # across both stages and exit once the head shuts down the Ray cluster.
    missed = 0
    while missed < 3:
        if reachable_head_ip(address) is not None:
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
        print(wait_for_head(address, time.monotonic() + int(args[0])))
    elif action == "wait-workers":
        wait_for_workers(address, *map(int, args))
    elif action == "wait-stop":
        wait_for_shutdown(address, int(args[0]))
    else:
        raise SystemExit(f"Unknown Ray preflight action: {action}")
