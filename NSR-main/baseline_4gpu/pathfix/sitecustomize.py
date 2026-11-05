"""Put on PYTHONPATH by baseline_4gpu.launch for training processes only (driver, Ray daemons, workers).

Drops the sys.path entries listed in BASELINE_DROP_SYS_PATH: entries a startup *.pth file inserted ahead of
the environment's site-packages that shadow installed packages (e.g. ModelArts' vendored, broken grpc).
Python imports sitecustomize after all *.pth files, so the entries are already present here. Nothing on disk
is changed. A sitecustomize this file shadows is still executed afterwards.
"""
import importlib.util
import os
import sys

_drop = {os.path.abspath(p) for p in os.environ.get("BASELINE_DROP_SYS_PATH", "").split(os.pathsep) if p}
if _drop:
    sys.path[:] = [p for p in sys.path if not p or os.path.abspath(p) not in _drop]

_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    _candidate = os.path.join(os.path.abspath(_entry or "."), "sitecustomize.py")
    if os.path.dirname(_candidate) != _here and os.path.isfile(_candidate):
        _spec = importlib.util.spec_from_file_location("_baseline_chained_sitecustomize", _candidate)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        break
