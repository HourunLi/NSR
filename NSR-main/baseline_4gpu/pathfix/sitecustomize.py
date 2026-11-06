"""Env-only import path for the baseline launcher, its training processes and manifests/pip-env-only.sh.

Loaded through PYTHONPATH. With BASELINE_ENV_ONLY=1 it removes every sys.path entry outside the active
environment (sys.prefix / sys.base_prefix, i.e. the stdlib and the env's own site-packages), keeping only
PYTHONPATH entries and the script directory. Directories added by startup *.pth files (e.g. ModelArts'
modelarts.pth) or by other tools are hidden, so imports and pip only see packages installed in the env;
a package missing from the env fails loudly instead of being borrowed from a shared directory.
Child processes (Ray daemons and workers, vLLM workers) inherit the variables and get the same view.

Python imports sitecustomize after processing all *.pth files, so those entries are already present here.
Nothing on disk is changed. The removed entries are kept in BASELINE_HIDDEN, and a sitecustomize that this
file shadows (and that is still visible) is executed afterwards.
"""
import importlib.util
import os
import sys

BASELINE_HIDDEN = []
if os.environ.get("BASELINE_ENV_ONLY") == "1":
    _keep = {os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p}
    _roots = tuple(os.path.join(os.path.abspath(r), "") for r in {sys.prefix, sys.base_prefix})
    _visible = []
    for _entry in sys.path:
        _full = os.path.abspath(_entry) if _entry else ""
        if not _entry or _full in _keep or _full.startswith(_roots):
            _visible.append(_entry)
        else:
            BASELINE_HIDDEN.append(_entry)
    sys.path[:] = _visible

_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    _candidate = os.path.join(os.path.abspath(_entry or "."), "sitecustomize.py")
    if os.path.dirname(_candidate) != _here and os.path.isfile(_candidate):
        _spec = importlib.util.spec_from_file_location("_baseline_chained_sitecustomize", _candidate)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        break
