#!/usr/bin/env bash
# Run pip against the active conda env only.
#
# A conda env is not a venv, so pip treats every distribution it can see on sys.path as installed in
# the env and tries to uninstall/replace it. On ModelArts, site-packages/modelarts.pth adds the
# read-only /modelarts/tools/visualization/tensorboard (with its own Markdown, grpcio, ...), so
# installs fail with "[Errno 30] Read-only file system". This wrapper hides every sys.path entry
# outside the env (startup *.pth entries, PYTHONPATH, user site) from pip, via the launcher's
# pathfix/sitecustomize.py. Nothing outside the env is touched, and no file is edited.
#
# Usage: bash manifests/pip-env-only.sh install --no-deps -r manifests/requirements-reference.txt
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pathfix="$here/../NSR-main/baseline_4gpu/pathfix"
[ -f "$pathfix/sitecustomize.py" ] || { echo "missing $pathfix/sitecustomize.py" >&2; exit 1; }
drop="$(PYTHONNOUSERSITE=1 python - <<'PY'
import os, sys
roots = tuple(os.path.join(os.path.abspath(r), "") for r in {sys.prefix, sys.base_prefix})
print(os.pathsep.join(p for p in sys.path if p and not os.path.abspath(p).startswith(roots)))
PY
)"
echo "pip-env-only: env $(python -c 'import sys; print(sys.prefix)'); hiding: ${drop:-nothing}" >&2
PYTHONNOUSERSITE=1 PYTHONPATH="$pathfix" BASELINE_DROP_SYS_PATH="$drop" exec python -m pip "$@"
