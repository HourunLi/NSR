#!/usr/bin/env bash
# Run pip against the active conda env only (same env-only rule the training launcher uses).
#
# A conda env is not a venv, so pip treats every distribution it can see on sys.path as installed in
# the env and tries to uninstall/replace it. On ModelArts, site-packages/modelarts.pth adds the
# read-only /modelarts/tools/visualization/tensorboard (with its own Markdown, grpcio, ...), so
# installs fail with "[Errno 30] Read-only file system", and packages missing from the env would
# silently be borrowed from there. With BASELINE_ENV_ONLY=1, pathfix/sitecustomize.py hides every
# sys.path entry outside the env (startup *.pth entries, PYTHONPATH, user site). Nothing is edited.
#
# Usage: bash manifests/pip-env-only.sh install --no-deps -r manifests/requirements-reference.txt
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pathfix="$here/../NSR-main/baseline_4gpu/pathfix"
[ -f "$pathfix/sitecustomize.py" ] || { echo "missing $pathfix/sitecustomize.py" >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONPATH="$pathfix" BASELINE_ENV_ONLY=1
python -c 'import sys, sitecustomize as s; print("pip-env-only: env", sys.prefix, "| hidden:", ":".join(s.BASELINE_HIDDEN) or "nothing", file=sys.stderr)'
exec python -m pip "$@"
