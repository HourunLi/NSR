"""Compute Pass@1/Pass@16 from the 32 saved samples per AIME problem."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Expected 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(path):
    groups = defaultdict(list)
    invalid = 0
    steps = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "acc" not in row or row["acc"] not in (True, False, 0, 1):
                raise ValueError("Expected an acc field; do not infer accuracy from shaped rewards")
            groups[row["input"]].append(int(row["acc"]))
            steps.add(row["step"])
            invalid += int(row.get("pred") in ("[INVALID]", None, ""))
    if len(steps) != 1 or len(groups) != 30 or any(len(v) != 32 for v in groups.values()):
        raise ValueError("Expected one evaluation step, 30 prompts and exactly 32 samples per prompt")
    return {
        "step": next(iter(steps)),
        "questions": 30,
        "samples_per_question": 32,
        "pass_at_1": sum(sum(v) / 32 for v in groups.values()) / 30,
        "pass_at_16": sum(pass_at_k(32, sum(v), 16) for v in groups.values()) / 30,
        "invalid_fraction": invalid / 960,
        "estimator": "mean over questions of 1 - C(n-c,k)/C(n,k); existing scorer unchanged",
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), ensure_ascii=False, indent=2))
