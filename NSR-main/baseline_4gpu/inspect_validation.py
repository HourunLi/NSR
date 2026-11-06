"""Explain why validation answers are (not) scored, from a dump such as validation/0.jsonl.

The math_dapo scorer only reads the last 300 characters of a response and looks for "Answer: <x>".
This report separates the usual failure modes: no "Answer:" at all (e.g. \\boxed{} only), an answer
that is followed by more text (generation did not stop, e.g. a new chat turn after <|im_end|>),
looping output that runs to the length limit, and plain wrong answers. With the AIME parquet it
also checks whether the answer the model wrote (via "Answer:" anywhere or the last \\boxed{}) matches
the ground truth, i.e. whether fixing the format/stopping alone would give a non-zero score.

Usage:
  python -m baseline_4gpu.inspect_validation <run_dir>/validation/0.jsonl [--root $NSR_ROOT] [--examples 3]
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
TURN = re.compile(r"(?m)^\s*(user|assistant|system)\s*$")


def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def norm(answer):
    answer = (answer or "").strip().strip("$").strip().rstrip(".")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    return answer.replace(" ", "").replace(",", "")


def looping(text, probe=60, window=3000, times=5):
    tail = text[-probe:]
    return len(text) > window and len(tail.strip()) > 10 and text[-window:].count(tail) >= times


def gold_answers(root):
    """Map the user text of each AIME question to its ground truth (None if pyarrow/data unavailable)."""
    try:
        import pyarrow.parquet as pq
        from .protocol import dataset_path
        table = pq.read_table(dataset_path(Path(root), "aime-2024.parquet"), columns=["prompt", "reward_model"])
    except Exception as exc:  # report still works without gold answers
        print(f"(no ground truth: {type(exc).__name__}: {exc})")
        return None
    gold = {}
    for row in table.to_pylist():
        user = next(m["content"] for m in row["prompt"] if m.get("role") == "user")
        gold[user.strip()] = str(row["reward_model"]["ground_truth"])
    return gold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--root", default=os.environ.get("NSR_ROOT", "/data/lihourun/NSR"))
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dump.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [row["output"] for row in rows]
    n = len(rows)
    if not n:
        raise SystemExit("empty dump")

    def share(predicate):
        count = sum(1 for row, out in zip(rows, outputs) if predicate(row, out))
        return f"{count}/{n} ({count / n:.1%})"

    lengths = sorted(len(out) for out in outputs)
    print(f"samples: {n}   scored correct (acc): {sum(bool(r.get('acc')) for r in rows)}")
    print(f"output length in characters: min {lengths[0]}, median {lengths[n // 2]}, max {lengths[-1]}")
    print("contains 'Answer:' anywhere:                    ", share(lambda r, o: ANSWER.search(o)))
    print("'Answer:' in the last 300 chars (scorer's view):", share(lambda r, o: ANSWER.search(o[-300:])))
    print("more than 300 chars after the first 'Answer:':  ",
          share(lambda r, o: (m := ANSWER.search(o)) is not None and len(o) - m.end() > 300))
    print("contains \\boxed{...}:                           ", share(lambda r, o: last_boxed(o) is not None))
    print("chat-turn lines (user/assistant) in the output: ", share(lambda r, o: TURN.search(o)))
    print("looping (same text repeated at the end):        ", share(lambda r, o: looping(o)))
    preds = collections.Counter(str(r.get("pred")) for r in rows)
    print("most common scorer predictions:", preds.most_common(6))

    gold = gold_answers(args.root)
    if gold:
        def gt(row):
            return next((answer for user, answer in gold.items() if user and user in row["input"]), None)
        matched = [gt(row) for row in rows]
        print(f"ground truth found for {sum(g is not None for g in matched)}/{n} samples")
        first_answer = lambda o: norm(ANSWER.search(o).group(1)) if ANSWER.search(o) else None
        print("first 'Answer:' equals ground truth:             ",
              share(lambda r, o: (g := gt(r)) is not None and first_answer(o) == norm(g)))
        print("last \\boxed{} equals ground truth:              ",
              share(lambda r, o: (g := gt(r)) is not None and norm(last_boxed(o)) == norm(g)))

    order = sorted(range(n), key=lambda i: len(outputs[i]))
    picks = [order[0], order[n // 2], order[-1]][: args.examples]
    for label, i in zip(("shortest", "median", "longest"), picks):
        out = outputs[i]
        print(f"\n===== {label} output ({len(out)} chars), pred={rows[i].get('pred')!r} =====")
        print("--- first 400 chars ---")
        print(out[:400])
        print("--- last 500 chars ---")
        print(out[-500:])


if __name__ == "__main__":
    main()
