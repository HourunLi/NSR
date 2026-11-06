"""Answer scoring for the baseline: math_dapo's "Answer:" rule, with a \\boxed{} fallback.

Deviation from the author setup (see README): at step 0 Qwen2.5-Math-7B wrote "Answer:" in 0/960
AIME validation samples but put the final answer in \\boxed{} in 59% of them (13.3% correct), so the
unmodified math_dapo scorer gave every sample -1 and DAPO's group filter left nothing to train on.

Rule, applied to the last 300 characters exactly as math_dapo does:
  1. If an "Answer: <x>" line is present, score it with math_dapo's Minerva check (unchanged).
  2. Otherwise take the last \\boxed{...} and compare it with the same normalisation.
The prompt is unchanged (it still asks for "Answer:"). Returns the math_dapo fields plus
boxed_fallback (1.0 when rule 2 decided), which verl reports as a validation/training metric.
"""
import importlib.util
from pathlib import Path

_MATH_DAPO = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "math_dapo.py"
_spec = importlib.util.spec_from_file_location("baseline_math_dapo", _MATH_DAPO)
math_dapo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_dapo)

TAIL_CHARS = 300  # math_dapo.compute_score reads solution_str[-300:]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    tail = solution_str[-TAIL_CHARS:]
    if math_dapo.re.search(r"(?i)Answer\s*:\s*([^\n]+)", tail):
        correct, pred = math_dapo.is_correct_minerva(tail, ground_truth)
        fallback = 0.0
    else:
        boxed = math_dapo.last_boxed_only_string(tail)
        if boxed is None:
            correct, pred, fallback = False, "[INVALID]", 0.0
        else:
            pred = math_dapo.normalize_final_answer(math_dapo.remove_boxed(boxed))
            correct = pred == math_dapo.normalize_final_answer(ground_truth)
            fallback = 1.0
    return {"score": 1.0 if correct else -1.0, "acc": bool(correct), "pred": pred, "boxed_fallback": fallback}
