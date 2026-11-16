"""Score Answer: first; use the last boxed answer only when Answer: is absent."""
import re

from verl.utils.reward_score import math_dapo


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    tail = solution_str[-300:]
    if re.search(r"(?i)Answer\s*:\s*([^\n]+)", tail):
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
    return {
        "score": 1.0 if correct else -1.0,
        "acc": bool(correct),
        "pred": pred,
        "boxed_fallback": fallback,
    }
