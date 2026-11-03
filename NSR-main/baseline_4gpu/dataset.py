"""Explicit system prompt, without editing the original Parquet or tokenizer."""
from verl.utils.dataset.rl_dataset import RLHFDataset
from baseline_4gpu.protocol import with_answer_system


class AnswerFormatDataset(RLHFDataset):
    def _build_messages(self, example):
        return with_answer_system(super()._build_messages(example))
