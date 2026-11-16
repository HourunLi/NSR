"""Use one explicit answer-format system message for train and validation."""
from copy import deepcopy

from verl.utils.dataset.rl_dataset import RLHFDataset


SYSTEM_PROMPT = (
    "Please reason step by step. The last line of your response must be "
    "Answer: <final answer>. Replace <final answer> with the actual answer. "
    "Do not wrap the final answer in LaTeX commands."
)


class AnswerFormatDataset(RLHFDataset):
    def _build_messages(self, example):
        messages = deepcopy(list(super()._build_messages(example)))
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("A user question is required")
        systems = [message for message in messages if message.get("role") == "system"]
        if systems:
            if len(systems) != 1 or messages[0] != systems[0] or systems[0].get("content") != SYSTEM_PROMPT:
                raise ValueError("Unexpected system prompt in dataset")
            return messages
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
