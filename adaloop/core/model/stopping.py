"""Stop rule of the thinking budget: end the rows that are still thinking when the budget is reached."""
from collections.abc import Sequence

import torch
from transformers import StoppingCriteria


def thinking_open(token_ids: Sequence[int], thinking_id: int, end_id: int) -> bool:
    """True when the last thinking message among `token_ids` has no `<|end_message|>` after it."""
    is_open = False
    for token_id in token_ids:
        if token_id == thinking_id:
            is_open = True
        elif token_id == end_id:
            is_open = False
    return is_open


class StopOpenThinking(StoppingCriteria):
    """After exactly `budget` new tokens, stop each row whose thinking message is open; other rows go on."""

    def __init__(self, prompt_width: int, budget: int, thinking_id: int, end_id: int):
        self.prompt_width, self.budget = prompt_width, budget
        self.thinking_id, self.end_id = thinking_id, end_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        done = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)  # (B,)
        if input_ids.shape[1] - self.prompt_width == self.budget:
            for i, row in enumerate(input_ids[:, self.prompt_width :].tolist()):
                done[i] = thinking_open(row, self.thinking_id, self.end_id)
        return done
