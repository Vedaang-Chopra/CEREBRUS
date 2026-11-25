"""Short-answer extraction from raw model output.

Ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/scorers.py @ e870bff.
The model writes thinking after `<|content_thinking|>` and the answer after `<|content_text|>`.
"""
import logging
import re

logger = logging.getLogger(__name__)

THINKING = "<|content_thinking|>"
ANSWER = "<|content_text|>"
END = "<|end_message|>"
SPECIAL = re.compile(r"<\|[^|]+\|>")
LEAD_IN = re.compile(r"^(final answer|answer)\s*:\s*", flags=re.IGNORECASE)


def extract_answer(raw_output: str, hit_limit: bool) -> str | None:
    """Return the final answer text, or None when none can be read.

    `raw_output` must be decoded with special tokens kept. `hit_limit` is True when generation stopped at
    `max_new_tokens`; an answer cut off before `<|end_message|>` counts as unparseable.
    """
    if not raw_output:
        return None
    if ANSWER in raw_output:
        answer = raw_output.rsplit(ANSWER, 1)[1]
    elif THINKING in raw_output:
        return None  # thinking without a final answer
    else:
        answer = raw_output
    if hit_limit and END not in answer:
        return None
    answer = SPECIAL.split(answer, maxsplit=1)[0].strip()
    answer = LEAD_IN.sub("", answer).strip()
    return answer or None
