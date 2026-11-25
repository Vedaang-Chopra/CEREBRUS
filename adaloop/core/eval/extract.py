"""Short-answer extraction from raw model output.

Ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/scorers.py @ e870bff.
The model writes thinking after `<|content_thinking|>` and the answer after `<|content_text|>`.
When the prompt already ends with `<|content_text|>` (answer prefill), the output starts at the answer itself.
"""
import re

THINKING = "<|content_thinking|>"
ANSWER = "<|content_text|>"
END = "<|end_message|>"
SPECIAL = re.compile(r"<\|[^|]+\|>")
LEAD_IN = re.compile(r"^(final answer|answer)\s*:\s*", flags=re.IGNORECASE)


def answer_is_open(prefill: str | None) -> bool:
    """True when the prefill ends with the answer marker, so generation starts inside the answer message."""
    return (prefill or "").endswith(ANSWER)


def _clean(text: str) -> str | None:
    """Cut at the first special token, drop whitespace and an "answer:" lead-in; None when nothing is left."""
    answer = SPECIAL.split(text, maxsplit=1)[0].strip()
    answer = LEAD_IN.sub("", answer).strip()
    return answer or None


def extract_answer(raw_output: str, hit_limit: bool, prefilled: bool = False) -> str | None:
    """Return the final answer text, or None when none can be read.

    `raw_output` must be decoded with special tokens kept. `hit_limit` is True when generation stopped at
    `max_new_tokens`; an answer cut off before `<|end_message|>` counts as unparseable.
    `prefilled` is True when the prompt ended with `<|content_text|>`: the answer is the text before the first
    special token, and anything after it (a later message of the model) is ignored.
    """
    if not raw_output:
        return None
    if prefilled:
        if hit_limit and not SPECIAL.search(raw_output):
            return None  # the answer itself was cut
        return _clean(raw_output)
    if ANSWER in raw_output:
        answer = raw_output.rsplit(ANSWER, 1)[1]
    elif THINKING in raw_output:
        return None  # thinking without a final answer
    else:
        answer = raw_output
    if hit_limit and END not in answer:
        return None
    return _clean(answer)
