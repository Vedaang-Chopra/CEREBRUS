"""Short-answer extraction from raw model output.

Ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/scorers.py @ e870bff.
The model writes thinking after `<|content_thinking|>` and the answer after `<|content_text|>`.
When the prompt already ends with `<|content_text|>` (answer prefill), the output starts at the answer itself.
With an `answer_format` marker ("ANSWER:") the rule is strict: thinking is dropped, the answer is the rest of the
last line that starts with the marker (any case), and without such a line there is no answer.
"""
import re
from collections.abc import Collection, Sequence

THINKING = "<|content_thinking|>"
ANSWER = "<|content_text|>"
END = "<|end_message|>"
SPECIAL = re.compile(r"<\|[^|]+\|>")
LEAD_IN = re.compile(r"^(final answer|answer)\s*:\s*", flags=re.IGNORECASE)
BOUNDARY = re.compile("|".join(map(re.escape, (THINKING, ANSWER, END))))
THINKING_BLOCK = re.compile(rf"{re.escape(THINKING)}.*?(?:{re.escape(END)}|\Z)", flags=re.DOTALL)


def answer_is_open(prefill: str | None) -> bool:
    """True when the prefill ends with the answer marker, so generation starts inside the answer message."""
    return (prefill or "").endswith(ANSWER)


def _clean(text: str) -> str | None:
    """Cut at the first special token, drop whitespace and an "answer:" lead-in; None when nothing is left."""
    answer = SPECIAL.split(text, maxsplit=1)[0].strip()
    answer = LEAD_IN.sub("", answer).strip()
    return answer or None


def _strip_thinking(raw_output: str, prefilled: bool) -> str:
    """Output without thinking messages, an unclosed one included; each becomes a message boundary."""
    first = BOUNDARY.search(raw_output)
    if not prefilled and first and first.group() == END:  # closing only: the output starts inside the thinking
        raw_output = raw_output[first.start() :]
    return THINKING_BLOCK.sub(END, raw_output)


def _format_line(raw_output: str, prefilled: bool, answer_format: str) -> tuple[str, bool] | None:
    """(rest of the last marker line, whether a newline or special token ends it); None without such a line."""
    text = _strip_thinking(raw_output, prefilled)
    # a line starts at the output start, after a newline, or after a special token
    start = rf"(?:^|\n|{SPECIAL.pattern})[ \t]*{re.escape(answer_format)}"
    found = list(re.finditer(start, text, flags=re.IGNORECASE))
    if not found:
        return None
    parts = re.split(rf"\n|{SPECIAL.pattern}", text[found[-1].end() :], maxsplit=1)
    return parts[0], len(parts) > 1


def has_answer_format(raw_output: str, answer_format: str, prefilled: bool = False) -> bool:
    """True when a line outside thinking starts with the marker (the answer on it may still be empty or cut)."""
    return _format_line(raw_output, prefilled, answer_format) is not None


def count_message_tokens(
    token_ids: Sequence[int], thinking_id: int, text_id: int, end_ids: Collection[int], open_part: str | None = None
) -> tuple[int, int]:
    """(thinking tokens, answer tokens) among generated ids: message contents only, markers excluded.

    `open_part` ("think" or "answer") is the message the prompt left open, where the output starts.
    """
    counts = {"think": 0, "answer": 0}
    part = open_part
    for token_id in token_ids:
        if token_id == thinking_id:
            part = "think"
        elif token_id == text_id:
            part = "answer"
        elif token_id in end_ids:
            part = None
        elif part:
            counts[part] += 1
    return counts["think"], counts["answer"]


def extract_answer(
    raw_output: str, hit_limit: bool, prefilled: bool = False, answer_format: str | None = None
) -> str | None:
    """Return the final answer text, or None when none can be read.

    `raw_output` must be decoded with special tokens kept. `hit_limit` is True when generation stopped at
    `max_new_tokens`; an answer cut off before `<|end_message|>` counts as unparseable.
    `prefilled` is True when the prompt ended with `<|content_text|>`: the answer ends at the first special token
    or newline, and anything after it (a later message, or reasoning written after the answer) is ignored. Only an
    output with neither that hit the limit counts as cut off.
    With `answer_format` only the marker line counts, whatever the prefill: the rest of the last line starting with
    it, up to a special token or newline (cut off if neither follows and the limit was hit). There is no fallback.
    """
    if not raw_output:
        return None
    if answer_format:
        found = _format_line(raw_output, prefilled, answer_format)
        if found is None or (hit_limit and not found[1]):
            return None
        return found[0].strip() or None
    if prefilled:
        first_line, newline, _ = raw_output.lstrip().partition("\n")
        if hit_limit and not newline and not SPECIAL.search(first_line):
            return None  # the answer itself was cut
        return _clean(first_line)
    if ANSWER in raw_output:
        answer = raw_output.rsplit(ANSWER, 1)[1]
    elif THINKING in raw_output:
        return None  # thinking without a final answer
    else:
        answer = raw_output
    if hit_limit and END not in answer:
        return None
    return _clean(answer)
