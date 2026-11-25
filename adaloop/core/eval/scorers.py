"""Answer normalization and per-benchmark scorers.

GQA scorer ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/scorers.py @ e870bff.
"""
import logging
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def normalize_answer(answer: str | None) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    if answer is None:
        return ""
    answer = re.sub(r"\s+", " ", str(answer).strip().lower())
    return answer.rstrip(".!?").strip()


def score_gqa(prediction: str | None, answers: Sequence[str]) -> float:
    """1.0 if the normalized prediction equals the (single) reference answer, else 0.0."""
    pred = normalize_answer(prediction)
    return float(bool(pred) and pred == normalize_answer(answers[0]))


SCORERS = {"gqa": score_gqa}


def score(benchmark: str, prediction: str | None, answers: Sequence[str]) -> float:
    """Score one prediction with the scorer of `benchmark`."""
    return SCORERS[benchmark](prediction, answers)
