"""Answer normalization and per-benchmark scorers.

GQA scorer ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/scorers.py @ e870bff.
TextVQA accuracy ported from EvolvingLMMs-Lab/lmms-eval:lmms_eval/tasks/textvqa/utils.py::textvqa_process_results
@ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef.
"""
import logging
import re
import statistics
from collections.abc import Sequence

from adaloop.core.eval.vqa_normalize import process_vqa_answer

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


def score_textvqa(prediction: str | None, answers: Sequence[str]) -> float:
    """VQA accuracy: mean over references of min(1, matches among the other references / 3).

    `answers` holds the 10 human answers. A missing or blank prediction scores 0.0.
    """
    if not prediction or not prediction.strip():
        return 0.0
    pred = process_vqa_answer(prediction)
    refs = [process_vqa_answer(a) for a in answers]
    accuracies = []
    for i in range(len(refs)):
        matches = sum(ref == pred for j, ref in enumerate(refs) if j != i)
        accuracies.append(min(1.0, matches / 3))
    return statistics.mean(accuracies) if accuracies else 0.0


SCORERS = {"gqa": score_gqa, "textvqa": score_textvqa}


def score(benchmark: str, prediction: str | None, answers: Sequence[str]) -> float:
    """Score one prediction with the scorer of `benchmark`."""
    return SCORERS[benchmark](prediction, answers)
