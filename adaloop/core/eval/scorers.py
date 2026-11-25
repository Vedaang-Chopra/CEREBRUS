"""Per-benchmark scorers.

GQA exact match ported from EvolvingLMMs-Lab/lmms-eval:lmms_eval/api/metrics.py::exact_match_hf_evaluate
(ignore_case, ignore_punctuation, as set in lmms_eval/tasks/gqa/gqa.yaml; the prediction is stripped first, as in
lmms_eval/api/task.py::ConfigurableTask.process_results) @ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef.
TextVQA accuracy ported from EvolvingLMMs-Lab/lmms-eval:lmms_eval/tasks/textvqa/utils.py::textvqa_process_results
@ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef.
"""
import statistics
import string
from collections.abc import Sequence

from adaloop.core.eval.vqa_normalize import process_vqa_answer


NO_PUNCTUATION = str.maketrans("", "", string.punctuation)  # deletes ASCII punctuation


def score_gqa(prediction: str | None, answers: Sequence[str]) -> float:
    """1.0 if prediction and the (single) reference match, ignoring case and ASCII punctuation, else 0.0.

    Articles, inner whitespace and number words are compared as written. A missing prediction scores 0.0.
    """
    if prediction is None:
        return 0.0
    pred = prediction.strip().lower().translate(NO_PUNCTUATION)  # strip comes before punctuation removal, as upstream
    return float(pred == answers[0].lower().translate(NO_PUNCTUATION))


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
