"""Scorers of the headroom-probe datasets: multiple choice, ChartQA relaxed accuracy, CLEVR, RealWorldQA.

`answers[0]` is the gold answer; for multiple choice it is the option letter and `answers[1]`, if present, that option's
text. A missing prediction scores 0.0.
"""
import re
import string
from collections.abc import Sequence

from adaloop.core.eval.vqa_normalize import process_vqa_answer

# a lone letter, or a letter followed by ")", ".", ":" or ","; "A dog" is not the letter A
LETTER = re.compile(r"^\(?\s*([A-Za-z])\s*(?:[).:,]|$)")
CHARTQA_TOLERANCE = 0.05  # part of the metric's definition, not a setting
NO_PUNCTUATION = str.maketrans("", "", string.punctuation)


def score_mc(prediction: str | None, answers: Sequence[str]) -> float:
    """1.0 if the predicted option letter is the gold one, or, without a letter, the gold option's text was written."""
    if not prediction:
        return 0.0
    pred = prediction.strip()
    found = LETTER.match(pred)
    if found:
        return float(found.group(1).upper() == answers[0])
    if len(answers) < 2:
        return 0.0
    return float(pred.rstrip(".").lower() == answers[1].strip().rstrip(".").lower())


def _number(text: str) -> float | None:
    """The value of "12.5" or "40%" (as 0.4); None if it is not a number."""
    try:
        return float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        return None


def score_chartqa(prediction: str | None, answers: Sequence[str]) -> float:
    """Relaxed accuracy (Methani et al. 2020; the `relaxed_correctness` rule of Pix2Struct and lmms-eval): numbers within
    5% of a non-zero gold number, anything else equal ignoring case. A final period of the prediction is dropped."""
    if not prediction:
        return 0.0
    pred, gold = prediction.strip().rstrip("."), answers[0].strip()
    pred_value, gold_value = _number(pred), _number(gold)
    if pred_value is not None and gold_value:
        return float(abs(pred_value - gold_value) / abs(gold_value) <= CHARTQA_TOLERANCE)
    return float(pred.lower() == gold.lower())


def score_clevr(prediction: str | None, answers: Sequence[str]) -> float:
    """Exact match after VQA answer processing (lower case, punctuation, "three" -> "3")."""
    if not prediction or not prediction.strip():
        return 0.0
    return float(process_vqa_answer(prediction) == process_vqa_answer(answers[0]))


def score_realworldqa(prediction: str | None, answers: Sequence[str]) -> float:
    """Option letter if the gold answer is a capital A-E, else equal ignoring case and ASCII punctuation."""
    gold = answers[0].strip()
    if len(gold) == 1 and gold in "ABCDE":
        return score_mc(prediction, (gold,))
    if not prediction:
        return 0.0
    return float(prediction.strip().lower().translate(NO_PUNCTUATION) == gold.lower().translate(NO_PUNCTUATION))
