"""Re-score saved result rows with the current scorers and collect the wrong ones."""
from pathlib import Path

from adaloop.core.eval.scorers import score, score_lmms
from adaloop.orchestration.evaluate import read_results


def wrong_predictions(results_dir: str | Path) -> tuple[list[Path], list[dict]]:
    """Return (files read, rows scoring 0) from the result JSONL files under `results_dir`; both may be empty.

    nb04 P and T files are used when there are any, otherwise every `.jsonl`. Rows without `pred` and `answers`
    are skipped, and a repeated (benchmark, item, prediction) is listed once. "Wrong" means the primary score is 0; the
    returned rows carry `correct` and `correct_lmms` re-scored with the current scorers.
    """
    files = sorted(Path(results_dir).rglob("*.jsonl"))
    files = [f for f in files if f.parent.name == "nb04" and ("_P_" in f.name or "_T_" in f.name)] or files
    wrong = {}
    for path in files:
        for row in read_results(path):
            if not {"benchmark", "item_id", "pred", "answers"} <= row.keys():
                continue
            scored = (row["benchmark"], row["pred"], row["answers"])
            if score(*scored) == 0.0:
                wrong[(row["benchmark"], row["item_id"], row["pred"])] = {
                    **row, "correct": 0.0, "correct_lmms": score_lmms(*scored)}
    return files, list(wrong.values())
