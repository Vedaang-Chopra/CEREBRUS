"""One source row of each headroom-probe dataset -> (item_id, question, answers, question_type).

The question text holds everything the model sees besides the shared `question_suffix` prompt: for multiple choice the
options and the request for the option letter. It is not a config value, so probe rows keep the grid's config hash.
"""
from collections.abc import Sequence
from pathlib import Path

# lmms-eval's multiple-choice post prompt
LETTER_REQUEST = "\nAnswer with the option's letter from the given choices directly."

Record = tuple[str, str, tuple[str, ...], str]


def _clevr(row: dict, index: int) -> Record:
    return f"{Path(row['filename']).stem}_{index}", row["question"], (row["answer"],), row["question_type"]


def _chartqa(row: dict, index: int) -> Record:
    return str(index), row["question"], (row["answer"],), row["type"]


def _options(options: Sequence[str]) -> str:
    return "".join(f"\n{chr(65 + n)}. {text}" for n, text in enumerate(options))


def _ai2d(row: dict, index: int) -> Record:
    gold = int(row["answer"])  # index into `options`
    question = row["question"] + _options(row["options"]) + LETTER_REQUEST
    return str(index), question, (chr(65 + gold), row["options"][gold]), ""


def _mmstar(row: dict, index: int) -> Record:
    # the options are already part of the question
    return str(row["index"]), row["question"] + LETTER_REQUEST, (row["answer"].strip().upper(),), row["category"]


def _blink(row: dict, index: int) -> Record:
    if row.get("image_2") is not None:
        raise ValueError(f"BLINK {row['idx']} has several images: list only single-image subtasks in `files`")
    letter = row["answer"].strip("() ").upper()  # "(B)", or "C" in Relative_Reflectance
    # `prompt` is the question with its lettered choices
    return row["idx"], row["prompt"].strip() + LETTER_REQUEST, (letter, row["choices"][ord(letter) - 65]), row["sub_task"]


def _realworldqa(row: dict, index: int) -> Record:
    # the question already holds the options and asks for a letter or a word
    gold = row["answer"].strip()
    kind = "mc" if len(gold) == 1 and gold in "ABCDE" else "short"
    return str(index), row["question"], (gold,), kind


ROWS = {"clevr": _clevr, "chartqa": _chartqa, "ai2d": _ai2d, "mmstar": _mmstar, "blink": _blink,
        "realworldqa": _realworldqa}
# the image column, where it is not "image"
IMAGE_COLUMN = {"blink": "image_1"}
# datasets whose rows name their image; the others are identified by a hash of the image bytes
IMAGE_NAME = {"clevr": "filename"}
