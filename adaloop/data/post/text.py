"""Text rules: ratings (filter 1), question and answer cleaning (filter 2), multi-reference answers (filter 6)."""
import random
import re
from collections import Counter

CJK = "぀-ヿ一-鿿가-힯"
TOKEN = re.compile(f"[{CJK}]|[^\\s{CJK}]+")
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[。！？]")
COUNT_QUESTION = re.compile(r"how many|number of|\bcount\b", re.I)
NUMBER = re.compile(r"\d[\d,]*|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve", re.I)


def token_count(text: str) -> int:
    """Answer length without a model: whitespace words, each CJK character counted as one."""
    return len(TOKEN.findall(text))


def clean_question(text: str, style_words: list[str]) -> str:
    """Plain question text: no <image> tag, no style suffix; with "Options:", only the question and the options."""
    lines = [line.strip() for line in text.replace("<image>", "").split("\n") if line.strip()]
    options = [line for line in lines[1:] if line.startswith("Options:")]
    if options:  # a-okvqa puts a (multi-line) answer-format instruction between the question and its options
        return "\n".join([lines[0], *options])
    style = re.compile("|".join(rf"\b{re.escape(word)}\b" for word in style_words), re.I)
    # choice lines ("A. short") are content, not style
    while len(lines) > 1 and "?" not in lines[-1] and style.search(lines[-1]) and not re.match(r"[A-H]\.", lines[-1]):
        lines.pop()
    return "\n".join(lines)


def clean_short_answer(text: str) -> str:
    """The first line after the last "Answer:" (if any), without the trailing period."""
    text = text.strip()
    if "Answer:" in text:  # CoSyn reasons first; a-okvqa adds a "Rationale:" line after
        text = text.rsplit("Answer:", 1)[1].strip().split("\n")[0]
    text = text.strip().strip("`").strip()
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip()
    return text


def cap_free_answer(text: str, max_tokens: int) -> str | None:
    """The longest prefix of whole sentences within `max_tokens`; None if even the first sentence is longer."""
    text = text.strip().removeprefix("Answer:").strip()
    if token_count(text) <= max_tokens:
        return text or None
    best = None
    for match in SENTENCE_END.finditer(text):
        if token_count(text[: match.end()]) > max_tokens:
            break
        best = text[: match.end()]
    return best


def is_english(question: str, answer: str, min_ascii: float) -> bool:
    """Mostly ASCII letters in both; digits and punctuation are ignored, an answer without letters (a number) passes."""
    def ok(text: str, if_no_letters: bool) -> bool:
        letters = [char for char in text if char.isalpha()]
        return sum(char.isascii() for char in letters) >= min_ascii * len(letters) if letters else if_no_letters

    return ok(question, False) and ok(answer, True)


def turn_ratings(row: dict, index: int, cfg: dict) -> tuple[dict | None, bool]:
    """(ratings of turn `index`, passes). A row or turn without a rating passes (ratings only drop the worst)."""
    values = {}
    for key in [*cfg["keys"], "visual_dependency"]:
        per_turn = row.get(f"{key}_ratings")
        if cfg["level"] == "row" or not per_turn or len(per_turn) != len(row["texts"]):
            values[key] = row.get(f"{key}_min")
        else:
            values[key] = per_turn[index]
    if all(values[key] is None for key in cfg["keys"]):
        return None, True
    return values, all(values[key] is None or values[key] >= cfg["min"] for key in cfg["keys"])


def majority_turns(turns: list[tuple]) -> list[tuple]:
    """Keep one turn per question; repeated questions need one answer with more than half of the references.

    `turns` are tuples whose first two entries are (question, answer).
    """
    groups: dict[str, list[tuple]] = {}
    for turn in turns:
        groups.setdefault(turn[0].lower(), []).append(turn)
    kept = []
    for group in groups.values():
        answer, n = Counter(turn[1].lower() for turn in group).most_common(1)[0]
        if n * 2 > len(group):
            kept.append(next(turn for turn in group if turn[1].lower() == answer))
    return kept


def pick_turn(row: dict, cfg: dict, free: bool, count_only: bool, seed: str, english: dict | None = None,
              tally: dict | None = None) -> tuple[dict | None, str]:
    """One turn of a FineVision row (filters 1, 2, 6): (turn, "") or (None, "ratings" | "answer").

    `english` is a source's English filter; `tally` gets the turns that reached it and the turns it kept.
    """
    answers_cfg = cfg["answers"]
    turns, any_rated_ok = [], False
    for index, text in enumerate(row["texts"] or []):
        ratings, ok = turn_ratings(row, index, cfg["ratings"])
        if not ok:
            continue
        any_rated_ok = True
        question = clean_question(text["user"] or "", answers_cfg["style_words"])
        full = (text["assistant"] or "").strip()
        answer = cap_free_answer(full, answers_cfg["free_max_tokens"]) if free else clean_short_answer(full)
        if not (question and answer):
            continue
        if english:
            passed = is_english(question, answer, english["min_ascii_letters"])
            tally["english_turns"] += 1
            tally["english_kept"] += passed
            if not passed:
                continue
        turns.append((question, answer, full, ratings))
    if not any_rated_ok:
        return None, "ratings"
    usable = []
    for question, answer, full, ratings in majority_turns(turns):
        if not free and token_count(answer) > answers_cfg["short_max_tokens"]:
            continue
        is_count = bool(COUNT_QUESTION.search(question) and NUMBER.fullmatch(answer))
        if count_only and not is_count:
            continue
        kind = "free" if free else "count" if is_count else "short"
        usable.append({"question": question, "answer": answer, "answer_full": full, "answer_type": kind, "ratings": ratings})
    if not usable:
        return None, "answer"
    return random.Random(seed).choice(usable), ""
