"""What each S9 training sample is: item, keep fraction k, student loops R, and the teacher's loop count.

The item, k and R depend only on the seed and the sample number, so every variant trains on the same sequence and a
resumed run repeats it. A variant differs only in its teacher-depth rule `lora.teacher_R`.
"""
import random
import re
from collections.abc import Sequence

from adaloop.core.model.loops import MAX_LOOPS
from adaloop.schemas.train import TrainSample


def parse_teacher_rule(rule: str) -> tuple[str, int | None]:
    """("same", None), ("fixed", N) or ("plus", N) from "same" | "fixed:N" | "plus:N"."""
    if rule == "same":
        return "same", None
    found = re.fullmatch(r"(fixed|plus):([1-9])", rule) if isinstance(rule, str) else None
    if found:
        kind, N = found[1], int(found[2])
        if N <= MAX_LOOPS:
            return kind, N
    raise ValueError(f'lora.teacher_R must be "same", "fixed:N" or "plus:N" with N in 1..{MAX_LOOPS}, got {rule!r}')


def teacher_depth(rule: str, R: int, k: float) -> int:
    """Teacher loop count of a student at (R, k). Full-token samples (k = 1) always use the student's R."""
    kind, N = parse_teacher_rule(rule)
    if k >= 1.0 or kind == "same":
        return R
    return N if kind == "fixed" else min(R + N, MAX_LOOPS)


def variant_name(cfg: dict) -> str:
    """Name of the variant, from its teacher rule; also its folder name."""
    kind, N = parse_teacher_rule(cfg["lora"]["teacher_R"])
    return "lora_depth_matched" if kind == "same" else f"lora_cross_depth_{kind}{N}"


def variant_config(cfg: dict, key: str) -> dict:
    """The config of variant `key` of cfg["lora_variants"] (A, B)."""
    rules = cfg["lora_variants"]
    if key not in rules:
        raise ValueError(f"unknown variant {key!r}; one of {list(rules)}")
    parse_teacher_rule(rules[key])
    return {**cfg, "lora": {**cfg["lora"], "teacher_R": rules[key]}}


def sample_budget(index: int, cfg: dict) -> tuple[float, int]:
    """(k, student R) of sample number `index`; the teacher rule plays no part."""
    lora = cfg["lora"]
    rng = random.Random(f"{lora['seed']}-sample-{index}")
    k = 1.0 if rng.random() < lora["identity_frac"] else rng.choice(lora["k_train"])
    return k, rng.choice(lora["R_choices"])


def _item_position(index: int, n_items: int, seed: int) -> int:
    """Item of sample `index`: a new seeded shuffle of all items every epoch."""
    order = random.Random(f"{seed}-epoch-{index // n_items}").sample(range(n_items), n_items)
    return order[index % n_items]


def train_sample(index: int, n_items: int, cfg: dict) -> TrainSample:
    """Everything about sample number `index` of a run over `n_items` items."""
    k, R = sample_budget(index, cfg)
    return TrainSample(index=index, item_n=_item_position(index, n_items, cfg["lora"]["seed"]), k=k, R=R,
                       R_teacher=teacher_depth(cfg["lora"]["teacher_R"], R, k))


def pairing_table(cfg: dict, rules: Sequence[str]) -> list[dict]:
    """The teacher depth of every (rule, k, student R) the schedule can draw."""
    return [{"rule": rule, "k": k, "R": R, "R_teacher": teacher_depth(rule, R, k)}
            for rule in rules for k in [*cfg["lora"]["k_train"], 1.0] for R in cfg["lora"]["R_choices"]]
