"""What the S9 student sees: cached teacher targets and compressed inputs, and the loss of one item."""
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from adaloop.core.train.distill import distill_loss
from adaloop.core.train.schedule import teacher_depth, train_sample
from adaloop.orchestration.compress import compress
from adaloop.orchestration.inputs import item_inputs
from adaloop.orchestration.teacher_cache import teacher_records, teacher_run_config
from adaloop.schemas.items import Item
from adaloop.schemas.train import TeacherTarget
from adaloop.schemas.vlm import LoadedVLM

Targets = Mapping[tuple[str, str, int], TeacherTarget]  # (benchmark, item_id, R_teacher) -> record


def load_targets(items: Sequence[Item], cfg: dict, root: str | Path) -> dict[tuple[str, str, int], TeacherTarget]:
    """The cached records of `items` at every R of `teacher.R_list`; an error if any is missing."""
    rows, tensors = teacher_records(cfg, root, items)
    targets = {
        (row["benchmark"], row["item_id"], row["R"]): TeacherTarget(
            benchmark=row["benchmark"], item_id=row["item_id"], R=row["R"], **record)
        for row, record in zip(rows, tensors, strict=True)
    }
    missing = [(item.benchmark, item.item_id, R) for item in items for R in cfg["teacher"]["R_list"]
               if (item.benchmark, item.item_id, R) not in targets]
    if missing:
        raise ValueError(f"teacher cache lacks {len(missing)} (benchmark, item, R) records, e.g. {missing[:3]}; "
                         "run the S8 cache for these items")
    print(f"teacher targets: {len(targets)} records of {len(items)} items")
    return targets


def target_for(targets: Targets, item: Item, R_teacher: int) -> TeacherTarget:
    """The record of `item` at the teacher's loop count."""
    key = (item.benchmark, item.item_id, R_teacher)
    if key not in targets:
        raise KeyError(f"no teacher record for {key}")
    return targets[key]


def student_inputs(vlm: LoadedVLM, item: Item, cfg: dict, k: float) -> Mapping:
    """The teacher's prompt with a fraction `k` of the visual tokens, under cfg["compressor"]."""
    inputs = item_inputs(vlm, item, teacher_run_config(cfg))
    return compress(vlm, inputs, item, k, cfg)[0]


def item_loss(vlm: LoadedVLM, item: Item, targets: Targets, cfg: dict, k: float, R: int) -> torch.Tensor:
    """Loss of the student at (k, R) against the teacher depth that cfg["lora"]["teacher_R"] gives."""
    target = target_for(targets, item, teacher_depth(cfg["lora"]["teacher_R"], R, k))
    return distill_loss(vlm, student_inputs(vlm, item, cfg, k), target, R)


def schedule_table(items: Sequence[Item], targets: Targets, cfg: dict, n: int) -> list[dict]:
    """The first `n` training samples with the loop count of the record each one is paired with."""
    rows = []
    for index in range(n):
        sample = train_sample(index, len(items), cfg)
        item = items[sample.item_n]
        record = target_for(targets, item, sample.R_teacher)
        rows.append({"index": index, "benchmark": item.benchmark, "item_id": item.item_id, "k": sample.k,
                     "R": sample.R, "R_teacher": sample.R_teacher, "record_item_id": record.item_id,
                     "record_R": record.R, "n_answer_tokens": len(record.answer_ids)})
    return rows
