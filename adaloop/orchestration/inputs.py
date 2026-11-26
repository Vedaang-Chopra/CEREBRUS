"""Model inputs of one benchmark item."""
from collections.abc import Mapping

from adaloop.core.model.prompting import prepare_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.images import resize_long_side


def item_prompt(item: Item, cfg: dict) -> str:
    """The question text sent to the model (capitalized for TextVQA) plus the `question_suffix` prompt."""
    # TextVQA: lmms-eval applies str.capitalize (first letter upper, the rest lower). Ported from
    # EvolvingLMMs-Lab/lmms-eval:lmms_eval/tasks/textvqa/utils.py::textvqa_doc_to_text
    # @ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef.
    question = item.question.capitalize() if item.benchmark == "textvqa" else item.question
    return question + cfg["prompts"]["question_suffix"]


def item_inputs(vlm: LoadedVLM, item: Item, cfg: dict) -> Mapping:
    """Model inputs for one item (system prompt, resized image, question, effort, prefill); every run uses this."""
    image = resize_long_side(item.image, cfg["res_px"])
    return prepare_inputs(vlm, item_prompt(item, cfg), image, prefill=cfg.get("answer_prefill"),
                          system=cfg["prompts"]["system"], reasoning_effort=cfg["reasoning_effort"])
