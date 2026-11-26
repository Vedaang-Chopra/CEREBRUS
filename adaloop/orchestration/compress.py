"""One interface for every visual-token compression method: `compress`, chosen by `cfg["compressor"]`."""
import time
import zlib
from collections.abc import Mapping
from copy import copy

import torch
from PIL import Image

from adaloop.core.cost.compression import compression_flops
from adaloop.core.tokens.methods.crisp_lite import noun_relevance, question_nouns, word_embeddings
from adaloop.core.tokens.methods.divprune import DivPrune
from adaloop.core.tokens.methods.rediprune import _select_text_cond, extract_question_embedding
from adaloop.core.tokens.methods.visionzip import visionzip_merge
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.tokens.selectors import (
    fill_uniform,
    n_keep_for,
    question_embeddings,
    random_selector,
    text_mean_selector,
    uniform_selector,
)
from adaloop.core.tokens.vision import capture_attention_inputs, unit_cls_attention
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vision import VisionRun
from adaloop.schemas.vlm import LoadedVLM
from utils.images import resize_long_side
from utils.timing import cuda_sync

METHODS = ("none", "random", "uniform", "text_mean", "lowres", "crisp_lite", "rediprune", "divprune", "visionzip")


def _clock(vlm: LoadedVLM) -> float:
    """Wall clock after the copy's GPUs are idle."""
    if torch.cuda.is_available():
        cuda_sync(vlm.devices)
    return time.perf_counter()


def _n_visual(vlm: LoadedVLM, inputs: Mapping) -> int:
    return int((inputs["input_ids"] == vlm.model.config.image_token_id).sum())


def lowres_res_px(vlm: LoadedVLM, image: Image.Image, res_px: int, k: float) -> int:
    """The long side whose token count is closest to k times the count at `res_px` (ties: the larger side)."""
    patch = vlm.processor.image_processor.size
    blank = Image.new("L", image.size)  # same size, so the same resize rounding, but cheap

    def count(side: int) -> int:
        w, h = resize_long_side(blank, side).size
        return -(-h // patch["height"]) * -(-w // patch["width"])  # ceil of the grid

    top = min(res_px, max(image.size))
    if k >= 1:
        return res_px
    target = n_keep_for(count(top), k)
    return min(range(top, 0, -4), key=lambda side: abs(count(side) - target))


@torch.no_grad()
def encode_vision(vlm: LoadedVLM, inputs: Mapping) -> VisionRun:
    """Run the vision tower once on one item's `item_inputs`; pass the result to every `compress` call of the item."""
    start = _clock(vlm)
    with capture_attention_inputs(vlm) as captured:
        tokens, outputs = extract_visual_tokens(vlm, inputs)
    return VisionRun(tokens, outputs, captured, _clock(vlm) - start)


def _select(vlm: LoadedVLM, inputs: Mapping, item: Item, cfg: dict, method: str, k: float,
            vision: VisionRun, n_keep: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(kept indices, token features (changed only by visionzip), extras for the cost count and info)."""
    tokens = vision.tokens
    N = len(tokens)
    K = n_keep_for(N, k) if n_keep is None else n_keep  # an exact count skips the rounding of k * N
    if K == N:  # nothing to choose: the same packed path, with every token
        return torch.arange(N), tokens, {}
    params = cfg.get("compression", {}).get(method, {})
    language_model = vlm.model.model.language_model
    embed_tokens = language_model.get_input_embeddings()

    if method == "random":
        seed = cfg["splits"]["seed"] + zlib.crc32(item.item_id.encode())
        return random_selector(tokens, k, seed=seed % 2**31), tokens, {}
    if method == "uniform":
        return uniform_selector(inputs["image_grid_thw"], k), tokens, {}
    if method == "text_mean":
        return text_mean_selector(tokens, question_embeddings(vlm, item.question), k)[0], tokens, {}
    if method == "divprune":
        chosen, _ = DivPrune(tokens.float(), N, threshold_ratio=K / N)
        indices = chosen.unique()  # sorted
        if len(indices) != K:
            raise ValueError(f"DivPrune picked a token twice ({len(indices)} unique of {K}); item {item.item_id}")
        return indices, tokens, {}
    if method == "rediprune":
        ids = inputs["input_ids"].to(embed_tokens.weight.device)
        mask = inputs["attention_mask"].to(ids.device).bool()
        query = extract_question_embedding(embed_tokens, vlm.model.config.image_token_id,
                                           language_model.config.hidden_size, ids, mask)[0]
        # post-projector: tokens and question embedding have the same width (upstream scores before the projector)
        indices = _select_text_cond(tokens.float(), query.to(tokens.device), K / N, params["alpha"], params["tau"])
        tail = ids[0, (ids[0] == vlm.model.config.image_token_id).nonzero()[-1, 0] + 1:]
        return indices, tokens, {"n_words": min(len(tail), 128)}
    if method == "crisp_lite":
        _, h, w = inputs["image_grid_thw"][0].tolist()
        nouns = question_nouns(item.question, params["spacy_model"])
        if nouns:
            scores = noun_relevance(tokens, word_embeddings(embed_tokens, vlm.processor.tokenizer, nouns))
        else:
            scores = unit_cls_attention(vlm, vision.attention_inputs)[0]
        n_fill = int(K * params["uniform_frac"])
        relevant = scores.topk(K - n_fill).indices
        extras = {"n_words": len(nouns), "fallback": None if nouns else "no_noun_cls_attention"}
        return fill_uniform(h, w, relevant, K), tokens, extras
    # visionzip: contextual tokens are a fixed share of the full count; dominant tokens fill the rest of K
    n_contextual = min(max(int(params["contextual_frac"] * N), 1), K)
    attention, keys = unit_cls_attention(vlm, vision.attention_inputs)
    indices, merged = visionzip_merge(attention, keys, tokens,
                                      K - n_contextual, n_contextual)
    return indices, merged, {"n_contextual": n_contextual}


@torch.no_grad()  # not inference_mode: LoRA training (S9) feeds these tensors to autograd
def compress(vlm: LoadedVLM, inputs: Mapping, item: Item, k: float, cfg: dict,
             vision: VisionRun | None = None, n_keep: int | None = None) -> tuple[Mapping, dict]:
    """Return (model inputs for generation, info) for keep fraction `k` under `cfg["compressor"]`.

    `inputs` is `item_inputs(vlm, item, cfg)`. `none` returns it untouched; `lowres` prepares the item again at a
    smaller size; every other method returns packed inputs with `mm_encoder_outputs`, from `vision`
    (`encode_vision` of these inputs; computed here when None). Its time is reported as `vit_s` either way.
    info: method, k, res_px, grid_hw (units of the image the model gets), n_vis_full, n_vis, vit_s, compression_s,
    compression_flops, fallback.

    `n_keep` (divprune only) keeps exactly that many tokens and replaces `k`, which is then recorded as kept / full.
    An item with fewer tokens keeps them all and says so in `fallback`.
    """
    method = cfg.get("compressor", "none")
    if method not in METHODS:
        raise ValueError(f"unknown compressor {method!r}; one of {METHODS}")
    if method == "none" and k != 1.0:
        raise ValueError(f"compressor 'none' keeps every token, got k={k}")
    N = _n_visual(vlm, inputs)
    fewer = None
    if n_keep is not None:
        if method != "divprune" or n_keep < 1:
            raise ValueError(f"n_keep needs compressor divprune and a count >= 1, got {method!r}, {n_keep}")
        fewer = "fewer_tokens_than_n_keep_kept_all" if n_keep > N else None
        n_keep = min(n_keep, N)
        k = n_keep / N
    info = {"method": method, "k": k, "res_px": cfg["res_px"], "n_vis_full": N, "n_vis": N, "vit_s": None,
            "compression_s": 0.0, "compression_flops": 0, "fallback": None,
            "grid_hw": tuple(inputs["image_grid_thw"][0, 1:].tolist())}
    if method == "none":
        return inputs, info
    if method == "lowres":
        res_px = lowres_res_px(vlm, item.image, cfg["res_px"], k)
        if res_px != cfg["res_px"]:
            inputs = item_inputs(vlm, item, {**cfg, "res_px": res_px})
        grid_hw = tuple(inputs["image_grid_thw"][0, 1:].tolist())
        return inputs, {**info, "res_px": res_px, "n_vis": _n_visual(vlm, inputs), "grid_hw": grid_hw}

    ### Step-1:- The vision tower, once per item (the caller may share it)
    if vision is None:
        vision = encode_vision(vlm, inputs)
    tokens, vision_outputs = vision.tokens, vision.outputs

    ### Step-2:- Choose the tokens, then drop the others from the prompt and the features
    start = _clock(vlm)
    indices, features, extras = _select(vlm, inputs, item, cfg, method, k, vision, n_keep)
    if features is not tokens:
        vision_outputs = copy(vision_outputs)
        vision_outputs.pooler_output = (features,)
    packed = prune_inputs(vlm, inputs, vision_outputs, indices.to(tokens.device))
    end = _clock(vlm)

    tower = vlm.model.config.vision_config
    flops = compression_flops(
        method, N, len(indices), tokens.shape[-1], n_words=extras.get("n_words", 0),
        head_dim=tower.hidden_size // tower.num_attention_heads, n_heads=tower.num_attention_heads,
        n_patches_per_unit=tower.spatial_merge_size**2, n_contextual=extras.get("n_contextual", 0))
    info.update(n_vis=len(indices), vit_s=vision.seconds, compression_s=end - start, compression_flops=flops,
                fallback=extras.get("fallback") or fewer)
    return packed, info
