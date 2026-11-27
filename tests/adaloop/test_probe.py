"""Headroom probe: scorers on hand cases, row mappers, config hash, splits / loader on local files, report."""
import io
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from adaloop import api
from adaloop.core.eval.extra_scorers import score_chartqa, score_clevr, score_mc, score_realworldqa
from adaloop.data.extra_rows import ROWS
from utils.paths import find_repo_root

ROOT = find_repo_root()


@pytest.mark.parametrize("pred, answers, want", [
    ("B", ("B",), 1), ("b", ("B",), 1), ("(B)", ("B",), 1), ("B.", ("B",), 1), ("B: the cat", ("B",), 1),
    ("A", ("B",), 0), ("A dog", ("A",), 0), (None, ("A",), 0), ("", ("A",), 0),
    ("Box B", ("B", "Box B"), 1), ("box b.", ("B", "Box B"), 1), ("Box A", ("B", "Box B"), 0),
    ("c", ("A", "c"), 0),  # a lone letter is read as the option letter, not as a diagram label
])
def test_mc(pred, answers, want):
    assert score_mc(pred, answers) == want


@pytest.mark.parametrize("pred, gold, want", [
    ("14", "14", 1), ("14.", "14", 1), ("104", "100", 1), ("105", "100", 1), ("106", "100", 0), ("94", "100", 0),
    ("0.57", "0.57", 1), ("0.6", "0.57", 0), ("-10.4", "-10", 1), ("0", "0", 1), ("0.01", "0", 0),
    ("62%", "62%", 1), ("62%", "62", 0), ("yes", "Yes", 1), ("no", "Yes", 0), (None, "1", 0),
])
def test_chartqa(pred, gold, want):
    assert score_chartqa(pred, (gold,)) == want


@pytest.mark.parametrize("pred, gold, want", [
    ("yes", "yes", 1), ("Yes.", "yes", 1), ("three", "3", 1), ("3", "3", 1), ("2", "3", 0), ("cube", "cube", 1),
    ("cubes", "cube", 0), (" ", "yes", 0), (None, "yes", 0),
])
def test_clevr(pred, gold, want):
    assert score_clevr(pred, (gold,)) == want


@pytest.mark.parametrize("pred, gold, want", [
    ("C", "C", 1), ("(c)", "C", 1), ("B", "C", 0), ("downhill", "Downhill", 1), ("Downhill.", "Downhill", 1),
    ("0", "0", 1), ("1", "0", 0), ("Yes", "No", 0), (None, "C", 0),
])
def test_realworldqa(pred, gold, want):
    assert score_realworldqa(pred, (gold,)) == want


def test_rows():
    _, question, answers, _ = ROWS["ai2d"]({"question": "Q?", "options": ["c", "D"], "answer": "1"}, 4)
    assert question.startswith("Q?\nA. c\nB. D\n") and answers == ("B", "D")
    row = {"idx": "val_x_1", "prompt": "P (A) yes (B) no", "answer": "(B)", "choices": ["yes", "no"],
           "sub_task": "s", "image_2": None}
    assert ROWS["blink"](row, 0)[2] == ("B", "no")
    assert ROWS["blink"]({**row, "answer": "A"}, 0)[2] == ("A", "yes")
    assert ROWS["realworldqa"]({"question": "q", "answer": "B"}, 0)[3] == "mc"


def test_hash_is_the_grids():
    mvp, probe = (api.load_config(ROOT / "configs" / name) for name in ("mvp.yaml", "probe.yaml"))
    assert "extra_benchmarks" in probe
    assert api.config_hash(probe) == api.config_hash(mvp)
    assert api.config_hash(api.grid_config(probe)) == api.config_hash(api.grid_config(mvp))
    assert api.cell_subset(probe, [0.25, 1.0], [1, 3, 5])["R_grid"] == [1, 3, 5]
    with pytest.raises(ValueError):
        api.cell_subset(probe, [0.3], [1])


def _png(seed: int) -> bytes:
    image = Image.effect_mandelbrot((64, 48), (-2 + 0.13 * seed, -1.5, 1 + 0.07 * seed, 1.5), 40 + seed).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_splits_and_loader(tmp_path):
    n = 20
    rows = {"image": [{"bytes": _png(i // 2), "path": None} for i in range(n)], "question": [f"q{i}" for i in range(n)],
            "answer": [str(i) for i in range(n)], "type": ["t"] * n}
    # written with pyarrow, with the metadata `datasets` stores for an image column
    meta = {"info": {"features": {"image": {"_type": "Image"}, **{name: {"dtype": "string", "_type": "Value"}
                                                                 for name in ("question", "answer", "type")}}}}
    table = pa.Table.from_pydict(rows, schema=pa.schema(
        [("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])), ("question", pa.string()),
         ("answer", pa.string()), ("type", pa.string())], metadata={"huggingface": json.dumps(meta)}))
    pq.write_table(table, tmp_path / "part.parquet")
    cfg = {"splits": {"seed": 42}, "extra_benchmarks": {"n_items": 4, "sets": {"chartqa": {
        "repo": str(tmp_path), "files": ["part.parquet"], "image_frac": 0.5, "manifest": "m.json"}}}}
    eval_image = Image.open(io.BytesIO(_png(0))).convert("RGB")
    hashes = [(api.phash(eval_image), api.dhash(eval_image))]
    # distance 0: only the identical image (rows 0 and 1) counts as an eval image
    manifest = api.build_extra_splits(cfg, "chartqa", hashes, {"phash_max": 0, "dhash_max": 0}, tmp_path)
    assert manifest["dropped_near_eval"]
    ids = {split: set(q) for split, q in manifest["splits"].items()}
    assert len(ids["probe"]) == 4 and not ids["probe"] & ids["pool"] and not {"0", "1"} & (ids["probe"] | ids["pool"])
    assert not set(manifest["images"]["probe"]) & set(manifest["images"]["pool"])
    items = api.load_extra(cfg, "chartqa", root=tmp_path)
    assert [item.item_id for item in items] == manifest["splits"]["probe"]
    assert items[0].benchmark == "chartqa" and items[0].split == "probe" and items[0].image.mode == "RGB"
    assert items[0].answers == (items[0].item_id,)


def test_report():
    cfg = {"extra_benchmarks": {"ks": [0.25, 1.0], "Rs": [1, 3], "depth_k": 1.0, "look_R": 3}}
    # item a: right only with full tokens; item b: right only from R=3 on at k=1; items c, d: always right
    right = {"a": {(1.0, 1): 1, (1.0, 3): 1}, "b": {(1.0, 3): 1, (0.25, 3): 1},
             "c": {cell: 1 for cell in [(0.25, 1), (0.25, 3), (1.0, 1), (1.0, 3)]}}
    right["d"] = right["c"]
    rows = [{"benchmark": "x", "item_id": item, "model_variant": "base", "rerun_id": rerun, "k": k, "R": R,
             "correct": float(right[item].get((k, R), 0))}
            for item in right for rerun in (0, 1) if rerun == 0 or item in "cd"
            for k in (0.25, 1.0) for R in (1, 3)]
    report = api.probe_report(pd.DataFrame(rows), cfg).iloc[0]
    assert (report["n_items"], report["n_rerun"]) == (4, 2)
    assert report["depth_change"] == 25 and report["depth_noise"] == 0 and report["depth_headroom"] == 25
    assert report["look_gain"] == 25 and report["look_headroom"] == 25
    assert report["best_cell"] == "k1.0_R3" and report["best_cell_acc"] == 100 and report["oracle_gain"] == 0
