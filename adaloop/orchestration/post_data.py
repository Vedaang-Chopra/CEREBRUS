"""Build the post-training candidate store: per bucket and source, a target, a read, a state file; then val."""
import math
import time
from pathlib import Path

from adaloop.data.post.eval_hashes import eval_hashes
from adaloop.data.post.run import SourceRun
from adaloop.data.post.sources import refuse_excluded
from adaloop.data.post.store import assign_splits, clear_source, load_post_data, load_states, save_state, write_summary
from adaloop.orchestration.post_read import read_source
from utils.hashing import config_hash
from utils.io import load_config


def source_target(cfg: dict, bucket: str, index: int, got: int) -> int:
    """Candidates wanted from source `index` of `bucket`, given `got` from the sources before it."""
    spec = cfg["buckets"][bucket]
    source = spec["sources"][index]
    missing = max(cfg["candidate_factor"] * spec["target"] - got, 0)
    sharing = [s for s in spec["sources"][index:] if not s.get("fallback")]
    share = missing if source.get("fallback") or not sharing else math.ceil(missing / len(sharing))
    if "cap" in source:
        share = min(share, cfg["candidate_factor"] * source["cap"])
    return share


def build_source(cfg: dict, folder: Path, bucket: str, source: dict, target: int, blocked: list, seen: set) -> dict:
    """Read one source until `target` candidates, write its shards, save and return its state."""
    refuse_excluded(source["config"])
    clear_source(folder, source["config"])
    run = SourceRun(cfg, folder, bucket, source, target, blocked, seen)
    t0 = time.time()
    print(f"[STAGE] {bucket} / {run.name}: target {target}")
    read_source(run)
    state = {"source_config": run.name, "bucket": bucket, "target": target, "kept": run.counts["after_dup"],
             **run.counts, "eval_dup_dropped": run.counts["after_image"] - run.counts["after_eval"],
             "dup_dropped": run.counts["after_eval"] - run.counts["after_dup"], "seconds": round(time.time() - t0, 1),
             "files_used": run.files_used, "files_total": run.files_total, "shards": run.shards,
             "note": run.skipped or ("byte budget reached" if run.over_budget else
                                     "source exhausted" if run.counts["after_dup"] < target else ""),
             "config_hash": _settings_hash(cfg)}
    save_state(folder, run.name, state)
    print(f"{run.name}: kept {state['kept']}/{target} from {state['rows_read']} rows | eval overlap "
          f"{state['eval_dup_dropped']} | duplicates {state['dup_dropped']} | {state['bytes_read'] / 1e9:.2f} GB | "
          f"{state['seconds']:.0f} s")
    return state


def _settings_hash(cfg: dict) -> str:
    return config_hash({k: v for k, v in cfg.items() if k not in ("out_dir", "smoke_rows")})


def post_data_dir(cfg: dict, root: str | Path, out_dir: str | Path | None = None) -> Path:
    """The store folder; a relative path is under `root`."""
    return Path(root) / (out_dir or cfg["out_dir"])


def post_data_eval_hashes(cfg: dict, root: str | Path, out_dir: str | Path | None = None) -> list[tuple[int, int]]:
    """(pHash, dHash) of all eval images, cached next to the store."""
    return eval_hashes(load_config(Path(root) / cfg["benchmarks_config"]), cfg["eval_splits"], root,
                       post_data_dir(cfg, root, out_dir) / "eval_hashes.json")


def build_post_data(cfg: dict, root: str | Path, out_dir: str | Path | None = None, config_name: str | None = None,
                    rows: int | None = None) -> Path:
    """Build the whole store (resumable), or smoke-run one source into `<out>/smoke`; return the store folder."""
    sources = [(bucket, i, s) for bucket, spec in cfg["buckets"].items() for i, s in enumerate(spec["sources"])]
    for name in [s["config"] for _, _, s in sources] + ([config_name] if config_name else []):
        refuse_excluded(name)
    folder = post_data_dir(cfg, root, out_dir)
    if config_name:
        match = [(b, s) for b, _, s in sources if s["config"] == config_name]
        if not match:
            raise ValueError(f"unknown source {config_name!r}; configured: {[s['config'] for _, _, s in sources]}")
        blocked = post_data_eval_hashes(cfg, root, out_dir)
        smoke = folder / "smoke"
        build_source(cfg, smoke, *match[0], rows or cfg["smoke_rows"], blocked, set())
        write_summary(smoke)
        return smoke

    states = load_states(folder)
    stale = [name for name, state in states.items() if state["config_hash"] != _settings_hash(cfg)]
    if stale:
        raise ValueError(f"{folder} was built with other settings ({stale}); use a new output folder")
    blocked = post_data_eval_hashes(cfg, root, out_dir)
    seen = set()
    if any(state["shards"] for state in states.values()):
        stored = load_post_data(folder, ["phash", "dhash"]).to_pylist()
        seen = {(int(r["phash"], 16), int(r["dhash"], 16)) for r in stored}
    for bucket, index, source in sources:
        name = source["config"]
        if name in states:
            print(f"{name}: finished ({states[name]['kept']} kept), not read again")
            continue
        got = sum(s["kept"] for s in states.values() if s["bucket"] == bucket)
        target = source_target(cfg, bucket, index, got)
        states[name] = build_source(cfg, folder, bucket, source, target, blocked, seen)
        write_summary(folder)
    val = assign_splits(folder, cfg["val_per_bucket"], cfg["seed"])
    for bucket, spec in cfg["buckets"].items():
        kept = sum(s["kept"] for s in states.values() if s["bucket"] == bucket)
        print(f"{bucket}: {kept}/{cfg['candidate_factor'] * spec['target']} candidates, {val[bucket]} val")
    print(f"summary: {write_summary(folder)}")
    return folder
