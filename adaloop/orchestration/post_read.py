"""Read one post-training source (FineVision config or PixMo URLs) into a `SourceRun` until its target."""
import time
from collections.abc import Iterator

import pyarrow as pa

from adaloop.data.post.run import SourceRun, add_image
from adaloop.data.post.sources import column_bytes, fetch_images, parquet_files, retry, row_groups
from adaloop.data.post.text import pick_turn


def _groups(run: SourceRun, repo: str, folder: str) -> Iterator[tuple]:
    """Random files, random row groups, until the target or the byte budget is reached."""
    seed = f"{run.cfg['seed']}:{run.name}"
    fs, files = parquet_files(repo, folder, seed)
    run.files_total = len(files)
    for path in files:
        run.files_used += 1
        for parquet, group in row_groups(fs, path, seed, run.cfg["read_retries"]):
            if run.full or run.over_budget or run.skipped:
                return
            yield parquet, group, f"{path.rsplit('/', 1)[-1].removesuffix('.parquet')}/{group}"
        print(f"{run.name}: file {run.files_used}/{len(files)} | kept {run.counts['after_dup']}/{run.target} | "
              f"{run.counts['bytes_read'] / 1e9:.2f} GB")


def _read(run: SourceRun, parquet, group: int, columns: list[str]) -> pa.Table:
    run.counts["bytes_read"] += column_bytes(parquet, group, columns)
    return retry(lambda: parquet.read_row_group(group, columns=columns), f"{run.name} group {group}",
                 run.cfg["read_retries"])


def _read_finevision(run: SourceRun) -> None:
    """Text and rating columns first; images only for row groups that still have a row."""
    spec = run.cfg["buckets"][run.bucket]
    for parquet, group, where in _groups(run, run.cfg["repo"], run.name):
        columns = [name for name in parquet.schema_arrow.names if name == "texts" or name.endswith(("_min", "_ratings"))]
        picked = []
        for index, row in enumerate(_read(run, parquet, group, columns).to_pylist()):
            row_id = f"{run.name}/{where}/{index}"
            tally = {"english_turns": 0, "english_kept": 0}
            turn, why = pick_turn(row, run.cfg, spec.get("free", False), spec.get("count_only", False),
                                  f"{run.cfg['seed']}:{row_id}", run.source.get("english"), tally)
            unrated = turn is not None and turn["ratings"] is None
            picked.append((index, row_id, turn, why, unrated, tally))
        passing = [index for index, _, turn, *_ in picked if turn]
        images = {}
        if passing:
            column = _read(run, parquet, group, ["images"]).column("images").take(passing).to_pylist()
            images = dict(zip(passing, column))
        for index, row_id, turn, why, unrated, tally in picked:
            run.check_english()
            if run.full or run.dropped:
                break
            for key, n in tally.items():
                run.counts[key] += n
            run.counts["rows_read"] += 1
            run.counts["after_ratings"] += why != "ratings"
            if not turn:
                continue
            run.counts["after_answer"] += 1
            run.counts["unrated"] += unrated
            if len(images[index] or []) != 1:  # one image per item
                continue
            run.counts["after_download"] += 1
            add_image(run, images[index][0]["bytes"], turn, row_id)


def _url_turn(run: SourceRun, row: dict, row_id: str) -> dict | None:
    """A PixMo row as one FineVision-like turn through the same text rules."""
    spec = run.cfg["buckets"][run.bucket]
    if "question_template" in run.source:
        question, answer = run.source["question_template"].format(label=row["label"]), str(row["count"])
    else:
        question, answer = row["question"], row["answer"]
    turn, _ = pick_turn({"texts": [{"user": question, "assistant": answer}]}, run.cfg, spec.get("free", False),
                        spec.get("count_only", False), row_id)
    return turn


def _read_urls(run: SourceRun) -> None:
    """PixMo: filter the text, download only those images, verify sha256, give the source up if it fails too often."""
    cfg = run.cfg["downloads"]
    attempts, ok, seconds, urls = 0, 0, 0.0, set()
    for parquet, group, where in _groups(run, run.source["url_repo"], "data"):
        picked = []
        for index, row in enumerate(_read(run, parquet, group, parquet.schema_arrow.names).to_pylist()):
            row_id = f"{run.name}/{where}/{index}"
            run.counts["rows_read"] += 1
            run.counts["after_ratings"] += 1  # these sources carry no ratings
            turn = _url_turn(run, row, row_id)
            if turn and row["image_url"] not in urls:
                urls.add(row["image_url"])
                run.counts["after_answer"] += 1
                run.counts["unrated"] += 1
                picked.append((row, row_id, turn))
        for start in range(0, len(picked), cfg["batch"]):
            if run.full or run.over_budget:
                break
            batch = picked[start : start + cfg["batch"]]
            t0 = time.time()
            downloads = fetch_images([row for row, _, _ in batch], cfg)
            seconds += time.time() - t0
            attempts += len(batch)
            for (_, row_id, turn), data in zip(batch, downloads):
                if data is None:
                    continue
                ok += 1
                run.counts["bytes_read"] += len(data)
                run.counts["after_download"] += 1
                if not run.full:
                    add_image(run, data, turn, row_id)
            fail_rate, rate = 1 - ok / attempts, ok / max(seconds, 1e-9)
            if attempts >= cfg["check_after"] and (fail_rate > cfg["max_fail_rate"] or rate < cfg["min_ok_per_s"]):
                run.skipped = f"downloads given up: {fail_rate:.0%} failed of {attempts}, {rate:.2f} ok/s"
                print(f"{run.name}: {run.skipped}")
                break


def read_source(run: SourceRun) -> None:
    """Fill `run` from its source and write the last shard."""
    if run.target > 0:
        (_read_urls if "url_repo" in run.source else _read_finevision)(run)
    run.check_english(at_end=True)
    run.flush()
