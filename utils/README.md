# utils

Stateless shared helpers. Must not import `adaloop`.

- `env.setup_notebook` (find repo root, chdir, sys.path, `env.env`, logging, config; checks but does not select GPUs: warns if `CUDA_VISIBLE_DEVICES` is not one of `gpu_groups` or their union; returns `(ROOT, cfg)`; `overrides` replaces top-level config keys)
- `gpus.select_gpu_group`, `gpus.free_gpu_mib` (free memory from `nvidia-smi`; sets `CUDA_VISIBLE_DEVICES` to one of `gpu_groups`, or with `group="all"` their union, whose GPUs have `min_free_gib` free; takes a config path, relative to the repo root, or a dict; call before anything touches CUDA, including `torch.cuda.is_available` and `import adaloop.api`. Notebooks import it from `utils.gpus`, the one import allowed outside `adaloop.api`, because the GPU choice must precede CUDA init. Its import chain stays free of torch and transformers)
- `gpus.claim_gpu_group(cfg, lock_dir)`: a free group that is also `flock`ed until the process exits (parallel `scripts/mvp.py train` runs)
- `io.load_config` (a top-level `extends: <file>` loads the parent from the same folder; child keys win, dict-valued keys merge one level)
- `images.load_image`, `images.resize_long_side`
- `hashing.config_hash`, `hashing.git_hash` (provenance on every result row; a copy without `.git` reads `REVISION` at the repo root)
- `servers.apply_server_overlay` (called by `load_config`: `ADALOOP_SERVER=<name>` applies the unhashed placement and batching keys of `configs/servers/<name>.yaml` to run configs; set it in the shell or Jupyter environment, not in `env.env`, which is read after the GPU choice), `servers.server_name`, `servers.is_gpu_host`
- `timing.cuda_sync`, `timing.reset_peak_memory`, `timing.peak_memory_gb` (each takes the GPUs to act on, default all), `timing.FirstTokenTimer` (TTFT streamer), `timing.DeviceClock` (per-device marks: CUDA events, wall clock on CPU)
- `gpus.select_for_notebook(config)` -> `(gpu_ids, n_copies)`: the notebook header call (before `adaloop.api` is imported)
- `gpus.full_gpu_overrides(config)` -> `{gpu_groups: full_gpu_groups, n_copies}`: nb07 `ALL_GPUS`; pass it to `select_for_notebook` and `setup_notebook`
