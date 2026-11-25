# utils

Stateless shared helpers. Must not import `adaloop`.

- `env.setup_notebook` (find repo root, chdir, sys.path, `env.env`, logging, config; checks but does not select GPUs: warns if `CUDA_VISIBLE_DEVICES` is not one of `gpu_groups` or their union; returns `(ROOT, cfg)`)
- `gpus.select_gpu_group`, `gpus.free_gpu_mib` (free memory from `nvidia-smi`; sets `CUDA_VISIBLE_DEVICES` to one of `gpu_groups`, or with `group="all"` their union, whose GPUs have `min_free_gib` free; takes a config path, relative to the repo root, or a dict; call before anything touches CUDA, including `torch.cuda.is_available` and `import adaloop.api`. Notebooks import it from `utils.gpus`, the one import allowed outside `adaloop.api`, because the GPU choice must precede CUDA init. Its import chain stays free of torch and transformers)
- `io.load_config` (a top-level `extends: <file>` loads the parent from the same folder; child keys win, dict-valued keys merge one level)
- `images.load_image`, `images.resize_long_side`
- `hashing.config_hash`, `hashing.git_hash` (provenance on every result row)
- `timing.cuda_sync`, `timing.reset_peak_memory`, `timing.peak_memory_gb` (each takes the GPUs to act on, default all), `timing.FirstTokenTimer` (TTFT streamer)
