# utils

Stateless shared helpers. Must not import `adaloop`.

- `io.load_config`
- `images.load_image`, `images.resize_long_side`
- `hashing.config_hash`, `hashing.git_hash` (provenance on every result row)
- `timing.cuda_sync`, `timing.reset_peak_memory`, `timing.peak_memory_gb`, `timing.FirstTokenTimer` (TTFT streamer)
