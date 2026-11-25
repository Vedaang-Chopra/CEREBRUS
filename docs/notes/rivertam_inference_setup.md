# Rivertam inference server — setup plan (handoff for one agent session)

Read this whole file, then execute it top to bottom. Report to the user after each milestone.

## Goal

Serve `thinkingmachines/TML-d24-varloop-research` on Rivertam using **all 8 GPUs as multiple model copies**,
behind one OpenAI-compatible endpoint, and **call it from the user's laptop**.

Done when, from the laptop:
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)   # via ssh tunnel
c.models.list()                                                # shows the served model(s)
c.chat.completions.create(model="inkling-loop-3", messages=[{"role": "user", "content": "2+2?"}])
```
works, 2 copies are serving (all 8 GPUs used), and a short concurrency benchmark is recorded.

Out of scope this session: Prometheus/Grafana, PACE workers, RL weight sync, auto model swapping, other models.

## Hard rules

- **Everything lives under `/coc/scratch/vchopra/host_models`**: env, model copy, caches, logs, tmp, configs, scripts.
  Never write to `/nethome/dgarg39` (shared, 78% full) or `/home`.
- The `dgarg39` account and its conda (`/coc/scratch/dgarg/miniconda3`) are **shared with other people**.
  Do not modify any existing env (`dg_vllm032`, `dg_metaserve`, ...), `~/.bashrc`, or other users' dirs.
  Use the base `conda` only to create a new env with `-p`.
- Do not touch `/coc/scratch/vchopra/learning_env` except to **read** the model cache.
- Before using GPUs, check `nvidia-smi` — if someone else is on a GPU, stop and ask the user.
- Research correctness > speed. Never report a loop count you did not verify. One served model name = one exact config.
- Max 2 attempts per approach, then stop and report (user's global rule). Ask before `rm -rf`.
- Global agent rules: `~/.claude/CLAUDE.md`. Project rules: `AGENTS.md` (this repo; the infra itself is NOT part of this repo).

## Access

- From the laptop: `ssh rivertam` (alias in `~/.ssh/config`, user `dgarg39`, host `rivertam.cc.gatech.edu`). Passwordless, works with `-o BatchMode=yes`.
- `/coc/scratch` is the same NFS mount on Rivertam and Inara (`ssh inara`). It is slow for big greps — target exact paths.
- Long-running processes: no systemd (`Linger=no`). Use `tmux` sessions on Rivertam (check `which tmux`; fall back to `nohup` + pid files).

## Facts already verified (2026-11-25)

| Item | Value |
|---|---|
| GPUs | 8× RTX 2080 Ti, 11 GiB, compute capability **7.5 (Turing)**, all idle, driver 575.57.08 / CUDA 12.9 |
| Topology | **two PCIe-switch islands**: GPUs 0–3 (NUMA 0) and 4–7 (NUMA 1); `SYS` between islands; no NVLink |
| CPU / RAM | 48 cores, 376 GB RAM (~367 GB free) |
| Disk | `/coc/scratch` NFS, 11 TB free |
| Model cache | `/coc/scratch/vchopra/learning_env/Deep-Learning/model_training/.hf/hub/models--thinkingmachines--TML-d24-varloop-research` — **3 snapshots** (`657b53ba…`, `91a57fe2…`, `c90921bd…`); snapshot ~18 GB |
| Model config | arch `Inkling2LoopForConditionalGeneration`; `quantization_config.quant_method = modelopt`; `text_config.loop_config = {core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64, is_varloop: True}` |
| Model card | 26.4B total / 3.4B active MoE, 24 layers, middle 12 looped, trained with 1–5 passes (post-trained at 3); experts NVFP4, last 4 layers MXFP8, rest BF16; multimodal (ViT); hybrid local/global attention. Card text: `docs/literature/model_cards/2026_TML-d24-varloop-research-model-card.txt` |
| Existing vLLM | `dg_vllm032` = vLLM 0.32.0+cu129: **no Inkling model class**; ModelOpt quant `get_min_capability() = 80` (blocks Turing); but Marlin FP4 and FP8 kernels allow capability **75** |
| HF behaviour (Inara) | loop count is a runtime knob: set `text_config.loop_num_passes` and `language_model.config.loop_num_passes` (`adaloop/core/model/loops.py`). Setting it at load time via `text_config=` was ignored. |

**Main risk:** the checkpoint's native formats (NVFP4/MXFP8/BF16) are not officially supported on Turing.
Turing has no BF16 math, so BF16 weights/activations must run as FP16 (overflow risk).

## Target layout

```
/coc/scratch/vchopra/host_models/
  envs/vllm/              conda env (python 3.12) with the chosen vLLM
  models/TML-d24-varloop-research/<revision>/   real files (no symlinks), one pinned revision
  cache/{hf,vllm,triton,inductor,pip,xdg}/      all caches
  tmp/                    TMPDIR
  logs/                   one log per server process, timestamped
  configs/serve.env       ports, GPU sets, model path, revision, api key path (no secrets inline)
  configs/api_key         API key, chmod 600; never print it
  patches/                any patch applied to vLLM, as .patch files + README of why
  scripts/env.sh          exports all cache/tmp vars; sourced by every script
  scripts/serve.sh        start one server from serve.env (tmux)
  scripts/stop.sh         graceful stop (SIGINT, wait, then SIGTERM)
  scripts/status.sh       nvidia-smi + curl /health + /v1/models
  scripts/bench.py        concurrency benchmark (client side)
  results/                benchmark + validation outputs (json/md)
  README.md               how to start/stop/use; updated at the end
```

`scripts/env.sh` must export at least: `HF_HOME`, `HF_HUB_CACHE`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TMPDIR`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1` (after the model is copied).

## Milestones

### M0 — Preflight (read-only, ~10 min)
1. `ssh rivertam`: `nvidia-smi`, `which tmux`, `ss -ltn | grep -E ':800[0-9]'` (pick free ports, default 8000 for the endpoint).
2. Read `refs/main` in the model cache to find the newest revision. Diff the 3 snapshots' `config.json` (and the `.safetensors.index.json` blob hashes) to see if they differ. Ask the user which revision Inara used if they differ in weights; otherwise use `refs/main`.
3. Find the **vLLM release that supports Inkling2Loop**: search PyPI release notes / vLLM GitHub for `Inkling2Loop` / `inkling` reasoning parser (the user's reference command uses `--reasoning-parser inkling --tool-call-parser inkling`). Also check the HF model card's "vLLM" section and any `vllm` version pin it gives. Record the version and whether it is a release or nightly.
4. Report: chosen revision, vLLM version, free ports. Proceed without waiting unless something blocks.

### M1 — Environment + model copy
1. `source scripts/env.sh` first (so pip cache/tmp go to host_models).
2. `conda create -p /coc/scratch/vchopra/host_models/envs/vllm python=3.12 -y`, then `pip install "vllm==<version>"` (+ `openai`, `requests`). Use the CUDA 12.x wheel the release ships; driver supports 12.9.
3. Copy the pinned snapshot with symlinks resolved: `cp -rL <snapshot> models/TML-d24-varloop-research/<revision>/`. Verify size (~18 GB) and `sha256` of the safetensors file matches the blob name.
4. Generate the API key once: `python -c "import secrets;print(secrets.token_urlsafe(32))" > configs/api_key && chmod 600 configs/api_key`. Tell the user the file path, not the value; they copy it to the laptop themselves (`scp rivertam:<path> ~/.rivertam_api_key`).
5. Write `results/env_manifest.json`: python, torch, vllm, transformers, CUDA versions, model revision, sha256.
6. Check: `python -c "import vllm, torch; print(vllm.__version__, torch.cuda.get_device_capability(0))"` on Rivertam.

### M2 — First load, one copy, GPUs 0–3 (the hard part)
Start with the user's reference flags adapted for Turing:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model_dir> \
  --served-model-name inkling-loop-3 --trust-remote-code \
  --host 0.0.0.0 --port 8001 --api-key "$(cat configs/api_key)" \
  --tensor-parallel-size 4 --dtype float16 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 --enable-prefix-caching \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
  --hf-overrides '{"text_config": {"loop_num_passes": 3, "loop_share_cache": false}}'
```
Expected failures and the fix to try (one at a time, log each in `logs/` and in `patches/README.md`):
- **"min capability 80" / quantization not supported** → patch `ModelOptNvFp4Config.get_min_capability` (and the MXFP8 config if it has the same gate) to return 75 so Marlin FP4/FP8 kernels are used. Save as `patches/modelopt_sm75.patch` + `scripts/apply_patches.sh` (idempotent). Do not edit any other env.
- **BF16 not supported on 7.5** → `--dtype float16` (already set). Watch for NaN/inf in outputs.
- **Attention backend** (FlashAttention needs ≥8.0) → let vLLM auto-pick; if it fails, try `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, then `FLEX_ATTENTION`. Sliding-window (local) layers must be supported by the chosen backend.
- **OOM** → lower `--max-model-len` (4096), `--max-num-seqs` (16), `--gpu-memory-utilization 0.92`; last resort `--enforce-eager`.
- **Vision encoder issues** → text-only is acceptable for this session (`--limit-mm-per-prompt '{"image":0}'`); record it as a limitation.

If after the patch + 2 attempts per issue it still does not load or gives garbage, **stop and report** with the exact errors.
Fallback options to present to the user (do not start them unasked): (a) dequantize to an FP16 checkpoint (~53 GB, all 8 GPUs, one copy); (b) serve with HF Transformers; (c) run varloop baselines on Inara A40s.

### M3 — Correctness gate (must pass before M4)
Run on the single copy from M2. Save everything to `results/correctness_<date>.json`.
1. **Sanity**: greedy (`temperature=0`) answers to 5 fixed prompts (math, factual, short reasoning) are coherent; no NaN; `finish_reason` sane.
2. **Loop count actually applied**: start a second short-lived server with `loop_num_passes: 1` (GPUs 4–7, port 8002). Same 5 prompts, `temperature=0`, `logprobs=true, top_logprobs=5`, `max_tokens=16`. First-token logprobs **must differ** between R=1 and R=3. If identical → the override is ignored; stop and report.
   Also inspect how the vLLM Inkling model reads `loop_num_passes` (grep the model file in the installed vLLM) and record: read once at init, or per forward? Is `loop_share_cache` honoured? This answers the "can R change without reloading" question for vLLM — write the answer in `README.md`.
3. **Matches the reference (best effort)**: on Inara, the HF path (`adaloop`, see `docs/IMPLEMENTATION.md`) is the reference. If cheap, produce top-5 first-token logprobs for the same 5 prompts at R=3 on Inara and compare (expect small FP16 vs BF16 differences; flag any top-1 mismatch). If not feasible this session, record it as open work — do not claim equivalence.
4. Stop the R=1 server after the gate.

### M4 — All 8 GPUs, multiple copies
Topology says: **one copy per PCIe island, TP=4 each → 2 copies**. Never span a copy across islands.
1. Preferred: one process, vLLM data parallel: `--tensor-parallel-size 4 --data-parallel-size 2`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, one port (8000), one model name `inkling-loop-3`. vLLM load-balances across the 2 copies internally. Verify with `nvidia-smi` that GPUs 0–3 and 4–7 each hold one copy (check that DP rank → GPU mapping keeps each copy inside one island).
2. If DP mode fails on this version/hardware: two separate `vllm serve` processes (GPUs 0–3 → port 8001, GPUs 4–7 → port 8002, same config), and front them with a tiny round-robin proxy is **not** in scope — instead report and use LiteLLM proxy only if the user approves.
3. Optional measurement (only if time remains): 4 copies at TP=2 (2 GPUs each, ~9 GB weights/GPU) with `--max-model-len 4096`. Only keep it if it loads and beats 2×TP4 throughput in M5.

Both copies always run the **same** config. Different loop counts are different model names and are never load-balanced together.

### M5 — Benchmark (infrastructure benchmark, not model quality)
`scripts/bench.py` (client side, runs on Rivertam against localhost first, then from the laptop):
- Fixed prompt set (e.g. 64 short prompts, `max_tokens=256`, `temperature=0`, `seed=0`), concurrency 1, 8, 32, 64.
- Record per run: requests/s, output tokens/s, mean/p50/p95 TTFT (streaming), mean inter-token latency, end-to-end latency, errors.
- Compare: 1 copy (TP4) vs 2 copies (DP2×TP4) [vs 4×TP2 if tried].
- Save `results/bench_<layout>_<date>.json` and a short table in `results/BENCHMARKS.md` with the env manifest (vLLM version, revision, dtype, R, GPUs, flags).

### M6 — Laptop access + ops scripts + README
1. Laptop access via SSH tunnel (works off-campus): `ssh -N -L 8000:localhost:8000 rivertam`. Then run the Goal snippet on the laptop. Also test `curl http://localhost:8000/health` and `/v1/models`. Note whether direct `http://rivertam.cc.gatech.edu:8000` works from Inara (`ssh inara 'curl -s -o /dev/null -w "%{http_code}" http://rivertam.cc.gatech.edu:8000/health'`) — record it, do not open firewall rules.
2. Finish `serve.sh` / `stop.sh` / `status.sh` (all source `env.sh`, read `configs/serve.env`, log to `logs/<name>_<timestamp>.log`, run in tmux session `vllm-<name>`). Stop must be graceful.
3. Test recovery once: kill the server (SIGTERM), confirm GPU memory is freed, restart with `serve.sh`, confirm health.
4. Write `host_models/README.md`: what runs where, start/stop/status commands, laptop tunnel + Python client example (API key read from a file/env var, never inline), served model names and their exact config, known limitations (patches applied, dtype, max context, vision on/off, R switching requires a restart unless M3 found otherwise), benchmark summary, troubleshooting (OOM, port in use, stale tmux, NaNs).
5. Leave the 2-copy server **running** at the end and tell the user the endpoint, model name, and where the API key file is.

## Final report to the user

Short: what works, endpoint + model name, measured numbers (one table), correctness gate result (R=1 vs R=3 differ; reference match status), patches applied, limitations, and next steps (e.g. R-switching via sleep mode or per-R model names, LiteLLM gateway for multiple models, Inara access, PACE workers later).
