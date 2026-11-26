# Evaluation: answer extraction and scoring

For the paper: "extraction (last ANSWER: line) + standard metric; GQA official exact match (LLaVA normalization);
TextVQA VQA accuracy (EvalAI); format failures scored 0 and reported separately."

- Extraction (`core/eval/extract.py`): thinking is dropped; the answer is the rest of the last line starting with
  `ANSWER:` (any case). No such line, or a line cut by the token limit: no answer, score 0. No LLM judge.
- GQA, `correct` (headline): `prediction.strip().rstrip('.').lower() == answer`, the gold answer as given.
- GQA, `correct_lmms`: the lmms-eval rule (ignores case and ASCII punctuation on both sides).
- TextVQA: VQA accuracy over the 10 human answers after the EvalAI normalization; `correct_lmms` equals `correct`.
- Rows keep `raw_output`, `pred`, `format_ok` (a marker line exists), `parse_ok` (an answer was read),
  `answer_len_ok` (answer of at most `answer_max_chars`; answers are never truncated).
- Thinking budget: `forced` rows had the answer message opened for them after `think_budget` tokens, so their
  `format_ok` is True by construction. Report the forced rate next to the format rate.

## Inference protocol vs common practice (checked Nov 26, 2026)
- Ours: greedy (`do_sample=False`), BF16, batch 1, 2-GPU model split per copy, parallel copies for speed.
- lmms-eval GQA task: temperature 0, do_sample false, num_beams 1, max_new_tokens 16; README examples use `--batch_size 1`.
- LLaVA `model_vqa_loader.py`: DataLoader batch_size 1 (asserted), textvqa.sh temperature 0, num_beams 1, fp16.
- InternVL `eval/vqa/evaluate_vqa.py`: temperature 0, num_beams 1, batch 1 asserted ("Only batch size 1 is supported"), bf16.
- VLMEvalKit: one sample per call; greedy or near-greedy (Qwen2-VL wrapper: temp 0.01, top_k 1).
- Batch nondeterminism: Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025): kernels are run-to-run deterministic but not batch-invariant; batch-invariant kernels remove it (vLLM `VLLM_BATCH_INVARIANT=1`, at a throughput cost). Our measurement: 13/128 batched short answers differ from batch 1 (nb07, Nov 26).
- Paper statement: headline numbers are greedy, batch 1, BF16 (as LLaVA / InternVL / VLMEvalKit); batching changed 13/128 outputs (cite Thinking Machines for the cause); batch-1 latency is per-sample latency, not throughput; the 2-GPU split is a hardware caveat.
- Not verified: pruning papers' latency protocol (FastV, SparseVLM, DivPrune, CDPruner, VisionZip READMEs give no batch size); check their appendices before comparing latency numbers.
