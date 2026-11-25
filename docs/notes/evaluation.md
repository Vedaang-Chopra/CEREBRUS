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
