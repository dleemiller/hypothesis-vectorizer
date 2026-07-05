# nli-boost

Text classification from **LM-written NLI hypotheses**: a frozen NLI cross-encoder
([finecat](https://huggingface.co/dleemiller/finecat-nli-m)) scores whether each text entails
each of ~64 English sentences written by an LLM; those scores are features for a CV-disciplined
classical head. No fine-tuning anywhere — task adaptation lives in the sentences.

**See [METHOD.md](METHOD.md)** for the full process and the measurement behind every design
choice. TREC-6 with 2k training examples, current recipe (seed 7): **0.934** test accuracy at `-m`
and **0.954** at `-l`, ~7 minutes and under $0.01 per fit. (The original instruction spanned
0.916–0.938 across seeds at `-m`; the answer-oriented instruction below is what raised the `-m`
number.)

## Current best recipe

The configuration that wins on TREC today (`configs/trec_best_l.yaml`):

- **Encoder `finecat-nli-l`** — the one lever that reliably moves accuracy (`-m`→`-l` ≈ +5 pts,
  p=0.024). Everything else below is within noise at `-l`; the encoder is where the accuracy is.
- **Hand-written answer-oriented instruction** (the code default) — each hypothesis describes both
  the question and the *answer form* it implies (e.g. *"equivalent to asking someone to name a
  person"* / *"can be answered with a short proper name"*). GEPA-tuning this instruction is neutral
  (McNemar p≈1.0), so it stays hand-written.
- **Covariance dedup** — reject a candidate whose entail-score vector correlates >0.95 with a kept
  hypothesis (removes *behavioral* duplicates that text-similarity dedup misses).
- **Pool of 64, evolved** — generate → rank by CV permutation-importance + cross-fold stability →
  prune confident deaths, refill against confusion hot-spots → repeat to a held-out plateau.
- **CV-selected classical head** (RF / HistGBM) over the entail+contradict features.

Result (seed 7): **0.934** at `-m`, **0.954** at `-l`. The answer-oriented instruction is what
lifts `-m` — the original instruction scored 0.920 at the same seed/dedup (`trec`), the
answer-oriented one 0.934 (`trec_newinstr`, +0.014). At `-l` that gain washes into the ~0.95
saturation band (`baseline_l` 0.952), so the instruction is only measured to help at `-m`. The new
instruction is currently validated at seed 7 only. ~7 min and <$0.01 per fit, and the model is a
human-readable list of ~64 English sentences.

**Add the lexical channel when inference cost matters** (`configs/trec_best_l_max.yaml`:
`lexical: {kind: tfidf_svd, dims: 128}`). TF-IDF is ~free at prediction time, while every NLI
hypothesis is a cross-encoder forward pass. So the lexical block joins evolution as a **fixed
baseline** and NLI hypotheses are pruned by their **marginal value over TF-IDF** — a hypothesis
whose signal TF-IDF already carries dies. The NLI pool (and thus per-prediction cost) shrinks to
only the hypotheses carrying semantics lexical can't reach, in the same accuracy band. Best point
estimate to date: **0.964** at `-l` (seed 7), though not yet significantly above the plain-TF-IDF
run (0.956, McNemar p=0.42) — treat as promising, not established.

> This recipe targets the **data-rich** regime. The method's expected edge is at **low-N**
> (2–5 examples/class), where a different pipeline applies (evolution off, prior-selected
> hypotheses, STS dedup, light head) — see [docs/low-n-plan.md](docs/low-n-plan.md).

## Setup

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # the hypothesis proposer LM
uv run pre-commit install
```

## Usage

```bash
uv run nli-boost run configs/trec.yaml            # full method: generate -> evolve -> head -> test
uv run nli-boost run configs/trec_finalize_l.yaml # reuse a fitted pool, re-score with -l encoder
uv run nli-boost report                           # pool_cv results across runs
uv run nli-boost diagnose runs/trec               # error decomposition + reward-hacking flags
uv run nli-boost compare runs/a runs/b            # paired McNemar: is a delta real or noise?
uv run nli-boost gepa-tune --fresh                # (optional) GEPA-tune the proposer instruction
```

### Optional: instruction tuning (GEPA)

`gepa-tune` optimizes the proposer's `GeneratePool` instruction offline (dspy GEPA, `auto` budget)
against a composite, **grounded-boolean** reward — noise-averaged held-out CV accuracy + a semantic
judge (yes/no criteria, no ungrounded float scores) − an artifact penalty, aggregated across
datasets by geometric mean. Tune on several domains and hold one out; load the result in any config
via `lm: {instruction_path: models/proposer_instruction.json}`.

Measured verdict: the tuned instruction **matched** the hand-written one at `-l` (0.946, McNemar
p=1.0) — the hand-written prompt is already near the ceiling and the **encoder is the real lever**
— so this is a reusable research loop, not a default. See `NOTES.md` for the full audit.

Artifacts per run in `runs/<run_name>/`: the pool itself (`model.json` — the model is a list of
English sentences), the evolution audit trail (`log.jsonl`: every prune with its reason, every
refill with its target-AUC), `metrics.json` (the single honest headline), and `costs.json`.
All NLI scores are cached in `cache/nli_scores.sqlite`; reruns and post-hoc analyses are ~free.

## Development

```bash
uv run pytest          # full pipeline runs under fakes — no GPU or LM key needed
uv run ruff check .    # also enforced via pre-commit
```

The pre-rewrite exploratory code (trees, boosting, and the experiments that selected this method)
is archived untracked in `src-bak/`; the experiment log lives in `NOTES.md`. (The live instruction
tuner is `src/nli_boost/gepa_tune.py` + `reward.py`, distinct from the archived tree-era GEPA.)
