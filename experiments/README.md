# Paper experiments — reproduction & handoff

This directory holds the experimental foundation for the Hypothesis Vectorization paper. The write-up
lives in [`../paper/`](../paper/): `research_report.md` (all result tables, shareable), `draft.md`
(full prose draft), `method.md`, `related_work.md`, and `experiment_notes.md` (the chronological run
log with every finding and caveat).

## Layout

```
experiments/
  hvexp/                 importable helpers (reused by every script)
    datasets.py          full-train-pool + fixed-test loader (the learning-curve protocol)
    features.py          NLIFeaturizer — cache-backed NLI scoring (+ in-process memo)
    hypotheses.py        hand-written class-tagged expert pools + zero-shot templates per dataset
    systems.py           pluggable systems: TF-IDF/embeddings/zero-shot/HV heads/prior head/fine-tune
    metrics.py           accuracy, macro/weighted-F1, per-class P/R/F1, ECE, bootstrap CIs
    repro.py             manifest.json (git+libs+dataset+seed) + split persistence
  scripts/
    run_learning_curve.py   k×seeds×systems on a fixed test set  (RQ1/RQ3)
    generate_pool.py        LLM pool generation (static or evolved) → JSON      (RQ4; needs LM key)
    run_ablation.py         score-mode / pool-size / head / encoder axes         (RQ4)
    run_text_tabular.py     CFPB-style 6-config marginal-value study              (RQ5)
    prep_cfpb.py            stream CFPB → CSV + generate its pool (train-only)    (RQ5; needs LM key)
    inspect_hypotheses.py   interpretability: importances, redundancy, exemplars (RQ2)
    summarize_results.py    results.jsonl → mean±CI markdown/LaTeX/CSV tables
    make_figures.py         learning-curve figures with CI bands (PDF+PNG)
  configs/runs/          example run config (most runs were driven by CLI flags)
  results/
    raw/<run_id>/        results.jsonl (one row per system×shots×seed) + manifest.json  [committed]
    tables/ processed/   generated markdown/LaTeX/CSV tables                              [committed]
    figures/             PDF + PNG                                                        [committed]
    pools/  cfpb/*.json  generated hypothesis pools                                       [committed]
    cfpb/*.csv           CFPB raw data — GITIGNORED, regenerate with prep_cfpb.py
    logs/                run logs — gitignored
```

## ⚠️ Read this first: the NLI cache and the filesystem

The NLI score cache is a SQLite DB. **This machine's workspace is a 9P-mounted ZFS share where
SQLite's per-lookup RPCs make a large scoring pass crawl (GPU sits idle).** Point the cache at a
**local** filesystem for every GPU run:

```bash
export HV_CACHE_DIR=/tmp/hv_cache      # local ext disk; ~500 MB after all runs
```

Two related facts: (1) `ScoreCache` auto-falls back from WAL to DELETE journaling on filesystems that
reject WAL (also ZFS) — see `src/hypothesis_vectorizer/cache.py`. (2) The cache is content-addressed
by (text, hypothesis, encoder), so **runs resume for free** — a killed scoring pass re-uses every
pair already scored (we paused/resumed the 1.9M-pair CFPB temporal pass this way).

## Setup

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env      # only for pool GENERATION (RQ4/RQ5); inference needs none
```

## Reproduce the headline results

```bash
# RQ1 — TREC low-label learning curve (baselines + expert HV + prior head), 10 seeds
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/run_learning_curve.py \
  --config experiments/configs/runs/trec_lown_baselines.yaml
uv run python experiments/scripts/summarize_results.py experiments/results/raw/lc_trec_baselines_l --name trec_lown
uv run python experiments/scripts/make_figures.py experiments/results/raw/lc_trec_baselines_l --out trec_learning_curve

# RQ2 — interpretability report for a fitted pool
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/inspect_hypotheses.py --dataset trec

# RQ4 — generate LLM pools, then add them to the curve
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/generate_pool.py --dataset trec --n 64 \
  --out experiments/results/pools/trec_gen_static.json
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/run_learning_curve.py --dataset trec \
  --generated-pool static=experiments/results/pools/trec_gen_static.json --run-id lc_trec_generated_l
# pool-size ablation (subsamples a generated pool 8..256)
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/run_ablation.py --dataset banking77 \
  --axis pool_size --pool-json experiments/results/pools/banking77_gen256.json --pool-sizes 8 16 32 64 128 256

# RQ5 — CFPB text+tabular (balanced controlled setting)
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/prep_cfpb.py --per-class 2000
HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/run_text_tabular.py \
  --csv experiments/results/cfpb/cfpb.csv --text-col narrative \
  --cat-cols Product Company State "Submitted via" --label-col relief \
  --split random --max-text-chars 512 --pool-json experiments/results/cfpb/cfpb_pool_random.json
```

Other datasets: swap `--dataset {trec,ag_news,sst2,banking77,goemotions}`. The CFPB temporal
(benchmark) run uses `prep_cfpb.py --limit 30000 --split temporal` then `run_text_tabular.py --split
temporal --date-col date`.

## Cost / time (RTX 5090, 450 W cap)

- A learning curve = **one** NLI scoring pass over the corpus (cached), then the whole seed×shot
  sweep is cheap CPU. Short-text datasets (TREC/SST-2) warm in minutes; long-text CFPB narratives
  (512 chars) score at ~300 pairs/s, so a 30k×64 pass is ~1.5 h — run it once, it's cached after.
- Pool generation is ~$0.01–0.07 of LM per pool (DeepSeek-v4-flash); inference is LM-free.
- The RF/HGB CV-grid head is slow on many-class data; the curves use a single RF for the flexible-head
  line (`head="rf"`), with the exact library CV grid available as `head="auto_full"` for a headline
  point.

## Status & handoff notes for the next person

**Done (all committed on `paper-experiments`):** RQ1–RQ5 across five datasets (TREC, SST-2, AG News,
GoEmotions, Banking77) + CFPB text+tabular in both balanced and temporal settings; all baselines incl.
a fine-tuned DistilBERT; interpretability; generation/evolution/pool-size/encoder/score-mode
ablations. See `paper/experiment_notes.md` for the full log and `paper/research_report.md` for tables.

**Known limitations / good next steps:**
- **CFPB reproducibility:** `prep_cfpb.py` pins the HF dataset revision
  (`088cc73…`, 2025-12-29), so a regenerated CSV is order-stable; the committed *pools* and
  *results.jsonl* remain the durable artifacts. Bump `_CFPB_REVISION` deliberately for newer data.
- **Single test split per dataset.** Learning curves resample the *train* draw over seeds but use one
  fixed test set; a few CFPB/ablation comparisons are single-split (deltas noted as "within noise").
- **Interpretability + significance** (`inspect_hypotheses.py`, the `hypothesis-vectorizer compare`
  McNemar tool) are wired but only run on TREC so far — extend to the other datasets for the paper.
- **Not run:** multilingual; larger/alternative NLI encoders and sentence-embedding models (bge/e5);
  a global-frontier evolution variant.
- **Negative results to preserve, not bury:** evolution saturates in ~2 rounds and its
  marginal-over-baseline pruning did not help CFPB on test; the score channel is within noise; HV
  loses to embeddings/TF-IDF in data-rich or thin-pool regimes. These are load-bearing for the
  honest Pareto framing.
