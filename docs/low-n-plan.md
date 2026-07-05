# Low-N plan: where NLI-hypothesis features actually win

## Thesis

The value of NLI-hypothesis features is **transfer knowledge substituting for labeled data**.
That value is largest when labels are scarce and vanishes when they are abundant. So the headline
experiment is not "our accuracy" — it is a **learning curve**: how the method, and each design
choice inside it, behaves as we go from data-rich to data-starved (2–5 examples/class).

Every null result so far (GEPA instruction tuning neutral, answer-oriented style +0.014 n.s., tree
grouping neutral, flash ≈ pro proposer) was measured **data-rich** (~2k TREC train). In that regime
the head compensates for everything: give RF/HGB enough labeled rows and it rediscovers the class
boundaries almost regardless of hypothesis quality. Those results are not wrong; they are null in
the one regime where they were guaranteed to be null. Low-N is where hypothesis quality has room to
matter — or to fail, which is a real finding either way.

## Governing principle

> **At low-N, a data-independent prior beats a data-dependent estimate. At high-N, the reverse.**

Anything computed *from the training rows* (empirical correlation, permutation importance, a fitted
tree head) degrades toward noise as N shrinks. Anything that injects knowledge *without* consuming
rows (hypothesis text, class-relationship structure, the encoder's transfer) is stable at any N and
therefore relatively more valuable when starved. Most of our design choices sort cleanly onto this
axis, and the principle predicts which ones flip.

## Component predictions (each is a crossover to test, not an assumption)

| Choice | Data-rich (measured) | Low-N prediction | Mechanism |
|---|---|---|---|
| **Dedup: STS vs covariance** | cov preferred (catches behavioral redundancy) | **STS wins** | cov estimates correlation from score-vectors; on 2–5/class that's noise. STS reads hypothesis text — zero rows needed, stable at any N. cov's edge only exists once behavior can be measured. |
| **Tree/grouping instructions** | neutral | **tree helps** | data-rich head rediscovers class-similarity from labels (one-vs-rest gives groups for free); starved head can't, so group-vs-group features inject a prior over class geometry. |
| **Instruction tuning (GEPA)** | neutral | may help | if better transfer wording ever helps, it's here — no head to paper over it. |
| **Answer-oriented style** | +0.014 (n.s.) | may grow | same reason. |
| **Proposer flash → pro** | neutral | may help | better transfer knowledge, uncompensated. |
| **More hypotheses / less pruning** | risks crowding (128 run) | **upside** | with prior-aggregation head, extra transfer knowledge costs no degrees of freedom. |

## The low-N method is a different pipeline

The data-rich pipeline (generate → CV-evolve/prune → RF/HGB head) **cannot** just be run on fewer
rows: evolution's permutation-importance pruning needs folds with several examples/class and returns
noise below that. The low-N pipeline:

1. **Evolution OFF.** Generate a fixed pool from the LLM prior; no permutation-importance pruning.
2. **Dedup = STS** (data-free) at low-N; covariance only re-enters once N is large enough to trust.
3. **Head = prior-aggregation or high-bias.** Tag each hypothesis with the class it was written to
   support (tree/list generation already carries this) → class score = mean entailment of its
   hypotheses → argmax. At **N=0 this is exactly a zero-shot NLI ensemble** (known to beat
   single-template zero-shot). As N grows, allow light reweighting (strong-L2 multinomial logreg).
4. **Data-rich tail** keeps the current RF/HGB + CV evolution; the curves should show it overtaking
   the prior head as N grows.

**Elegant unifying version (build only if the low-N gap is real):** a single **shrinkage head** that
interpolates between "each hypothesis votes for its intended class" (N=0 prior) and "data reweights
the votes" (N-rich), shrinkage scheduled by N — one continuous curve instead of swapping heads.

## Experiment

**Deliverable:** learning curves on a fixed test set.
- **x-axis:** examples/class ∈ {1, 2, 3, 5, 10, 20, 50, 100, all}
- **y-axis:** test accuracy (full, fixed test set — never subsampled)
- **baselines:** zero-shot NLI ensemble (our N=0 point), TF-IDF+logreg, optionally a fine-tuned
  encoder as the strong data-rich baseline
- **method lines:** the crossover variants above (STS vs cov dedup; tree vs flat pool; tuned vs not)
- **story:** where is each crossover? Claim: we own the 1–10/class band; baselines catch up as N
  grows. The finding is *the optimal configuration is a function of N.*

**Discipline:** at 2–5/class variance is large — 30–50 resampled training draws per point with CIs;
test set fixed; never touch test for selection.

**Cost:** each *pool variant* (tree/flat × tuned/not) = one generation + one-time encoder scoring on
train+test. After that the entire N-sweep with all seeds is **free CPU on cached features** (only a
cheap head refits per subsample). So: a few pool builds, then minutes.

**Order:**
1. Low-N harness (subsample sweep, resample seeds, fixed test, prior-aggregation + strong-L2 heads,
   zero-shot ensemble anchor, CIs) on **TREC** — we know it cold and features are cached.
2. Baselines on the same sweep (TF-IDF+logreg, zero-shot ensemble).
3. First crossover: **STS vs covariance dedup** — cheapest direct test of the governing principle.
4. Second crossover: **tree vs flat pool**.
5. Confirm the shape on **one more dataset** (AG News or 20 Newsgroups) before claiming generality.

## Open decisions

- **Datasets for generality:** TREC first to find the crossover cheaply; then AG News and/or 20ng.
  20ng is the most interesting stretch (long texts, truncation, 20 classes) but scoring cost is
  higher.
- **N=0 anchor mechanics:** does the prior-aggregation head require class-tagged hypotheses from the
  proposer (tree/list already implies the tag) or a separate per-class template pass?
- **Where cov overtakes STS on N** — measure it, don't guess the threshold.
- **Whether to build the shrinkage head** — decide after seeing whether the low-N gap is real.

## Status / notes

- `trec_tree128_m` (data-rich, pool 128) still evolving as of this writing — a *ceiling* datapoint,
  not part of the low-N story; pool collapsing 128 → 86 → … toward the natural ~50s size.
- This plan supersedes the data-rich framing for prioritization: the reward/GEPA/style/tree work is
  now justified as *inputs to the low-N crossover study*, not as data-rich accuracy levers.
