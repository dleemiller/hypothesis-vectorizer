# Method: Hypothesis Vectorization

## Overview

Hypothesis Vectorization represents a text by its natural-language *entailment profile*: how
strongly the text entails (and contradicts) each member of a compact pool of human-readable
hypotheses. An LLM proposes the hypotheses, a frozen NLI cross-encoder measures them, and a
classical model estimates the task over the resulting interpretable feature matrix. No neural
weights are fine-tuned; task adaptation lives entirely in the hypothesis text and a light head.

The method factorizes cleanly into three roles, and this abstraction is used throughout the paper:

| role | component | what it does |
|---|---|---|
| **proposer** | LLM (e.g. DeepSeek-v4-flash) | proposes a semantic *basis* — a pool of hypotheses |
| **measurement operator** | frozen NLI cross-encoder | scores each (text, hypothesis) pair; the fixed instrument |
| **estimator** | classical head (sklearn) | fits the task over the measured features |
| **search** | prune/evolve loop | searches over *semantic features*, not neural weights |

## Formalization

Let a labeled dataset be $D = \{(x_i, y_i)\}_{i=1}^n$ with $y_i \in \{1, \dots, K\}$, and a
hypothesis pool $H = \{h_1, \dots, h_m\}$ of short natural-language statements about the input.
A frozen NLI model $f_\theta$ maps a (premise, hypothesis) pair to a distribution over
{entail, neutral, contradict}:

$$f_\theta(x, h) = \big(p_e(x,h),\; p_n(x,h),\; p_c(x,h)\big).$$

The feature map scores $x$ against the whole pool:

$$\phi(x) = \big[\,g(f_\theta(x, h_1)),\; \dots,\; g(f_\theta(x, h_m))\,\big],$$

where $g$ selects the **score mode** (columns per hypothesis):

- **entail** — $g = p_e$ (1 column): "does the text support this statement?"
- **entail+contradict** — $g = (p_e, p_c)$ (2 columns): support *and* refutation, the default.
- **contrast** — $g = p_e - p_c$ (1 column): a signed axis.

A classical estimator $g_\psi$ (logistic regression, random forest, gradient boosting, or a
prior-aggregation head) predicts $\hat{y} = g_\psi(\phi(x))$. Because $\phi$ produces one named,
readable dimension per hypothesis, $g_\psi$'s feature importances are directly interpretable:
each weight attaches to an English sentence.

At inference there is **no LLM**: the hypothesis list is fixed, the NLI model is frozen, and the
head is a serialized sklearn object. The per-example cost is exactly $m$ NLI forward passes,
which are cacheable and independent across examples.

## Generation

The proposer is prompted with the task description, class names + one-line *definitions* (not bare
labels), and a small stratified sample of **training** examples only. It returns hypotheses as
structured JSON, `{text, intended_class, rationale}`, where only `text` becomes a feature and the
rest is metadata for analysis and for the prior-aggregation head. Prompt rules enforce that
hypotheses be short, atomic, affirmative statements *about the text*, verifiable from the text
alone, class-relevant, semantically diverse, and not bare label names or dataset-leaking strings.

Hand-written **fixed hypotheses** can be supplied by a domain expert; they are always scored and
fit alongside the generated pool and never pruned, letting expert knowledge anchor the basis while
the LLM fills in around it.

## Deduplication

Candidate hypotheses are deduplicated to avoid redundant feature dimensions:

- **covariance** — reject a candidate whose entail-score vector correlates above a threshold with
  a kept one. This removes *behavioral* duplicates (different wording, same measurement) but needs
  enough data to estimate the correlation, so it is unreliable at very low $N$.
- **STS** — reject a candidate whose *text* is too cosine-similar to a kept hypothesis under a
  sentence encoder. This is data-free and therefore the correct default at 1–5 examples/class.

A variance floor additionally rejects near-constant (vacuous) hypotheses whose scores carry no
information on the corpus.

## Evolution (optional)

For data-rich settings, an evolutionary loop refines the pool:

1. **score** the pool on a stratified train subsample (cache-through);
2. **rank** hypotheses by cross-fold permutation importance *and* cross-fold sign stability —
   ranking by a single split churned ~50% of prune decisions across seeds, so stability is the
   discipline;
3. **prune** only *confident deaths* (importance ≤ 0 in every fold), at most half the pool;
4. **refill** against confusion hot-spots (connected components of the thresholded confusion
   graph), giving the proposer a batch of errors per mutually-confused class group with the
   failure reason for each pruned hypothesis;
5. **stop** on a held-out plateau (patience).

With a baseline feature block configured (TF-IDF or tabular columns), ranking is *marginal over
that block*, so hypotheses whose signal a free channel already carries are dropped. Measured
finding: generation **saturates in ~2 rounds** — the encoder's class-relevant directions are
exhausted quickly, and further rounds/pool-size add little (an important negative result the paper
reports rather than hides).

## Head selection

The head family and its regularization are selected by cross-validation **on the training set
only**, then refit on full train, and the model is evaluated on the test set **exactly once**
(the `pool_cv` protocol). Selecting the best of several heads by their *test* scores inflated
accuracy by ~2 points in early experiments; the honest protocol removes that illusion. Two-run
comparisons on a shared test set use paired **McNemar** with Wilson confidence intervals, so a
single-run A/B delta is only claimed when it clears the noise floor.

## The low-N head

At 1–5 examples/class, a flexible RF/HGB head overfits (very high CV-train, poor test) and is
also compute-heavy. The paper introduces a **prior-aggregation head** that exploits the
`intended_class` tags: the score for class $k$ is the mean entailment of the hypotheses written to
support $k$, and the prediction is the arg-max. At $N=0$ this is exactly a *zero-shot NLI
ensemble* over multiple hypotheses per class — strictly richer than single-template zero-shot. As
$N$ grows, a strongly-regularized multinomial logistic regression reweights the class votes
(`prior_reweight`), and the flexible RF/HGB head eventually overtakes both. The governing
principle (see `docs/low-n-plan.md`): *at low $N$ a data-independent prior beats a data-dependent
estimate; at high $N$ the reverse* — so the optimal configuration is a function of $N$.

## Reproducibility & cost

Every (text, hypothesis, encoder) score is cached (SQLite, raw logits), so reruns and post-hoc
analyses are near-free and the entire seed/train-size sweep costs one GPU scoring pass. Each run
records its git commit, library versions, dataset revision, seed, split indices, config, pool,
metrics, predictions, and LM/encoder cost accounting in a `manifest.json`. Inference issues zero
LLM calls by construction.
