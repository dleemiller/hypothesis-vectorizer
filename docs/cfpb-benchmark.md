# CFPB Consumer Complaints — a text + tabular benchmark

The CFPB Consumer Complaint Database is an ideal test of `HypothesisVectorizer` composed with tabular
features: a long free-form **narrative** plus structured metadata (product, company, state, channel),
where the two are complementary. Data: `BEE-spoke-data/consumer-finance-complaints` on HF (parquet
mirror of the official DB; the official `CFPB/...` repo is a legacy dataset script and no longer
loads). The stream is recent-first — most recent rows are `In progress` with no narrative — so filter
to **closed complaints that have a narrative**.

## Benchmarks to compare against

- **Monetary-relief prediction (binary)** — *From Complaint Narratives to Monetary Relief*, Wang,
  Zhu & Chen, 2026 ([arXiv:2606.22664](https://arxiv.org/abs/2606.22664)). Predict whether a
  complaint closes with monetary relief, from narrative + LDA topics + engineered text features +
  categorical (company, state); **temporal** train/test split (older→train, newer→test).
  Reported **AUC-ROC 0.78** (their hybrid GBM) vs **0.69** (TF-IDF baseline). This is our primary
  target — it directly exercises the text+tabular fusion and the `baseline_features` pruning.
- **Product classification (multi-class)** — the canonical CFPB text task (predict `Product` from
  the narrative; ~9–11 consolidated classes). Widely reproduced (e.g. *Supervised ML for Text
  Analysis in R*, ch. 7; various LSTM/logreg papers). A text-only sanity comparison.
- **Zero-shot LLM classification (5 classes)** — *DeepSeek and GPT Fall Behind: Claude Leads in
  Zero-Shot Consumer Complaints Classification*, 2025 ([preprints.org 202502.0720](https://www.preprints.org/manuscript/202502.0720)).
  Directly comparable to our zero-shot-NLI baseline story.

## Our setup (monetary relief)

`examples/cfpb.py` reproduces the monetary-relief task:
- filter to closed complaints with a narrative; label `1` iff response == "Closed with monetary relief";
- temporal split by `Date received` (oldest 80% train, newest 20% test);
- tabular block = one-hot(`Product`, top-k `Company`, `State`, `Submitted via`);
- `HypothesisVectorizer` on the narrative, **generated with the tabular block as `baseline_features`**
  so hypotheses are pruned by marginal value over the metadata, then served in a `ColumnTransformer`
  alongside the same one-hot block → `HistGradientBoostingClassifier`;
- report **test AUC-ROC** vs the 0.78 / 0.69 benchmark.

Cost note: narratives are long and there are hundreds of thousands with the label; a full run is
multi-hour GPU + an LM generation pass. `--limit` subsamples for a first pass.
