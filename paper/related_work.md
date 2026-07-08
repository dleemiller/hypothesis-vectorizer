# Related Work

Hypothesis Vectorization sits at the intersection of zero-shot NLI classification, natural-language
supervision, interpretable/concept-based models, and low-resource text classification. We position
it against each, emphasizing what is *distinct*: HV learns a downstream estimator over a **pool of
many readable semantic probes** measured by a **frozen** NLI model, with **no LLM at inference**.

## Zero-shot classification via NLI

Yin et al. (2019) recast text classification as textual entailment: score each candidate label as
a hypothesis ("This text is about {label}.") and take the arg-max entailment. This is the direct
ancestor of HV and is our N=0 baseline. HV differs in three ways: (i) it uses **many** hypotheses
per class rather than one template, forming an ensemble that we show beats single-template
zero-shot; (ii) the hypotheses are **LLM-generated and diverse**, not fixed label templates;
(iii) a **learned head** over the full entailment profile replaces the arg-max, so a little labeled
data reweights and combines probes. Our `zeroshot_nli` and `hv_prior_fixed` systems make this
progression explicit and measurable.

## Natural-language supervision & LLM-generated features

A growing line of work supplies *natural-language descriptions* as supervision or as features:
generating attributes/rationales with an LLM and scoring them (e.g. describe-and-classify,
LLM-generated concept features, "language-model-as-feature-extractor"). HV is an instance of this
family in which the scoring function is specifically a **frozen NLI entailment operator** and the
features are **hypotheses** (declarative statements the text may entail/contradict), which gives
each feature a precise, auditable semantics and lets domain experts inject or veto probes.
Relative to methods that call an LLM per feature per example at inference, HV moves all LLM cost to
a one-time generation step and serves with a cheap frozen encoder.

## Concept Bottleneck Models

Concept Bottleneck Models (Koh et al., 2020) predict human-specified concepts, then predict the
label from the concept layer, for interpretability and intervention. HV is a concept bottleneck
whose (i) concepts are **generated in natural language** rather than hand-annotated, (ii) concept
*scores* come from a frozen NLI model rather than a trained concept predictor, and (iii) concepts
are **searched** (pruned/evolved by marginal utility) rather than fixed. Like CBMs, HV supports
intervention: an operator can edit, add, or remove a hypothesis and immediately see the effect,
because the bottleneck is a readable list.

## Weak supervision & labeling functions

Data-programming frameworks (Snorkel; Ratner et al., 2017) combine noisy labeling functions —
typically code/regex heuristics — into probabilistic labels. HV's hypotheses are related in spirit
(many weak, interpretable signals combined by a model) but differ crucially: the signals are
**semantic NLI probes** evaluated by a neural entailment model, not brittle lexical rules, so they
generalize beyond surface forms; and they feed a **discriminative head with labels** rather than a
generative label model. HV can be read as "labeling functions written in natural language and
graded by an NLI encoder."

## Prompt-based classification & LLM direct classification

Direct zero-/few-shot LLM classification prompts a large model to emit the label. It is a strong
baseline but incurs per-inference LLM cost/latency, is operationally brittle (format drift, prompt
sensitivity), and is hard to audit. HV keeps the LLM's semantic prior (via generated hypotheses)
while removing it from the serving path, trading a fixed frozen-encoder cost for auditability and
determinism. We report LLM classification cost/latency separately rather than as a headline
accuracy contest.

## Prompt / program optimization (DSPy, GEPA)

Automatic prompt and program optimizers (DSPy; GEPA-style reflective evolution, Agrawal et al.,
2025) search over instructions/demonstrations to improve an LLM program. HV borrows the
*optimization discipline* (minibatch-then-full screening, acceptance gates before a mutation
enters the pool, reflective refill, instance/class-level frontiers) but applies it to a different
search space: **which hypotheses form the best semantic basis**, scored by a frozen encoder. Our
own ablations found instruction tuning of the proposer to be neutral for this method — the
generation prompt is already near a diversity ceiling and the **encoder is the capacity lever** —
which we report as a negative result.

## Dense sentence embeddings & linear probes

Sentence-embedding + linear-probe pipelines (Sentence-BERT; E5; BGE) are strong, cheap baselines
and we include them. They differ from HV on the axis the paper foregrounds: embedding dimensions
are **opaque**, whereas HV dimensions are **named English hypotheses**. HV trades some raw capacity
for readability, expert controllability, and a data-independent prior that helps most at low $N$.

## Low-resource text classification

Few-shot text classification spans pattern-exploiting training (PET; Schick & Schütze, 2021),
SetFit (Tunstall et al., 2022), and prompt-based tuning. These typically fine-tune an encoder on
the few labels. HV instead imports prior knowledge *without gradient updates* — through generated
hypotheses and a frozen NLI model — and adapts with a light head, which is what makes its low-$N$
behavior and its no-fine-tuning auditability distinctive. Our learning curves position HV against
these regimes by train-set size rather than claiming a single operating point.

## Text + tabular modeling

In applied settings text co-occurs with structured metadata (the CFPB consumer-complaints case).
Prior work fuses TF-IDF/topic features with categorical fields in a GBM. HV contributes a
**readable text channel** that can be evaluated by its **marginal value over the existing tabular
block** (hypotheses are pruned by marginal utility over a fixed baseline feature matrix), which
makes it practical to justify in a real ML pipeline and to audit for regulatory settings.

---

_References to be finalized with full citations during write-up (Yin et al. 2019; Koh et al. 2020;
Ratner et al. 2017; Reimers & Gurevych 2019; Wang et al. E5 2022; Xiao et al. BGE 2023; Schick &
Schütze 2021; Tunstall et al. 2022; Khattab et al. DSPy 2023; Agrawal et al. GEPA 2025)._
