"""Run configuration. Only the knobs of the converged method (METHOD.md) exist."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field


class DataConfig(BaseModel):
    name: Literal["ag_news", "sst2", "trec", "20newsgroups"]
    train_size: int = 2000
    val_size: int = 500
    test_size: int = 2000
    # few-shot: exactly this many train examples PER CLASS (K-shot). Overrides train_size/val_size
    # when set — the honest low-N setup (proportional sampling starves rare classes).
    shots_per_class: int | None = None


class EncoderConfig(BaseModel):
    """The frozen NLI cross-encoder — the method's capacity knob (-m -> -l = +5 pts)."""

    model: str = "dleemiller/finecat-nli-m"
    batch_size: int = 128
    max_text_chars: int = 1200  # normalize+truncate BEFORE hashing so cache keys are stable
    device: str = "cuda"
    verbose: bool = True  # progress lines on long GPU scoring passes (CLI on; vectorizer off)
    # feature layout per hypothesis: [entail | contradict] (2 cols) or 'full' = [entail |
    # contradict | neutral] (3 cols). Neutral is linearly dependent (e+n+c=1) so it adds nothing
    # to a LINEAR head — but trees split ONE feature at a time, and a threshold on neutral is not
    # expressible as any single threshold on e or c: a real extra axis for DT/RF heads. Raw logits
    # are cached, so the third column is free.
    features: Literal["entail_contradict", "full"] = "entail_contradict"


class DedupConfig(BaseModel):
    """Candidate dedup during generation. `covariance` drops a candidate whose entailment score
    vector is ~collinear with a kept one (behavioral, needs data); `sts` compares hypothesis
    TEXTS with a bi-encoder (data-free — the low-data choice)."""

    kind: Literal["covariance", "sts"] = "covariance"
    # rejection threshold: |Pearson| for covariance; cosine for sts (~0.9 is sensible there).
    # ("corr_threshold" accepted for configs saved by older runs.)
    threshold: float = Field(0.95, validation_alias=AliasChoices("threshold", "corr_threshold"))
    ref_size: int = 400  # covariance only: stratified train subsample the vectors are correlated on
    # covariance only: reject candidates whose entail std on the ref texts is below this — a
    # ~constant feature is dead weight however it is worded. Conservative: even a detector for a
    # 1.6%-prevalence class measures ~0.10 (NOTES 2026-07-05).
    min_std: float = 0.02
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # sts only: the sentence encoder


class LMConfig(BaseModel):
    """The hypothesis proposer. Cheap by design; a full fit costs ~$0.01."""

    model: str = "openrouter/deepseek/deepseek-v4-flash"
    max_tokens: int = 12000  # reasoning + hypotheses can exceed 4k and truncate mid-JSON
    temperature: float = 1.0
    # provider passthrough, e.g. {"provider": {"order": ["deepseek"], "allow_fallbacks": false}}
    # NOTE: part of the LM cache key — changing it invalidates cached proposals.
    extra_body: dict | None = None
    # optional GEPA-tuned GeneratePool instruction (a saved dspy program json); overrides the
    # hand-written GeneratePool docstring when set.
    instruction_path: str | None = None


class TreeConfig(BaseModel):
    """Tree-guided evolve (PoolConfig.method='tree'): fit an entropy CART on the pool's features,
    find the highest-entropy leaf, K-shot its examples to the LLM in a Refine/BestOfN loop to
    propose ONE hypothesis that splits it (reward = info gain), add it, refit, repeat."""

    rounds: int = 24  # LLM-in-loop rounds; one hypothesis added per productive round
    refine_attempts: int = 4  # refine-loop / best-of-n budget per round
    strategy: Literal["refine", "best_of_n"] = "refine"
    leaf_shots: int = 12  # K examples sampled (stratified) from the target leaf for the LLM
    leaf_min_samples: int = 20  # only target leaves with at least this many train examples
    # UNCAPPED depth by default: a depth cap froze the frontier — the exact leaf a new hypothesis
    # was written for sat AT the cap, so the tree could never use it (measured 2026-07-07).
    # Regularization comes from min_impurity_decrease instead: a split must remove this much
    # WEIGHTED entropy (sklearn semantics: N_node/N x decrease). 0.002 froze a mid-size leaf too
    # (a 0.00197 win lost to the gate by 1.5%); 0.001 measured: leaves 98->152 (min_samples_leaf
    # bounds shredding), train_acc 0.906->0.920, and the remaining impure leaves are GENUINE
    # confusion rather than gate-suppressed splits.
    max_depth: int | None = None
    min_impurity_decrease: float = 0.002
    min_samples_leaf: int = 10
    patience: int = 3  # stop after this many consecutive rounds that add no new hypothesis


class PoolConfig(BaseModel):
    size: int = 64  # ~30 useful directions exist per task (measured); 64 gives slack
    # Stage-2 refinement: 'stability' (CV permutation-importance prune/refill) or 'tree'
    # (tree-guided LLM-in-loop split proposal — see TreeConfig). 'tree' grows `size` initial
    # hypotheses by up to `tree.rounds` more; 32 is a sensible initial `size` for it.
    method: Literal["stability", "tree"] = "stability"
    tree: TreeConfig = TreeConfig()
    rounds: int = 6  # stability: hard cap; patience exits ~round 2-3 in practice
    patience: int = 2  # stability: stop when held-out CV accuracy stops improving
    # stability: minimum held-out improvement that resets patience. Measured round noise is
    # ~0.0037 median (NOTES 2026-07-05): 1e-4 lets sub-noise upticks reset patience (over-
    # running); ~0.003 requires patience >= 3-4 or delayed real jumps get cut off.
    plateau_epsilon: float = 1e-4
    min_keep_frac: float = 0.5  # stability: never prune below this fraction in one round
    rank_sample: int = 800  # stability: ranking needs no full-matrix precision
    # hand-written hypotheses that are ALWAYS kept: scored and fit alongside the generated
    # pool, but treated as a fixed baseline in evolution (never pruned; generated hypotheses
    # must add marginal value over them — same mechanism as the TF-IDF channel).
    fixed_hypotheses: list[str] = []
    # reuse the pool from a previous run instead of generating: this is how a
    # pool is finalized with a bigger encoder (hypotheses transfer; only re-score).
    # With method='tree' the reused pool is the STARTING pool and is tree-evolved —
    # ideal for cache-warm experiments (only new proposals hit the GPU).
    from_run: str | None = None
    from_run_top: int | None = None  # truncate the reused pool to its first N hypotheses


class LexicalConfig(BaseModel):
    """Optional static lexical channel concatenated with hypothesis features."""

    kind: Literal["none", "tfidf_svd"] = "none"
    dims: int = 128


class RunConfig(BaseModel):
    run_name: str
    seed: int = 7
    data: DataConfig
    encoder: EncoderConfig = EncoderConfig()
    dedup: DedupConfig = DedupConfig()
    lm: LMConfig = LMConfig()
    pool: PoolConfig = PoolConfig()
    lexical: LexicalConfig = LexicalConfig()
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
