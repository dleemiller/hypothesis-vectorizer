"""NLI feature computation for the experiment harness, reusing the library's cached scorer.

For a learning curve we score the *entire* train pool + the fixed test set against a
hypothesis pool exactly once (per encoder), persist the raw logits in the shared sqlite
cache, and then every (k-shot, seed) subsample just slices rows out of the cached matrix —
so the whole seed/size sweep is free CPU after one GPU pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hypothesis_vectorizer.cache import ScoreCache
from hypothesis_vectorizer.config import EncoderConfig
from hypothesis_vectorizer.costs import CostTracker
from hypothesis_vectorizer.encoder import EntailmentScorer

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "cache" / "nli_scores.sqlite"


class NLIFeaturizer:
    """Thin wrapper: (texts, pool) -> probability tensor / feature matrix, cached on disk."""

    def __init__(self, encoder: str = "dleemiller/finecat-nli-l", *, device: str = "cuda",
                 batch_size: int = 128, max_text_chars: int = 1200,
                 cache_path: str | Path = DEFAULT_CACHE, verbose: bool = True):
        self.cfg = EncoderConfig(
            model=encoder, device=device, batch_size=batch_size,
            max_text_chars=max_text_chars, verbose=verbose,
        )
        self.cache = ScoreCache(str(cache_path))
        self.costs = CostTracker()
        self.scorer = EntailmentScorer(self.cfg, self.cache, self.costs)

    def probs(self, texts: list[str], pool: list[str]) -> np.ndarray:
        """(n_texts, n_hyp, 3) probabilities [entail, neutral, contradict]."""
        return self.scorer.probs(list(texts), list(pool))

    def features(self, texts: list[str], pool: list[str],
                 score_mode: str = "entail_contradict") -> np.ndarray:
        """Feature matrix per score_mode.

        entail_contradict -> (n, 2m) = [P(entail) | P(contradict)]
        entail            -> (n, m)  = P(entail)
        contrast          -> (n, m)  = P(entail) - P(contradict)
        """
        p = self.probs(texts, pool)
        e, c = p[:, :, 0], p[:, :, 2]
        if score_mode == "entail":
            return e
        if score_mode == "contrast":
            return e - c
        if score_mode == "entail_contradict":
            return np.concatenate([e, c], axis=1)
        raise ValueError(f"unknown score_mode {score_mode!r}")

    def cost_summary(self) -> dict:
        return self.costs.to_dict()
