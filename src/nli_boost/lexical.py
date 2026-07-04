"""Optional static lexical features, concatenated with the hypothesis features.

Motivation (measured): TF-IDF standalone reaches 0.828 on TREC and 0.565 on
20NG — signal the NLI encoder may not carry. If complementary, concatenation
buys points; if subsumed, the CV head ignores the extra columns.

Two kinds:
- tfidf_svd:  TF-IDF (fit on TRAIN ONLY — no leakage) reduced to `dims` by SVD
- wordllama:  static per-text embeddings (corpus-independent, deterministic)

Applied at the head stage only; evolution remains hypothesis-only.
"""

import numpy as np

from .config import LexicalConfig


class LexicalFeaturizer:
    def __init__(self, cfg: LexicalConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self._pipeline = None
        self._wl = None

    def fit(self, train_texts: list[str]) -> "LexicalFeaturizer":
        if self.cfg.kind == "tfidf_svd":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            tf = vec.fit_transform(train_texts)
            dims = min(self.cfg.dims, tf.shape[1] - 1)  # SVD needs dims < vocab size
            self._pipeline = make_pipeline(vec, TruncatedSVD(n_components=dims, random_state=self.seed))
            self._pipeline.fit(train_texts)
        elif self.cfg.kind == "wordllama":
            from wordllama import WordLlama

            self._wl = WordLlama.load(trunc_dim=self.cfg.dims)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.cfg.kind == "tfidf_svd":
            return np.asarray(self._pipeline.transform(texts), dtype=np.float32)
        if self.cfg.kind == "wordllama":
            return np.asarray(self._wl.embed(texts), dtype=np.float32)
        raise ValueError(f"no lexical features for kind={self.cfg.kind!r}")
