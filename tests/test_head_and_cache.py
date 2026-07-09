import numpy as np
from conftest import FakeScorer, make_bundle

from hypothesis_vectorizer.cache import ScoreCache
from hypothesis_vectorizer.encoder import digest, normalize
from hypothesis_vectorizer.train.head import cv_selected_head, evaluate


def test_cv_selected_head_learns_separable_data(fast_models):
    bundle = make_bundle(n=600)
    scorer = FakeScorer()
    pool = [f"f{i}" for i in range(8)]
    x = scorer.features(bundle.train_texts, pool)
    head, params, cv_acc = cv_selected_head(x, bundle.y_train, seed=0)
    assert cv_acc > 0.9  # f0/f1 define the label
    assert params["kind"] in ("rf", "hgb")
    x_test = scorer.features(bundle.test_texts, pool)
    m = evaluate(bundle.y_test, head.predict_proba(x_test), bundle.n_classes)
    assert m["accuracy"] > 0.9
    assert set(m) == {"accuracy", "macro_f1", "logloss"}


def test_cache_roundtrip_and_model_isolation(tmp_path):
    cache = ScoreCache(tmp_path / "c.sqlite")
    hh = digest("The text is about sports.")
    rows = [(digest(f"t{i}"), f"t{i}", np.array([i, 0.5, -1.0], dtype=np.float32)) for i in range(700)]
    cache.put_logits("m", hh, "The text is about sports.", rows)
    got = cache.get_logits("m", hh, [r[0] for r in rows] + [digest("missing")])
    assert len(got) == 700
    np.testing.assert_allclose(got[rows[3][0]], [3.0, 0.5, -1.0])
    assert cache.get_logits("other-model", hh, [rows[0][0]]) == {}


def test_normalize_is_cache_key_stable():
    assert normalize("Hello   world\n\tfoo", 1200) == normalize("Hello world foo", 1200)
    assert len(normalize("x" * 5000, 1200)) == 1200


def test_encoder_full_features_layout():
    """features='full' appends P(neutral) as a third block: (n, 3m), [entail | contradict | neutral],
    derived from the SAME cached logits (no re-scoring)."""
    import numpy as np

    from hypothesis_vectorizer.config import EncoderConfig
    from hypothesis_vectorizer.costs import CostTracker
    from hypothesis_vectorizer.encoder import EntailmentScorer, digest, normalize

    cache = ScoreCache(":memory:")
    texts, hyp = ["alpha text", "beta text"], "the hypothesis"
    logits = {"alpha text": np.array([2.0, 0.0, -2.0]), "beta text": np.array([-2.0, 2.0, 0.0])}
    rows = [(digest(normalize(t, 1200)), normalize(t, 1200), z) for t, z in logits.items()]
    cache.put_logits("m", digest(hyp), hyp, rows)

    def scorer(mode):
        return EntailmentScorer(EncoderConfig(model="m", features=mode, device="cpu"), cache, CostTracker())

    two = scorer("entail_contradict").features(texts, [hyp])
    full = scorer("full").features(texts, [hyp])
    assert two.shape == (2, 2) and full.shape == (2, 3)
    np.testing.assert_allclose(full[:, :2], two)  # first 2m block identical in both layouts
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-6)  # e+c+n = softmax simplex
    assert full[0, 0] > 0.8 and full[1, 2] > 0.4  # text0 entails; text1's neutral is the mid logit


def test_wal_probe_is_per_process_and_cache_works(tmp_path):
    """Two caches in the SAME dir (the concurrent-run case) both init cleanly; the WAL probe is
    per-pid so neither unlinks the other's probe mid-write. On a normal FS both get WAL."""
    import numpy as np

    from hypothesis_vectorizer.cache import ScoreCache
    from hypothesis_vectorizer.encoder import digest

    c1 = ScoreCache(tmp_path / "nli.sqlite")
    c2 = ScoreCache(tmp_path / "nli.sqlite")  # second opener, same dir + file
    assert c1.journal_mode in ("WAL", "DELETE") and c2.journal_mode == c1.journal_mode
    c1.put_logits("m", digest("h"), "h", [(digest("t"), "t", np.array([1.0, 0.0, -1.0]))])
    assert digest("t") in c2.get_logits("m", digest("h"), [digest("t")])  # round-trips across handles
    # no probe files leaked
    assert not list(tmp_path.glob(".wal_probe*"))
