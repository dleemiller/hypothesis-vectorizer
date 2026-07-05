import numpy as np
from conftest import FakeProposer, FakeScorer, TextOnlyDeduper, make_bundle

from nli_boost.config import PoolConfig
from nli_boost.evolve import Checkpoint, evolve, hotspots, rank_hypotheses, select_checkpoint


def test_rank_informative_hypotheses_first():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]
    scorer = FakeScorer()
    x = scorer.features(bundle.train_texts, pool)
    r = rank_hypotheses(x, bundle.y_train, m=len(pool), seed=0)
    assert set(r.order[:2].tolist()) == {0, 1}
    assert r.stability[7] == 0.0  # the constant feature never helps in any fold
    assert 0 < r.heldout_accuracy <= 1


def test_hotspots_group_mutually_confused_classes():
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    # classes 0 and 1 confuse each other heavily; class 2 is clean
    errors = [(i, 1) for i in range(10)] + [(i, 0) for i in range(50, 60)]
    groups = hotspots(errors, y, n_classes=3)
    assert groups == [[0, 1]]


def test_evolve_grow_select_flags_dead_feature_and_ships_checkpoint():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]  # f7 is a constant -> dead weight
    proposer = FakeProposer(refill_batches=[[f"f{2 + i} variant {i}" for i in range(4)]] * 6)
    cfg = PoolConfig(size=8, rounds=6, patience=2, rank_sample=0)

    final, history, checkpoints = evolve(bundle, pool, FakeScorer(), proposer, TextOnlyDeduper(), cfg, seed=0)

    # grow-then-select schema: every round records the merge decision
    keys = {"heldout_acc", "merged_acc", "accepted", "survivors", "failed", "refills"}
    assert all(keys <= h.keys() for h in history)
    # the constant feature is flagged weak with the undetectable reason
    all_failed = [f for h in history for f in h["failed"]]
    assert any(f.startswith("f7") and "undetectable" in f for f in all_failed)
    # informative features are present in the shipped pool
    assert any(h.startswith("f0") for h in final) and any(h.startswith("f1") for h in final)
    # a checkpoint per round; shipped pool is the best-held-out checkpoint, sized to the target
    assert len(checkpoints) == len(history)
    assert final == select_checkpoint(checkpoints).pool
    assert all(c.n_hyps == cfg.size for c in checkpoints)
    # the refill LM saw the current pool and failure reasons
    assert proposer.refill_calls and proposer.refill_calls[0]["failed"]


def test_select_checkpoint_prefers_best_then_smaller_pool():
    cks = [
        Checkpoint(round=0, heldout_acc=0.90, pool=["a", "b", "c", "d"]),
        Checkpoint(round=1, heldout_acc=0.95, pool=["a", "b", "c", "d", "e"]),  # peak, larger
        Checkpoint(round=2, heldout_acc=0.949, pool=["a", "b"]),  # within noise of peak, smaller
        Checkpoint(round=3, heldout_acc=0.80, pool=["a"]),  # post-peak dip, ignored
    ]
    # 0.949 is within default noise (0.003) of the 0.95 peak -> take the smaller (2-hyp) pool
    chosen = select_checkpoint(cks)
    assert chosen.round == 2 and chosen.n_hyps == 2
    # with zero noise tolerance, the strict max wins
    assert select_checkpoint(cks, noise=0.0).round == 1


def test_evolve_ships_peak_not_last_round():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]
    proposer = FakeProposer(refill_batches=[[f"f{2 + i} variant {i}" for i in range(4)]] * 6)
    cfg = PoolConfig(size=8, rounds=6, patience=2, rank_sample=0)
    final, _, checkpoints = evolve(bundle, pool, FakeScorer(), proposer, TextOnlyDeduper(), cfg, seed=0)
    # never ship a pool strictly worse than the best checkpoint's held-out
    best = select_checkpoint(checkpoints)
    assert all(best.heldout_acc + 1e-9 >= c.heldout_acc or c.round == best.round for c in checkpoints)
    assert final == best.pool


def test_evolve_accept_gate_ships_best_checkpoint():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]
    proposer = FakeProposer(refill_batches=[["f2 fresh"], ["f3 fresh"], ["f4 fresh"]])
    cfg = PoolConfig(size=8, rounds=3, patience=3, rank_sample=0)
    final, history, checkpoints = evolve(bundle, pool, FakeScorer(), proposer, TextOnlyDeduper(), cfg, seed=0)
    # shipped pool is the best checkpoint (never a post-peak dip)
    best = select_checkpoint(checkpoints)
    assert final == best.pool
    assert best.heldout_acc == max(c.heldout_acc for c in checkpoints)
    # every round records an accept/revert decision and a merged-pool score
    assert all("accepted" in h and "merged_acc" in h for h in history)
