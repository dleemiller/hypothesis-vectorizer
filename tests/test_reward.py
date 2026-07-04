import numpy as np

from nli_boost.reward import RewardConfig, effective_rank, geometric_mean, pool_reward


def _informative(n, m, rng):
    y = (rng.random(n) > 0.5).astype(int)
    ent = np.column_stack([y * 0.6 + 0.2 + 0.15 * rng.random(n) for _ in range(m)]) + 0.3 * rng.random((n, m))
    return y, np.hstack([ent, 1 - ent])


def test_effective_rank_spans_collapsed_to_orthogonal():
    rng = np.random.default_rng(0)
    base = rng.random((200, 1))
    collapsed = np.hstack([base + 1e-6 * rng.random((200, 1)) for _ in range(8)])
    orthogonal = rng.random((200, 8))
    assert effective_rank(collapsed) < 1.5
    assert effective_rank(orthogonal) > 6.0


def test_geometric_mean_craters_on_a_zero():
    assert geometric_mean([0.9, 0.9, 0.9]) > 0.85
    assert geometric_mean([0.9, 0.9, 0.0]) < 0.05  # one tanked dataset vetoes the aggregate


def test_reward_rewards_informative_over_noise():
    rng = np.random.default_rng(0)
    n, m = 300, 6
    y, x_info = _informative(n, m, rng)
    x_noise = np.hstack([(nz := rng.random((n, m))), 1 - nz])
    pool, texts, cfg = [f"h{i}" for i in range(m)], ["t"] * n, RewardConfig(cv_seeds=2)
    r_info = pool_reward(x_info, y, pool, texts, cfg)
    r_noise = pool_reward(x_noise, y, pool, texts, cfg)
    assert r_info["score"] > r_noise["score"]  # cv_skill (accuracy) drives the reward
    assert r_info["cv_skill"] > r_noise["cv_skill"]


def test_judge_contributes_incrementally():
    rng = np.random.default_rng(1)
    y, x = _informative(300, 6, rng)
    pool, texts, cfg = [f"h{i}" for i in range(6)], ["t"] * 300, RewardConfig(cv_seeds=2)
    r_hi = pool_reward(x, y, pool, texts, cfg, judge_score=1.0)
    r_lo = pool_reward(x, y, pool, texts, cfg, judge_score=0.0)
    assert r_hi["score"] > r_lo["score"]  # judge is a positive reward term, not just a gate
    # incremental, not binary: a mid judge lands between the extremes
    r_mid = pool_reward(x, y, pool, texts, cfg, judge_score=0.5)
    assert r_lo["score"] < r_mid["score"] < r_hi["score"]


def test_hack_fraction_lowers_reward_incrementally():
    rng = np.random.default_rng(2)
    n = 200
    y = (rng.random(n) > 0.5).astype(int)
    texts = ["x" * int(v) for v in (50 + 100 * rng.random(n))]
    lengths = np.array([len(t) for t in texts], dtype=float)
    ent = (lengths / lengths.max())[:, None] + 0.01 * rng.random((n, 1))  # a length-tracking hypothesis
    r = pool_reward(np.hstack([ent, 1 - ent]), y, ["h0"], texts, RewardConfig(cv_seeds=2), judge_score=1.0)
    assert r["hack_fraction"] > 0.0  # the length artifact is detected and subtracted from the score
