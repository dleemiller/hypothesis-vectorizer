"""STAGE 2 — evolve the pool by GROW-THEN-SELECT (monotonic).

Each round MERGES the current pool with fresh LM refills into a superset (~2x size), then PRUNES
back to the target size by marginal-over-lexical permutation importance with covariance dedup. A
refill enters only if it OUT-RANKS an incumbent — so, unlike a blind prune-then-replace, a
good-but-unlucky feature is never swapped out for an untested worse one. An ACCEPT GATE then keeps
the merged pool only if it does not regress beyond round noise (else it reverts and next round tries
different refills), and the shipped pool is the best-held-out CHECKPOINT, not the last. Net:
evolution moves forward or holds, never blindly backward, while still exploring every round.

- rank by CV-fold permutation importance + cross-fold sign STABILITY (single-split ranking churned
  50% across seeds); errors drive confusion HOT SPOTS fed to the refill proposer;
- grow: propose refills targeting hot spots (batches force pattern-level proposals);
- select: keep the top `size` of (pool ∪ refills) by importance, skipping covariance-redundant ones;
- accept gate + best-checkpoint => forward-only; stop on plateau or when refills run dry.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from .config import PoolConfig
from .data import Bundle, labeled_examples, stratified_indices
from .dedup import Deduper, _zscore
from .encoder import EntailmentScorer
from .proposer import Proposer

_ROUND_NOISE = 0.003  # held-out jitter from HGB thread nondeterminism; accept/tie thresholds use it


@dataclass
class Ranking:
    """Per-HYPOTHESIS ranking aggregated over its two feature columns."""

    order: np.ndarray  # hypothesis indices, most useful first
    importance: np.ndarray  # mean CV permutation importance, summed over the 2 columns
    stability: np.ndarray  # fraction of folds where either column helped
    errors: list[tuple[int, int]]  # (local text index, predicted class) over all held-out folds

    @property
    def heldout_accuracy(self) -> float:
        return 1.0 - len(self.errors) / self._n

    _n: int = 0


@dataclass
class Checkpoint:
    """A snapshot of the pool as it ENTERED a round, with the held-out accuracy it achieved.
    Evolution can dip after its peak (and the final round's survivors+refills are never scored),
    so the pool we ship is CHOSEN from these checkpoints, not taken as the last round's output.
    Persisted to runs/<name>/checkpoints.jsonl so any round is recoverable and the choice is
    auditable."""

    round: int
    heldout_acc: float
    pool: list[str]

    @property
    def n_hyps(self) -> int:
        return len(self.pool)

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "heldout_acc": round(self.heldout_acc, 4),
            "n_hyps": self.n_hyps,
            "pool": self.pool,
        }


def select_checkpoint(checkpoints: list[Checkpoint], noise: float = 0.003) -> Checkpoint:
    """Choose the pool to ship: highest held-out accuracy, breaking ties within `noise` toward
    the SMALLER pool (fewer cross-encoder forward passes at inference), then the earlier round.
    `noise` should exceed round-to-round held-out jitter (~0.003 from HGB thread nondeterminism)
    so we neither ship a post-peak dip nor chase a +1e-4 gain into a larger, costlier pool."""
    best = max(c.heldout_acc for c in checkpoints)
    contenders = [c for c in checkpoints if c.heldout_acc >= best - noise]
    return min(contenders, key=lambda c: (c.n_hyps, c.round))


def rank_hypotheses(
    x: np.ndarray, y: np.ndarray, m: int, seed: int, folds: int = 4, lex: np.ndarray | None = None
) -> Ranking:
    """x is the (n, 2m) NLI feature matrix; every sample is held out in exactly one fold.

    `lex` (n, d), if given, is the cheap lexical channel as a FIXED baseline: it joins every
    fold's model and drives the errors/hot spots, but is never ranked or pruned. Each NLI
    hypothesis's importance then measures its MARGINAL value ON TOP OF lexical — a hypothesis
    whose signal TF-IDF already carries scores ~0 and dies. Since an NLI feature costs a
    cross-encoder forward pass at inference and TF-IDF is ~free, this minimizes the NLI pool
    (and thus per-prediction cost) down to hypotheses that add semantics lexical cannot."""
    xx = x if lex is None else np.concatenate([x, lex], axis=1)
    imps = np.zeros((folds, xx.shape[1]))
    errors: list[tuple[int, int]] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (itr, ihe) in enumerate(skf.split(xx, y)):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(xx[itr], y[itr])
        imps[f] = permutation_importance(
            clf, xx[ihe], y[ihe], n_repeats=3, random_state=seed
        ).importances_mean
        preds = clf.predict(xx[ihe])
        errors += [(int(i), int(p)) for i, p in zip(ihe, preds) if p != y[i]]

    # rank/prune only the NLI columns (first 2m: [entail | contradict]); lexical columns are baseline
    col_stability = (imps > 0).mean(axis=0)
    mean_imp = imps.mean(axis=0)
    hyp_importance = mean_imp[:m] + mean_imp[m : 2 * m]
    hyp_stability = np.maximum(col_stability[:m], col_stability[m : 2 * m])
    order = np.lexsort((-hyp_stability, -hyp_importance))
    r = Ranking(order=order, importance=hyp_importance, stability=hyp_stability, errors=errors)
    r._n = len(y)
    return r


def hotspots(
    errors: list[tuple[int, int]], y: np.ndarray, n_classes: int, min_rate: float = 0.04
) -> list[list[int]]:
    """Groups of mutually-confused classes: connected components of the symmetrized
    confusion graph, thresholded on confusion rate. Pairwise views miss block
    structure — a clique of confused classes needs carving as a group."""
    counts = np.zeros((n_classes, n_classes))
    for i, p in errors:
        counts[y[i], p] += 1
    sym = counts + counts.T
    support = np.bincount(y, minlength=n_classes).astype(float)

    adj: list[list[int]] = [[] for _ in range(n_classes)]
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            if sym[a, b] / max(1.0, min(support[a], support[b])) >= min_rate:
                adj[a].append(b)
                adj[b].append(a)

    seen: set[int] = set()
    groups = []
    for start in range(n_classes):
        if start in seen or not adj[start]:
            continue
        stack, comp = [start], []
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                comp.append(c)
                stack.extend(adj[c])
        if len(comp) >= 2:
            groups.append(sorted(comp))
    groups.sort(key=lambda g: -sum(sym[a, b] for a in g for b in g))
    return groups


def _failure_reason(col_e: np.ndarray, col_c: np.ndarray, survivor_cols: np.ndarray) -> str:
    """Why did a pruned hypothesis fail? Tells the LM whether to abandon the
    concept (undetectable) or just this angle on it."""
    if np.std(col_e) < 0.02 and np.std(col_c) < 0.02:
        return "the NLI encoder scores every text identically on this — undetectable property"
    if survivor_cols.shape[1]:
        best = 0.0
        for col in (col_e, col_c):
            cc = col - col.mean()
            sc = survivor_cols - survivor_cols.mean(axis=0)
            denom = np.linalg.norm(cc) * np.linalg.norm(sc, axis=0)
            best = max(best, float(np.max(np.abs((cc @ sc) / np.where(denom == 0, np.inf, denom)))))
        if best > 0.9:
            return "redundant — signal nearly identical to a kept hypothesis"
    return "detectable but carries no held-out predictive value for these classes"


def _confusion_evidence(
    bundle: Bundle,
    sub: np.ndarray,
    errors: list[tuple[int, int]],
    groups: list[list[int]],
    rng: np.random.Generator,
) -> list[str]:
    """Hot spots as grouped example batches; scattered errors as counts only."""
    names = bundle.class_names
    global_errors = [(int(sub[i]), p) for i, p in errors]
    rng.shuffle(global_errors)
    evidence, used = [], set()
    for g in groups[:3]:
        gset = set(g)
        in_group = [(i, p) for i, p in global_errors if bundle.y_train[i] in gset and p in gset][:8]
        if not in_group:
            continue
        lines = [
            f"HOT SPOT — the classes {{{', '.join(names[c] for c in g)}}} are mutually "
            f"confused ({len(in_group)}+ errors shown). Write hypotheses for what these "
            f"errors SHARE that would carve the classes apart:"
        ]
        lines += [
            f"  [true: {names[bundle.y_train[i]]}, predicted: {names[p]}] {bundle.train_texts[i][:220]}"
            for i, p in in_group
        ]
        evidence.append("\n".join(lines))
        used.update(i for i, _ in in_group)
    scattered = Counter((bundle.y_train[i], p) for i, p in global_errors if i not in used)
    if scattered:
        evidence.append(
            "Scattered confusions outside hot spots (counts only): "
            + "; ".join(f"{names[t]}→{names[p]}: {c}" for (t, p), c in scattered.most_common(5))
        )
    return evidence


def _heldout_accuracy(
    x: np.ndarray, y: np.ndarray, seed: int, lex: np.ndarray | None = None, folds: int = 4
) -> float:
    """Cheap CV held-out accuracy (no permutation importance) — for the accept gate. Uses the same
    fold seed and HGB seed as rank_hypotheses, so it matches that round's reported held-out."""
    xx = x if lex is None else np.concatenate([x, lex], axis=1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    errors = 0
    for itr, ihe in skf.split(xx, y):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(xx[itr], y[itr])
        errors += int((clf.predict(xx[ihe]) != y[ihe]).sum())
    return 1.0 - errors / len(y)


def _select_topk(order: np.ndarray, x: np.ndarray, m: int, k: int, cov_threshold: float) -> list[int]:
    """From `order` (candidate indices, most important first), keep up to k, skipping any whose
    entail-score vector correlates > cov_threshold with an already-kept, higher-ranked one — so we
    don't fill slots with covariance-redundant features. Backfill (ignoring covariance) if dedup
    leaves us short of k. `x` is the (n, 2m) candidate feature matrix; entail cols are x[:, :m]."""
    ent = x[:, :m]
    n = x.shape[0]
    kept: list[int] = []
    kept_z: list[np.ndarray] = []
    for idx in order:
        i = int(idx)
        zv = _zscore(ent[:, i])
        if any(abs(float(zv @ zk)) / n > cov_threshold for zk in kept_z):
            continue
        kept.append(i)
        kept_z.append(zv)
        if len(kept) >= k:
            return kept
    for idx in order:  # backfill ignoring covariance if dedup left us short
        i = int(idx)
        if i not in kept:
            kept.append(i)
            if len(kept) >= k:
                break
    return kept


def evolve(
    bundle: Bundle,
    pool: list[str],
    scorer: EntailmentScorer,
    proposer: Proposer,
    deduper: Deduper,
    cfg: PoolConfig,
    seed: int,
    lex_train: np.ndarray | None = None,
) -> tuple[list[str], list[dict], list[Checkpoint]]:
    """GROW-THEN-SELECT evolution. Returns (shipped pool, per-round history, checkpoints).

    Each round: (1) score+rank the current pool (marginal over `lex_train`) and CHECKPOINT it;
    (2) propose refills targeting confusion hot spots and MERGE them into a superset (~2x size);
    (3) SELECT the top `size` of the superset by importance + covariance dedup; (4) an ACCEPT GATE
    keeps the merged pool only if its held-out doesn't regress beyond round noise, else it reverts.
    The shipped pool is the best-held-out checkpoint. A refill thus enters only by OUT-RANKING an
    incumbent, and the pool never blindly regresses — while every round still explores.

    `lex_train` (n_train, d), if given, is the lexical channel over ALL train texts; the ranking
    sees it as a fixed baseline so hypotheses are ranked by MARGINAL value over lexical."""
    rng = np.random.default_rng(seed)
    if cfg.rank_sample and cfg.rank_sample < len(bundle.train_texts):
        sub = stratified_indices(bundle.y_train, cfg.rank_sample, rng)
    else:
        sub = np.arange(len(bundle.train_texts))
    sub_texts = [bundle.train_texts[i] for i in sub]
    sub_y = bundle.y_train[sub]
    lex_sub = lex_train[sub] if lex_train is not None else None
    examples = labeled_examples(bundle, per_class=3, rng=rng)

    seen = {s.casefold() for s in pool}
    history: list[dict] = []
    checkpoints: list[Checkpoint] = []
    best_acc, since_best = -1.0, 0
    grow_to = 2 * cfg.size  # merge pool + refills up to ~2x, then select back to `size`

    round_i = 0
    for round_i in range(cfg.rounds):
        m = len(pool)
        x = scorer.features(sub_texts, pool)
        ranking = rank_hypotheses(x, sub_y, m, seed, lex=lex_sub)
        acc = ranking.heldout_accuracy
        checkpoints.append(Checkpoint(round=round_i, heldout_acc=acc, pool=list(pool)))
        if acc > best_acc + 1e-4:
            best_acc, since_best = acc, 0
        else:
            since_best += 1

        # confusion targeting + weak-feature feedback (lowest marginal importance first)
        groups = hotspots(ranking.errors, sub_y, bundle.n_classes)
        evidence = _confusion_evidence(bundle, sub, ranking.errors, groups, rng)
        ranked_pool = [pool[int(i)] for i in ranking.order]
        survivor_cols = x[:, [int(i) for i in ranking.order]]
        weakest = [int(i) for i in ranking.order[::-1][: max(1, m // 3)]]
        failed = [f"{pool[i]} ({_failure_reason(x[:, i], x[:, m + i], survivor_cols)})" for i in weakest]

        # GROW: propose refills to reach the ~2x superset
        n_refill = max(0, grow_to - m)
        refills: list[str] = []
        if n_refill:
            proposed = proposer.refill(
                bundle.task, bundle.class_descriptions, examples, ranked_pool, failed, evidence, n=n_refill
            )
            refills, _ = deduper.filter(proposed, against=pool, seen=seen)

        # MERGE + SELECT: keep the best `size` of (pool ∪ refills) by importance + covariance dedup;
        # ACCEPT GATE commits the merge only if it does not regress beyond round noise
        accepted, merged_acc = True, acc
        if refills:
            candidates = list(pool) + refills
            xc = scorer.features(sub_texts, candidates)
            rank_c = rank_hypotheses(xc, sub_y, len(candidates), seed, lex=lex_sub)
            keep_idx = _select_topk(rank_c.order, xc, len(candidates), cfg.size, deduper.thr)
            new_pool = [candidates[i] for i in keep_idx]
            xn = scorer.features(sub_texts, new_pool)  # cached: all scored just above
            merged_acc = _heldout_accuracy(xn, sub_y, seed, lex=lex_sub)
            if merged_acc >= best_acc - _ROUND_NOISE:
                pool = new_pool
            else:
                accepted = False

        history.append(
            {
                "round": round_i,
                "heldout_acc": round(acc, 4),
                "merged_acc": round(merged_acc, 4),
                "accepted": accepted,
                "survivors": list(pool),
                "failed": failed,
                "refills": refills,
            }
        )
        print(
            f"--- evolve round {round_i}: heldout {acc:.4f}, pool {m}, +{len(refills)} refills "
            f"-> merged {merged_acc:.4f} ({'accepted' if accepted else 'REVERTED'})",
            flush=True,
        )

        if since_best >= cfg.patience:
            print(f"--- evolve stop: no held-out improvement for {cfg.patience} rounds", flush=True)
            break
        if not refills:
            print("--- evolve stop: exploration exhausted (no new refills)", flush=True)
            break

    if not checkpoints:  # cfg.rounds == 0
        return pool, history, checkpoints
    chosen = select_checkpoint(checkpoints)
    if chosen.round != round_i:
        print(
            f"--- evolve: shipping best checkpoint from round {chosen.round} "
            f"(heldout {chosen.heldout_acc:.4f}, {chosen.n_hyps} hyps), not the last round",
            flush=True,
        )
    return chosen.pool, history, checkpoints
