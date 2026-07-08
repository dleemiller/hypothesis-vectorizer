#!/usr/bin/env python
"""Hypothesis-inspection utilities for the interpretability section (RQ2).

Given a dataset + a hypothesis pool + an encoder, this fits an HV head on the full train set
and dumps the readable, paper-ready analyses:

  * global feature importance (permutation importance of each hypothesis)
  * top hypotheses per class (by class-conditional mean entailment)
  * redundant hypothesis clusters (|correlation| of entail-score vectors above a threshold)
  * high-variance hypotheses across CV folds (stability)
  * per hypothesis: the test examples that most entail and most contradict it
  * error cases with their top-activating hypotheses

Outputs a markdown report + CSVs under experiments/results/processed/<run_id>/.

Usage:
    uv run python experiments/scripts/inspect_hypotheses.py --dataset trec \
        --encoder dleemiller/finecat-nli-l --run-id trec_expert_inspect
    # with an LM-generated pool once available:
    uv run python experiments/scripts/inspect_hypotheses.py --dataset trec --pool-json pool.json ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, hypotheses, repro  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "processed"


def load_pool(args, dataset: str) -> tuple[list[str], list[str] | None]:
    if args.pool_json:
        obj = json.loads(pathlib.Path(args.pool_json).read_text())
        if obj and isinstance(obj[0], dict):  # [{"text":..,"intended_class":..}]
            return [h["text"] for h in obj], [h.get("intended_class") for h in obj]
        return list(obj), None
    pool, tags = hypotheses.expert_pool(dataset)
    return pool, tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-json", default=None)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--redundancy-threshold", type=float, default=0.9)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold

    from hypothesis_vectorizer.train.head import cv_selected_head

    raw = datasets.load_raw(args.dataset, test_size=args.test_size, test_seed=args.seed)
    pool, tags = load_pool(args, args.dataset)
    fz = NLIFeaturizer(encoder=args.encoder, device=args.device, verbose=True)

    # subsample train to train_size (stratified via the k-shot machinery on a large k is overkill;
    # use the full pool if smaller)
    n_tr = min(args.train_size, len(raw.train_texts))
    ridx = datasets.subsample_indices(raw.y_train, "all", args.seed)[:n_tr]
    tr_texts = [raw.train_texts[i] for i in ridx]
    tr_y = raw.y_train[ridx]

    print(f"[inspect] {args.dataset}: {len(tr_texts)} train / {len(raw.test_texts)} test, "
          f"{len(pool)} hypotheses on {args.encoder}")
    Xtr = fz.features(tr_texts, pool, score_mode="entail_contradict")
    Xte = fz.features(raw.test_texts, pool, score_mode="entail_contradict")
    Etr = fz.features(tr_texts, pool, score_mode="entail")   # (n, m) P(entail), for readability
    Ete = fz.features(raw.test_texts, pool, score_mode="entail")

    head, params, cv = cv_selected_head(Xtr, tr_y, args.seed, folds=4)
    m = len(pool)
    names = pool

    # ---- global permutation importance (serial: no process forks in a CUDA process) ----------
    pi = permutation_importance(head, Xte, raw.y_test, n_repeats=10, random_state=args.seed, n_jobs=1)
    # aggregate the entail(j) and contradict(j+m) columns back onto hypothesis j
    imp = pi.importances_mean[:m] + pi.importances_mean[m:]
    order = np.argsort(imp)[::-1]

    # ---- per-class mean entailment (which hypotheses light up for each class) ----------------
    per_class_top: dict[str, list] = {}
    for c, cname in enumerate(raw.class_names):
        mask = raw.y_test == c
        if mask.sum() == 0:
            continue
        mean_e = Ete[mask].mean(axis=0)
        top = np.argsort(mean_e)[::-1][: args.top]
        per_class_top[cname] = [(names[j], float(mean_e[j]), tags[j] if tags else None) for j in top]

    # ---- redundancy clusters (|corr| of entail vectors) --------------------------------------
    C = np.corrcoef(Etr.T)
    redundant = []
    for i in range(m):
        for j in range(i + 1, m):
            if abs(C[i, j]) >= args.redundancy_threshold:
                redundant.append((names[i], names[j], float(C[i, j])))
    redundant.sort(key=lambda t: -abs(t[2]))

    # ---- cross-fold importance stability -----------------------------------------------------
    fold_imp = []
    skf = StratifiedKFold(4, shuffle=True, random_state=args.seed)
    for tri, vai in skf.split(Xtr, tr_y):
        h, _p, _c = cv_selected_head(Xtr[tri], tr_y[tri], args.seed, folds=3)
        pif = permutation_importance(h, Xtr[vai], tr_y[vai], n_repeats=5, random_state=args.seed, n_jobs=1)
        fold_imp.append(pif.importances_mean[:m] + pif.importances_mean[m:])
    fold_imp = np.array(fold_imp)
    imp_std = fold_imp.std(axis=0)

    # ---- per-hypothesis exemplars (most entailing / most contradicting test texts) -----------
    exemplars = {}
    Cte = fz.features(raw.test_texts, pool, score_mode="contrast")  # entail - contradict
    for j in order[: args.top]:
        e_top = np.argsort(Ete[:, j])[::-1][:3]
        c_top = np.argsort(Cte[:, j])[:3]  # most contradicted
        exemplars[names[j]] = {
            "entails": [raw.test_texts[i][:120] for i in e_top],
            "contradicts": [raw.test_texts[i][:120] for i in c_top],
        }

    # ---- error cases with top-activating hypotheses ------------------------------------------
    pred = head.predict(Xte)
    errs = np.flatnonzero(pred != raw.y_test)[:10]
    error_rows = []
    for i in errs:
        acts = np.argsort(Ete[i])[::-1][:3]
        error_rows.append({
            "text": raw.test_texts[i][:120],
            "true": raw.class_names[raw.y_test[i]],
            "pred": raw.class_names[pred[i]],
            "top_hypotheses": [names[j] for j in acts],
        })

    # ---- write report ------------------------------------------------------------------------
    run_id = args.run_id or f"inspect_{args.dataset}"
    outdir = OUT / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Hypothesis inspection — {args.dataset} ({args.encoder})", "",
             f"Head: `{params}`  CV-train acc {cv:.4f}. {len(pool)} hypotheses.", "",
             f"## Top {args.top} hypotheses by global permutation importance", "",
             "| rank | importance | ±fold-std | hypothesis | intended |", "|---|---|---|---|---|"]
    for rank, j in enumerate(order[: args.top], 1):
        tag = tags[j] if tags else ""
        lines.append(f"| {rank} | {imp[j]:.4f} | {imp_std[j]:.4f} | {names[j]} | {tag} |")

    lines += ["", "## Top hypotheses per class (mean P(entail) on that class's test texts)", ""]
    for cname, rows in per_class_top.items():
        lines.append(f"**{cname}**")
        for h, e, tag in rows[:5]:
            lines.append(f"- {e:.3f}  {h}" + (f"  _(→{tag})_" if tag else ""))
        lines.append("")

    lines += [f"## Redundant hypothesis pairs (|corr| ≥ {args.redundancy_threshold})", ""]
    if redundant:
        for a, b, r in redundant[:15]:
            lines.append(f"- corr {r:+.2f}: “{a}” ⟷ “{b}”")
    else:
        lines.append("_None above threshold — the pool is non-redundant._")

    lines += ["", "## Per-hypothesis exemplars (top by importance)", ""]
    for h, ex in exemplars.items():
        lines.append(f"**{h}**")
        lines.append("- entailed by: " + " | ".join(f"“{t}”" for t in ex["entails"]))
        lines.append("- contradicted by: " + " | ".join(f"“{t}”" for t in ex["contradicts"]))
        lines.append("")

    lines += ["## Error cases with top-activating hypotheses", ""]
    for r in error_rows:
        lines.append(f"- [true **{r['true']}** → pred **{r['pred']}**] “{r['text']}”")
        lines.append("  top: " + "; ".join(r["top_hypotheses"]))

    (outdir / "inspection.md").write_text("\n".join(lines) + "\n")

    # CSVs
    import csv

    with (outdir / "importance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "importance", "fold_std", "hypothesis", "intended_class"])
        for rank, j in enumerate(order, 1):
            w.writerow([rank, f"{imp[j]:.5f}", f"{imp_std[j]:.5f}", names[j], tags[j] if tags else ""])

    repro.Manifest(run_id=run_id, config=vars(args), seed=args.seed, dataset=args.dataset,
                   encoder=args.encoder, pool_id="expert" if not args.pool_json else args.pool_json,
                   extra={"cv_train_acc": cv, "head_params": params}).write(outdir)
    print(f"[inspect] wrote {outdir / 'inspection.md'} and importance.csv")


if __name__ == "__main__":
    main()
