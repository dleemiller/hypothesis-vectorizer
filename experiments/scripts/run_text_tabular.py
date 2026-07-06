#!/usr/bin/env python
"""RQ5 text+tabular marginal-value study (CFPB-style, written generically over any CSV).

Question: on a task with BOTH free text and categorical metadata, how much does each feature
family add on top of the others? We fit six feature configurations with the same head and
compare accuracy / macro-F1 / (binary) ROC-AUC:

  tabular_only     one-hot of --cat-cols  (OneHotEncoder(handle_unknown='ignore'))
  tfidf_only       TF-IDF word 1-2 grams on --text-col -> TruncatedSVD(--svd-dims)
  hv_only          NLIFeaturizer.features on --text-col against an NLI hypothesis pool
  tabular_tfidf    tabular + tfidf
  tabular_hv       tabular + HV
  tabular_tfidf_hv all three

Every featurizer is fit on TRAIN ONLY and applied to test (no leakage). The split is temporal
(`--split temporal --date-col ...`, last --test-frac by date) or random-stratified
(`--split random`). The head is a HistGradientBoostingClassifier (or LogisticRegression via
`--head logreg`). No LLM is called — HV scoring is the cached NLI encoder only.

CFPB has no built-in expert pool, so the HV pool comes from --pool-json (a JSON list of
hypothesis strings). If a known dataset key is given via --dataset-key it falls back to
hvexp.hypotheses.expert_pool. If neither yields a pool, the hv_* configs are skipped (printed
note) rather than crashing.

Usage:
    uv run python experiments/scripts/run_text_tabular.py --config experiments/configs/runs/cfpb.yaml
    uv run python experiments/scripts/run_text_tabular.py --csv data/cfpb.csv \
        --text-col narrative --cat-cols product issue state --label-col disputed \
        --split temporal --date-col date_received --test-frac 0.2 \
        --pool-json experiments/pools/cfpb_lm.json --head hgb
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # experiments/
from hvexp import hypotheses, metrics, repro  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"
CONFIGS = ["tabular_only", "tfidf_only", "hv_only",
           "tabular_tfidf", "tabular_hv", "tabular_tfidf_hv"]


def load_pool(pool_json: pathlib.Path | None, dataset_key: str | None) -> tuple[list[str], str]:
    """(pool texts, pool_id) or ([], '') when no HV pool is available."""
    if pool_json:
        pool = json.loads(pathlib.Path(pool_json).read_text())
        if not (isinstance(pool, list) and all(isinstance(h, str) for h in pool)):
            raise ValueError(f"--pool-json {pool_json} must be a JSON list of strings")
        return pool, pathlib.Path(pool_json).stem
    if dataset_key and dataset_key in hypotheses.EXPERT_POOLS:
        pool, _tags = hypotheses.expert_pool(dataset_key)
        return pool, "expert"
    return [], ""


def split_indices(df: pd.DataFrame, *, split: str, date_col: str | None,
                  test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test row-index arrays for a temporal (last test_frac by date) or random split."""
    n = len(df)
    n_test = max(1, int(round(test_frac * n)))
    if split == "temporal":
        if not date_col:
            raise ValueError("--split temporal requires --date-col")
        order = np.argsort(pd.to_datetime(df[date_col]).to_numpy(), kind="stable")
        return order[:-n_test], order[-n_test:]
    if split == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return perm[n_test:], perm[:n_test]
    raise ValueError(f"unknown --split {split!r}")


def make_head(kind: str, seed: int):
    """The shared classifier head applied to every feature configuration."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    raise ValueError(f"unknown --head {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--cat-cols", nargs="+", default=[])
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--head", default="hgb", choices=["hgb", "logreg"])
    ap.add_argument("--svd-dims", type=int, default=128)
    ap.add_argument("--pool-json", type=pathlib.Path, default=None)
    ap.add_argument("--dataset-key", default=None, help="fall back to hvexp expert_pool if known")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-mode", default="entail_contradict")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)
    if not args.csv:
        ap.error("--csv (or a config with `csv:`) is required")

    dataset_id = args.dataset_key or pathlib.Path(args.csv).stem
    run_id = args.run_id or f"tt_{dataset_id}_{args.split}_{repro.git_commit()}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)
    text = df[args.text_col].fillna("").astype(str).to_numpy()
    y_all = LabelEncoder().fit_transform(df[args.label_col].to_numpy())
    n_classes = int(len(np.unique(y_all)))
    tr, te = split_indices(df, split=args.split, date_col=args.date_col,
                           test_frac=args.test_frac, seed=args.seed)
    print(f"[split] {args.split}: n_train={len(tr)} n_test={len(te)} classes={n_classes}")

    pool, pool_id = load_pool(args.pool_json, args.dataset_key)
    use_hv = bool(pool)
    if not use_hv:
        print("[hv] no pool (--pool-json / known --dataset-key) -> skipping hv_* configs")

    # ---- fit each featurizer on TRAIN ONLY, then transform train + test ------------------------
    comps_tr: dict[str, np.ndarray] = {}
    comps_te: dict[str, np.ndarray] = {}

    if args.cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat = df[args.cat_cols].astype(str).to_numpy()
        comps_tr["tabular"] = ohe.fit_transform(cat[tr])
        comps_te["tabular"] = ohe.transform(cat[te])
    else:
        print("[tabular] no --cat-cols -> tabular_* configs will be empty/skipped")

    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr_tfidf = tfidf.fit_transform(text[tr])
    n_svd = max(2, min(args.svd_dims, Xtr_tfidf.shape[1] - 1))  # SVD rank < vocab size
    svd = TruncatedSVD(n_components=n_svd, random_state=args.seed)
    comps_tr["tfidf"] = svd.fit_transform(Xtr_tfidf)
    comps_te["tfidf"] = svd.transform(tfidf.transform(text[te]))

    if use_hv:
        fz = NLIFeaturizer(encoder=args.encoder, device=args.device, verbose=True)
        t0 = time.time()
        print(f"[nli] warming cache: {len(text)} texts x {len(pool)} hyps on {args.encoder} ...")
        fz.features(list(text), pool, score_mode=args.score_mode)  # one pass over train+test
        print(f"[nli] warm in {time.time()-t0:.0f}s  cost={fz.cost_summary()}")
        comps_tr["hv"] = fz.features(list(text[tr]), pool, score_mode=args.score_mode)
        comps_te["hv"] = fz.features(list(text[te]), pool, score_mode=args.score_mode)

    # ---- which feature keys does each named configuration stack? -------------------------------
    recipe = {
        "tabular_only": ["tabular"],
        "tfidf_only": ["tfidf"],
        "hv_only": ["hv"],
        "tabular_tfidf": ["tabular", "tfidf"],
        "tabular_hv": ["tabular", "hv"],
        "tabular_tfidf_hv": ["tabular", "tfidf", "hv"],
    }

    repro.Manifest(
        run_id=run_id,
        config={"csv": str(args.csv), "text_col": args.text_col, "cat_cols": args.cat_cols,
                "label_col": args.label_col, "split": args.split, "date_col": args.date_col,
                "test_frac": args.test_frac, "head": args.head, "svd_dims": args.svd_dims,
                "encoder": args.encoder, "score_mode": args.score_mode, "pool_id": pool_id,
                "pool_size": len(pool), "seed": args.seed, "n_classes": n_classes},
        seed=args.seed, dataset=dataset_id, encoder=args.encoder,
        pool_id=pool_id, extra={"pool": pool},
    ).write(run_dir)

    rows: list[dict] = []
    with results_path.open("w") as fh:
        for config in CONFIGS:
            keys = [k for k in recipe[config] if k in comps_tr]
            if not keys:  # e.g. hv_only with no pool, or tabular_only with no cat cols
                print(f"  {config:<18} skipped (no available feature blocks)")
                continue
            Xtr = np.hstack([comps_tr[k] for k in keys])
            Xte = np.hstack([comps_te[k] for k in keys])
            t = time.time()
            try:
                head = make_head(args.head, args.seed)
                head.fit(Xtr, y_all[tr])
                proba = head.predict_proba(Xte)
                pred = proba.argmax(axis=1)
                m = metrics.compute_metrics(y_all[te], pred, proba, n_classes=n_classes)
                roc = (float(roc_auc_score(y_all[te], proba[:, 1]))
                       if n_classes == 2 else float("nan"))
                err = None
            except Exception as e:  # keep the sweep going; record the failure
                m, roc, err = {}, float("nan"), f"{type(e).__name__}: {e}"
            row = {"config": config, "blocks": keys, "roc_auc": roc,
                   "n_train": int(len(tr)), "n_test": int(len(te)),
                   "n_features": int(Xtr.shape[1]), "fit_seconds": round(time.time() - t, 3),
                   "error": err, **m}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            tag = err or (f"acc={m.get('accuracy', float('nan')):.3f} "
                          f"f1={m.get('macro_f1', float('nan')):.3f} auc={roc:.3f}")
            print(f"  {config:<18} nfeat={Xtr.shape[1]:<5} {tag}")

    # ---- small markdown table of the configurations -------------------------------------------
    lines = ["| config | n_features | accuracy | macro_f1 | roc_auc |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['n_features']} | "
                     f"{r.get('accuracy', float('nan')):.4f} | "
                     f"{r.get('macro_f1', float('nan')):.4f} | {r['roc_auc']:.4f} |")
    table = "\n".join(lines)
    (run_dir / "summary.md").write_text(table + "\n")
    print(f"\n{table}\n")
    print(f"[done] {len(rows)} rows -> {results_path}")


if __name__ == "__main__":
    main()
