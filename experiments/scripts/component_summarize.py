"""Component summarization — interpretable pruning by naming behavioral axes.

Idea (NOTES 2026-07-09): instead of selecting a keeper SUBSET, decompose the cached score matrix of
an abundant pool into its dominant axes (PLS = supervised/label-aligned by default; PCA = variance)
and ask the LLM to write ONE contrastive hypothesis naming each axis's +pole vs -pole. M hypotheses
-> ~effective-rank clean contrastive summaries. Then VERIFY each summary actually recovers its axis
(correlate the new hypothesis's entailment vector with the component score) — naming a component is
asking the LLM to invert the encoder, which is not guaranteed to succeed, so we measure it.

Stages: (1) decompose + build pole prompts [CPU]; (2) LLM naming [network, no GPU]; (3) --verify
re-scores the named hypotheses [GPU]. Run without --verify under a GPU hold to preview the summaries.

    uv run python experiments/scripts/component_summarize.py --run trec_full --k 8 --method pls
    uv run python experiments/scripts/component_summarize.py --run trec_full --k 8 --verify   # GPU
"""

import argparse
import json
import os

import numpy as np


class NameContrast:  # dspy.Signature, defined at runtime to keep import cheap
    pass


def _decompose(Xs, y, n_classes, k, method):
    """(loadings (k,m), per-component var/importance). pls=supervised (label-aligned), pca=variance,
    fa=varimax-rotated factor analysis (rotates toward SPARSE loadings -> each factor loads on few
    hypotheses -> more nameable; unsupervised like pca)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA, FactorAnalysis

    if method == "pls":
        Y = np.eye(n_classes)[y]
        pls = PLSRegression(n_components=k).fit(Xs, Y)
        loadings = pls.x_loadings_.T  # (k, m)
        total = Xs.shape[1]  # standardized -> total variance == n_features; normalize to a fraction
        var = [float(np.var(Xs @ pls.x_weights_[:, c]) / total) for c in range(k)]
    elif method == "fa":
        fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=7).fit(Xs)
        loadings = fa.components_  # (k, m)
        ss = (loadings**2).sum(axis=1)  # varimax has no natural order; sort by SS-loadings
        order = np.argsort(ss)[::-1]
        loadings, ss = loadings[order], ss[order]
        var = (ss / ss.sum()).tolist()
    else:
        pca = PCA(n_components=k).fit(Xs)
        loadings = pca.components_
        var = pca.explained_variance_ratio_.tolist()
    return loadings, var


def _poles(load_vec, pool, top=4):
    order = np.argsort(load_vec)
    pos = [pool[i] for i in order[::-1][:top]]
    neg = [pool[i] for i in order[:top]]
    return pos, neg


def _name_components(components, task):
    import dspy

    sig = dspy.Signature(
        "task, pole_high: list[str], pole_low: list[str] -> hypothesis: str",
        "Two groups of NLI hypotheses form the opposite ends of ONE behavioral axis a text "
        "classifier uses: `pole_high` fire together on some texts, `pole_low` on the others. Write "
        "ONE new declarative sentence about 'the text' whose ENTAILMENT is HIGH for the pole_high "
        "texts and LOW for the pole_low texts — i.e. state the DISTINCTION itself as a contrast "
        "('The text asks for X rather than Y'). Single sentence, verifiable from the text alone.",
    )
    lm = dspy.LM(model="openrouter/deepseek/deepseek-v4-flash", max_tokens=8000, temperature=0.7)
    predict = dspy.Predict(sig)
    named = []
    with dspy.context(lm=lm):
        for c in components:
            try:
                out = predict(task=task, pole_high=c["pos"], pole_low=c["neg"])
                named.append((out.hypothesis or "").strip())
            except Exception as e:
                print(f"  naming comp {c['idx']} failed: {type(e).__name__}", flush=True)
                named.append("")
    return named


def _rehypothesize(components, named, task):
    """Second pass: show the LLM the poles + its own first draft, ask for a SHARPER single
    hypothesis and (if the contrast is really two properties) an atomic split. Inspection only."""
    import dspy

    sig = dspy.Signature(
        "task, pole_high: list[str], pole_low: list[str], draft: str -> "
        "refined: str, atomic_split: list[str]",
        "`draft` is a first attempt at ONE hypothesis separating the pole_high texts from the "
        "pole_low texts for a classifier. Improve it: `refined` = the single sharpest declarative "
        "sentence about 'the text' capturing that contrast (fix vagueness, keep it verifiable from "
        "the text alone, prefer the answer-anticipation the encoder can infer). `atomic_split` = if "
        "the contrast is really TWO independent properties, the 1-2 standalone affirmative "
        "hypotheses it splits into; else an empty list.",
    )
    lm = dspy.LM(model="openrouter/deepseek/deepseek-v4-flash", max_tokens=8000, temperature=0.7)
    predict = dspy.Predict(sig)
    out = []
    with dspy.context(lm=lm):
        for c, d in zip(components, named):
            if not d:
                out.append((d, []))
                continue
            try:
                r = predict(task=task, pole_high=c["pos"], pole_low=c["neg"], draft=d)
                out.append(
                    ((r.refined or "").strip(), [s.strip() for s in (r.atomic_split or []) if s.strip()])
                )
            except Exception as e:
                print(f"  refine comp {c['idx']} failed: {type(e).__name__}", flush=True)
                out.append((d, []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="trec_full", help="run dir whose pool+scores to decompose")
    ap.add_argument("--data", default="trec")
    ap.add_argument("--k", type=int, default=8, help="number of components to summarize")
    ap.add_argument("--method", choices=["pls", "pca", "fa"], default="pls")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--refine", action="store_true", help="second LLM pass: sharpen + atomic split")
    ap.add_argument("--poles", type=int, default=1, help="how many +/- pole hyps to print per component")
    ap.add_argument("--verify", action="store_true", help="re-score named hyps on GPU + correlate")
    ap.add_argument("--cache", default="cache/nli_scores.sqlite")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("APIKEY") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["APIKEY"]

    from sklearn.preprocessing import StandardScaler

    from hypothesis_vectorizer.cache import ScoreCache
    from hypothesis_vectorizer.config import DataConfig, EncoderConfig
    from hypothesis_vectorizer.costs import CostTracker
    from hypothesis_vectorizer.encoder import EntailmentScorer
    from hypothesis_vectorizer.train.data import load

    pool = json.load(open(f"runs/{args.run}/model.json"))["hypotheses"]
    bundle = load(DataConfig(name=args.data, train_size=5452, val_size=0, test_size=2000), seed=7)
    scorer = EntailmentScorer(
        EncoderConfig(model=args.encoder, device="cuda"), ScoreCache(args.cache), CostTracker()
    )
    m = len(pool)
    X = scorer.features(bundle.train_texts, pool)[:, :m]  # entail columns (cached)
    if scorer.costs.encoder_gpu_pairs:
        print(f"WARNING: {scorer.costs.encoder_gpu_pairs} GPU pairs — pool not fully cached", flush=True)
    Xs = StandardScaler().fit_transform(X)

    loadings, var = _decompose(Xs, bundle.y_train, bundle.n_classes, args.k, args.method)
    components = []
    for c in range(args.k):
        pos, neg = _poles(loadings[c], pool)
        components.append({"idx": c, "var": var[c], "pos": pos, "neg": neg})

    print(f"=== {args.method.upper()} on {args.run}: {m} hyps -> naming {args.k} components ===\n")
    named = _name_components(components, bundle.task)
    refined = _rehypothesize(components, named, bundle.task) if args.refine else None
    for i, (c, h) in enumerate(zip(components, named)):
        print(f"[comp {c['idx'] + 1}  var {c['var']:.1%}]")
        for p in c["pos"][: args.poles]:
            print(f"  +pole: {p[:78]}")
        for p in c["neg"][: args.poles]:
            print(f"  -pole: {p[:78]}")
        print(f"  NAMED:   {h}")
        if refined is not None:
            ref, split = refined[i]
            print(f"  REFINED: {ref}")
            if split:
                print(f"  SPLIT:   {' | '.join(split)}")
        print()

    if args.verify:
        newX = scorer.features(bundle.train_texts, [h for h in named if h])[:, : sum(bool(h) for h in named)]
        print("=== verify: |corr| of each named hyp's entailment vs its component score ===")
        j = 0
        for c, h in zip(components, named):
            if not h:
                continue
            comp_score = Xs @ loadings[c["idx"]]
            v = newX[:, j]
            j += 1
            r = abs(np.corrcoef(v, comp_score)[0, 1]) if np.std(v) > 1e-9 else 0.0
            flag = "OK" if r > 0.6 else "WEAK — LLM couldn't invert this axis"
            print(f"  comp {c['idx'] + 1}: |corr|={r:.2f}  {flag}")
        print(f"\nGPU pairs scored (new hyps only): {scorer.costs.encoder_gpu_pairs}")


if __name__ == "__main__":
    main()
