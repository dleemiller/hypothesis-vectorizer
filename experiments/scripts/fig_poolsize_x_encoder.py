#!/usr/bin/env python
"""Pool-size x encoder-size: can more hypotheses substitute for encoder capacity?

Overlays the accuracy-vs-pool-size curve at each finecat size on ONE axis (reusing the existing
-l run abl_trec_poolsize + the new abl_trec_poolsize_{xs,s,m}). Read: if a big pool on a small
encoder catches a small pool on a big encoder, hypotheses substitute for capacity; if the small-
encoder curve plateaus below the big-encoder curve, they do not. Second panel = logloss (calibration)
vs pool size per encoder size. Pure plotting from cached ablation results.

    uv run python experiments/scripts/fig_poolsize_x_encoder.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"
FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
# run-id -> (size label, color); -l reuses the pre-existing abl_trec_poolsize
RUNS = [
    ("abl_trec_poolsize_xs", "-xs", "#d6604d"),
    ("abl_trec_poolsize_s", "-s", "#f4a582"),
    ("abl_trec_poolsize_m", "-m", "#4393c3"),
    ("abl_trec_poolsize", "-l", "#2166ac"),
]


def agg(run_id: str):
    """pool_size -> (mean_acc, mean_logloss) over seeds, or None if the run is missing."""
    path = RAW / run_id / "results.jsonl"
    if not path.exists():
        return None
    acc, ll = collections.defaultdict(list), collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("axis") == "pool_size" and r.get("error") is None:
            n = int(r["variant"])
            acc[n].append(r["accuracy"])
            if r.get("log_loss") is not None:
                ll[n].append(r["log_loss"])
    return {n: (statistics.mean(acc[n]), statistics.mean(ll[n]) if ll[n] else None)
            for n in sorted(acc)}


def main() -> None:
    curves = [(lbl, col, agg(rid)) for rid, lbl, col in RUNS]
    have = [(lbl, col, c) for lbl, col, c in curves if c]
    missing = [lbl for (rid, lbl, _), (_, _, c) in zip(RUNS, curves) if not c]
    if missing:
        print(f"[warn] missing runs (skipped): {missing}", file=sys.stderr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for lbl, col, c in have:
        ns = sorted(c)
        ax1.plot(ns, [c[n][0] for n in ns], "o-", color=col, lw=2, label=f"finecat{lbl}")
        ll = [(n, c[n][1]) for n in ns if c[n][1] is not None]
        if ll:
            ax2.plot([n for n, _ in ll], [v for _, v in ll], "o-", color=col, lw=2, label=f"finecat{lbl}")
    for ax, ylab, ttl in [(ax1, "accuracy", "accuracy vs pool size (per encoder size)"),
                          (ax2, "log loss (lower = better calibrated)", "calibration vs pool size")]:
        ax.set_xscale("log"); ax.set_xlabel("pool size (# hypotheses)")
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=10)
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8)
    ax1.set_xticks([8, 16, 32, 64, 128, 256]); ax1.set_xticklabels([8, 16, 32, 64, 128, 256])
    ax2.set_xticks([8, 16, 32, 64, 128, 256]); ax2.set_xticklabels([8, 16, 32, 64, 128, 256])
    fig.suptitle("TREC: do more hypotheses substitute for encoder capacity? "
                 "(pool trec_gen256, train 20/class, 5 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"poolsize_x_encoder_trec.{ext}", dpi=160)
    plt.close(fig)
    print(f"[figure] poolsize_x_encoder_trec.(png|pdf)")

    # readout: does a big pool on -s reach a small pool on -l?
    print("\n===== accuracy: pool size x encoder size =====")
    sizes = [lbl for lbl, _, _ in have]
    allns = sorted({n for _, _, c in have for n in c})
    print(f"{'n_hyp':>6}" + "".join(f"{s:>8}" for s in sizes))
    for n in allns:
        print(f"{n:>6}" + "".join(f"{c[n][0]:>8.3f}" if n in c else f"{'-':>8}" for _, _, c in have))


if __name__ == "__main__":
    main()
