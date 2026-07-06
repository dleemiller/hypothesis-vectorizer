#!/usr/bin/env python
"""Aggregate learning-curve results (JSONL) into paper-ready tables with confidence intervals.

Reads one or more results.jsonl files, aggregates over seeds per (system, shots) into
mean + bootstrap 95% CI for accuracy and macro-F1, and writes:
  * processed tidy CSV (one row per system×shots with mean/lo/hi)
  * a markdown table (accuracy) for the paper
  * a LaTeX table (accuracy)

Usage:
    uv run python experiments/scripts/summarize_results.py experiments/results/raw/lc_trec_baselines
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import metrics  # noqa: E402

PROCESSED = pathlib.Path(__file__).resolve().parents[1] / "results" / "processed"
TABLES = pathlib.Path(__file__).resolve().parents[1] / "results" / "tables"


def load_rows(paths: list[pathlib.Path]) -> list[dict]:
    rows = []
    for p in paths:
        jl = p / "results.jsonl" if p.is_dir() else p
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _shot_key(s):
    return 10**9 if s == "all" else int(s)


def aggregate(rows: list[dict], metric: str = "accuracy") -> dict:
    """(system, shots) -> (mean, lo, hi, n) over seeds."""
    buckets = defaultdict(list)
    for r in rows:
        if r.get("error") or metric not in r:
            continue
        buckets[(r["system"], r["shots"])].append(r[metric])
    agg = {}
    for key, vals in buckets.items():
        mean, lo, hi = metrics.bootstrap_ci(np.array(vals))
        agg[key] = (mean, lo, hi, len(vals))
    return agg


def _systems_and_shots(agg):
    systems = sorted({s for s, _ in agg})
    shots = sorted({k for _, k in agg}, key=_shot_key)
    return systems, shots


def write_processed_csv(agg: dict, out: pathlib.Path, metric: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["system,shots,mean,lo,hi,n"]
    for (sysname, shots), (mean, lo, hi, n) in sorted(agg.items(), key=lambda kv: (kv[0][0], _shot_key(kv[0][1]))):
        lines.append(f"{sysname},{shots},{mean:.4f},{lo:.4f},{hi:.4f},{n}")
    out.write_text("\n".join(lines) + "\n")


def markdown_table(agg: dict, metric: str) -> str:
    systems, shots = _systems_and_shots(agg)
    head = "| system | " + " | ".join(str(s) for s in shots) + " |"
    sep = "|" + "---|" * (len(shots) + 1)
    out = [f"### {metric} (mean over seeds; **bold** = best per column)", "", head, sep]
    best = {sh: max((agg[(sy, sh)][0] for sy in systems if (sy, sh) in agg), default=None) for sh in shots}
    for sy in systems:
        cells = []
        for sh in shots:
            if (sy, sh) in agg:
                m = agg[(sy, sh)][0]
                cells.append(f"**{m:.3f}**" if best[sh] is not None and abs(m - best[sh]) < 1e-9 else f"{m:.3f}")
            else:
                cells.append("—")
        out.append(f"| {sy} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def latex_table(agg: dict, metric: str) -> str:
    systems, shots = _systems_and_shots(agg)
    cols = "l" + "r" * len(shots)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "system & " + " & ".join(str(s) for s in shots) + r" \\", r"\midrule"]
    for sy in systems:
        cells = [f"{agg[(sy, sh)][0]:.3f}" if (sy, sh) in agg else "--" for sh in shots]
        lines.append(sy.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--name", default="summary")
    ap.add_argument("--metrics", nargs="+", default=["accuracy", "macro_f1"])
    args = ap.parse_args()

    rows = load_rows(args.runs)
    print(f"[summarize] {len(rows)} rows from {len(args.runs)} run(s)")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        agg = aggregate(rows, metric)
        write_processed_csv(agg, PROCESSED / f"{args.name}_{metric}.csv", metric)
        md = markdown_table(agg, metric)
        (TABLES / f"{args.name}_{metric}.md").write_text(md + "\n")
        (TABLES / f"{args.name}_{metric}.tex").write_text(latex_table(agg, metric) + "\n")
        print("\n" + md + "\n")
    print(f"[summarize] wrote tables to {TABLES} and processed CSVs to {PROCESSED}")


if __name__ == "__main__":
    main()
