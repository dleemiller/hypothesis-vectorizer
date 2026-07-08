"""hvexp — paper-experiment helpers for hypothesis-vectorizer.

Importable utilities used by the scripts in experiments/scripts/. Keeps the paper's
experiment harness (learning curves, baselines, ablations, text+tabular) separate from the
installable `hypothesis_vectorizer` library while reusing its encoder + cache.

Import from a script with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # experiments/
    from hvexp import datasets, systems, metrics, repro
"""
