"""Reproducibility: git provenance, environment versions, run manifests, split persistence.

Every result the paper reports must be traceable to a git commit, config, seed, dataset
version, and the exact train/test indices used. `write_manifest` records all of that in one
JSON file per run directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def git_commit(repo: Path | None = None) -> str:
    """Current git HEAD sha (short), with a '-dirty' suffix if the tree has changes."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", str(repo), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "scipy", "sklearn", "torch", "transformers", "sentence_transformers", "datasets"]:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "absent"
    try:
        import torch

        out["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        out["cuda"] = "?"
    return out


def config_hash(obj: Any) -> str:
    """Stable short hash of a JSON-serializable config, for run identity."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Manifest:
    """Provenance record written to every run directory as manifest.json."""

    run_id: str
    config: dict[str, Any]
    seed: int
    dataset: str
    encoder: str = ""
    pool_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, run_dir: Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "config": self.config,
            "config_hash": config_hash(self.config),
            "seed": self.seed,
            "dataset": self.dataset,
            "encoder": self.encoder,
            "pool_id": self.pool_id,
            "libraries": library_versions(),
            **self.extra,
        }
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


def save_split(run_dir: Path, *, train_idx: np.ndarray, test_idx: np.ndarray,
               val_idx: np.ndarray | None = None) -> Path:
    """Persist the exact index arrays used, so a result never silently depends on RNG drift."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": np.asarray(train_idx).tolist(),
        "test_idx": np.asarray(test_idx).tolist(),
        "val_idx": None if val_idx is None else np.asarray(val_idx).tolist(),
    }
    path = run_dir / "split_indices.json"
    path.write_text(json.dumps(payload))
    return path
