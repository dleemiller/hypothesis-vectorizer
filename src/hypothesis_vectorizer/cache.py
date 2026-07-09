"""Persistent sqlite cache of raw NLI logits keyed by (text, hypothesis, model) hashes.

Raw logits are stored so any downstream representation can be derived without
re-scoring. Writes are chunk-committed by the scorer, so an interrupted run
loses at most one chunk of GPU work. Access is serialized behind a lock:
sqlite objects are thread-affine, and this cache has been shared with worker
threads before — cheap insurance against a known crash class.
"""

import os
import sqlite3
import threading
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nli_scores (
  text_hash TEXT NOT NULL,
  hyp_hash  TEXT NOT NULL,
  model     TEXT NOT NULL,
  z_e REAL NOT NULL, z_n REAL NOT NULL, z_c REAL NOT NULL,
  PRIMARY KEY (text_hash, hyp_hash, model)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hypotheses (hyp_hash TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS texts      (text_hash TEXT PRIMARY KEY, text TEXT NOT NULL);
"""

_CHUNK = 500  # sqlite bound-variable limit safety


def _wal_supported(directory: Path) -> bool:
    """Whether WAL actually works on this filesystem, probed on a throwaway file.

    ZFS and some network filesystems accept `PRAGMA journal_mode=WAL` yet fail with
    'disk I/O error' on the first real write — and that error poisons both the connection and
    the half-written main file. So we never attempt WAL on the real cache: we probe a temp file
    (with an actual write), then open the real DB directly in the supported mode.

    The probe filename is PER-PROCESS (pid): concurrent runs sharing a cache dir must not unlink
    each other's probe mid-write, which would spuriously fail the survivor into DELETE mode.
    """
    probe = directory / f".wal_probe.{os.getpid()}.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(str(probe) + suffix).unlink(missing_ok=True)
    conn = None
    ok = False
    try:
        conn = sqlite3.connect(probe)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("CREATE TABLE IF NOT EXISTS p(a TEXT PRIMARY KEY) WITHOUT ROWID;")
        conn.execute("INSERT OR REPLACE INTO p VALUES('x')")  # the write that fails under bad WAL
        conn.commit()
        ok = True
    except sqlite3.OperationalError:
        ok = False
    finally:
        if conn is not None:
            conn.close()  # must close even on failure — a leaked poisoned handle breaks later opens
        for suffix in ("", "-wal", "-shm"):
            Path(str(probe) + suffix).unlink(missing_ok=True)
    return ok


class ScoreCache:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        if str(path) == ":memory:":  # ephemeral in-process cache (HypothesisVectorizer default)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.journal_mode = "memory"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self.journal_mode = "WAL" if _wal_supported(path.parent) else "DELETE"
            self._conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_logits(self, model: str, hyp_hash: str, text_hashes: list[str]) -> dict[str, np.ndarray]:
        """{text_hash: logits(3,)} for the cached subset of text_hashes."""
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(text_hashes), _CHUNK):
                chunk = text_hashes[i : i + _CHUNK]
                q = (
                    "SELECT text_hash, z_e, z_n, z_c FROM nli_scores "
                    f"WHERE hyp_hash=? AND model=? AND text_hash IN ({','.join('?' * len(chunk))})"
                )
                for th, z_e, z_n, z_c in self._conn.execute(q, [hyp_hash, model, *chunk]):
                    out[th] = np.array([z_e, z_n, z_c], dtype=np.float32)
        return out

    def put_logits(
        self, model: str, hyp_hash: str, hypothesis: str, rows: list[tuple[str, str, np.ndarray]]
    ) -> None:
        """rows: [(text_hash, text, logits(3,)), ...] — committed immediately."""
        if not rows:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO hypotheses (hyp_hash, text) VALUES (?, ?)", (hyp_hash, hypothesis)
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO texts (text_hash, text) VALUES (?, ?)",
                [(th, text) for th, text, _ in rows],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO nli_scores (text_hash, hyp_hash, model, z_e, z_n, z_c) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(th, hyp_hash, model, float(z[0]), float(z[1]), float(z[2])) for th, _, z in rows],
            )
            self._conn.commit()
