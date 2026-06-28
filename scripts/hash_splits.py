#!/usr/bin/env python3
"""Hash local split IDs without redistributing raw data."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def hash_ids(ids: list[str]) -> str:
    joined = "\n".join(map(str, sorted(ids))) + "\n"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def ids_from_npz(path: Path) -> list[str]:
    data = np.load(path, allow_pickle=True)
    if "ids" not in data:
        raise KeyError(f"{path} has no 'ids' array")
    return [str(x) for x in data["ids"]]


def ids_from_parquet(path: Path) -> list[str]:
    df = pd.read_parquet(path, columns=["id"])
    return [str(x) for x in df["id"].tolist()]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/hash_splits.py <data/slurp_real>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    candidates = [
        root / "features_train.npz",
        root / "features_devel.npz",
        root / "features_test.npz",
        root / "slurp_asr.parquet",
    ]
    found = False
    for path in candidates:
        if not path.exists():
            continue
        found = True
        ids = ids_from_npz(path) if path.suffix == ".npz" else ids_from_parquet(path)
        print(f"{path.name}\tn={len(ids)}\tsha256(sorted_ids)={hash_ids(ids)}")
    if not found:
        print(f"No supported split files found under {root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
