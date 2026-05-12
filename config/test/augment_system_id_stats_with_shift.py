#!/usr/bin/env python3
"""Augment system_id_stats.csv with shift/top from oc20dense_mapping.pkl.

The original CSV doesn't include slab shift/top. This script writes a new CSV
(default: system_id_stats_with_shift.csv) that adds two columns:
- shift (float)
- top (bool)

We derive shift/top per system_id by taking the most common (mode) among all
matching entries in oc20dense_mapping.pkl.

Usage:
  python config/test/augment_system_id_stats_with_shift.py \
    --csv config/test/system_id_stats.csv \
    --mapping data/external/oc20dense_mapping.pkl \
    --out config/test/system_id_stats_with_shift.csv
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def _clean_keys(d: dict) -> dict:
    # Some mapping files have keys with stray whitespace/newlines.
    return {str(k).strip(): v for k, v in d.items()}


def build_sid_to_shift_top(
    mapping_path: Path,
    target_system_ids: set[str] | None = None,
) -> dict[str, tuple[float | None, bool | None]]:
    with mapping_path.open("rb") as f:
        mapping = pickle.load(f)

    shifts: dict[str, Counter] = defaultdict(Counter)
    tops: dict[str, Counter] = defaultdict(Counter)

    remaining = set(target_system_ids) if target_system_ids else None

    for _, raw in mapping.items():
        if not isinstance(raw, dict):
            continue
        v = _clean_keys(raw)
        sid = v.get("system_id")
        if not sid:
            continue
        sid = str(sid)
        if remaining is not None and sid not in remaining:
            continue
        try:
            shift = v.get("shift", None)
            if shift is not None:
                shift = float(shift)
        except Exception:
            shift = None
        top = v.get("top", None)
        if isinstance(top, str):
            top = top.strip().lower() in ("true", "1", "yes")

        if shift is not None:
            # round to stabilize float modes
            shifts[sid][round(float(shift), 6)] += 1
        if top is not None:
            tops[sid][bool(top)] += 1

        # Early exit once we found at least one entry for each target SID.
        if remaining is not None and sid in remaining:
            # consider found if we have at least one shift; top is optional
            if len(shifts.get(sid, {})) > 0:
                remaining.discard(sid)
            if not remaining:
                break

    sid_to = {}
    for sid in set(list(shifts.keys()) + list(tops.keys())):
        shift_val = None
        top_val = None
        if sid in shifts and len(shifts[sid]) > 0:
            shift_val = shifts[sid].most_common(1)[0][0]
        if sid in tops and len(tops[sid]) > 0:
            top_val = tops[sid].most_common(1)[0][0]
        sid_to[sid] = (shift_val, top_val)

    return sid_to


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="config/test/system_id_stats.csv")
    ap.add_argument(
        "--mapping",
        default="data/external/oc20dense_mapping.pkl",
    )
    ap.add_argument("--out", default="config/test/system_id_stats_with_shift.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    mapping_path = Path(args.mapping)
    out_path = Path(args.out)

    # system_id_stats.csv has NO header row; parse by position.
    df = pd.read_csv(csv_path, engine="python", header=None)
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected CSV format: expected >=6 columns, got {df.shape[1]}")

    base_cols = ["system_id", "mpid", "bulk_symbol", "ads_smiles", "miller", "num_site"]
    # Preserve any extra columns if present.
    extra_cols = [f"extra_{i}" for i in range(df.shape[1] - len(base_cols))]
    df.columns = base_cols + extra_cols

    target_sids = set(df["system_id"].astype(str).tolist())
    sid_to = build_sid_to_shift_top(mapping_path, target_system_ids=target_sids)

    shifts = []
    tops = []
    missing = 0
    for sid in df["system_id"].astype(str).tolist():
        shift, top = sid_to.get(str(sid), (None, None))
        if shift is None:
            missing += 1
        shifts.append(shift)
        tops.append(top)

    df["shift"] = shifts
    df["top"] = tops

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the same style as the original file: no header row.
    df.to_csv(out_path, index=False, header=False)
    print(f"Wrote: {out_path} (missing shift for {missing}/{len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
