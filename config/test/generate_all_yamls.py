#!/usr/bin/env python3
"""Generate per-system YAML configs for ALL rows in system_id_stats.csv.

Outputs YAML files under config/test/all_systems/ (by default).

Usage:
  python config/test/generate_all_yamls.py \
    --csv config/test/system_id_stats.csv \
    --out config/test/all_systems \
    --mapping data/external/oc20dense_mapping.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


def _parse_miller(miller_str: str):
    # CSV stores like "1,0,2" or "1,1,1".
    parts = [p.strip() for p in str(miller_str).split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if len(vals) != 3:
        return None
    return tuple(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="config/test/system_id_stats.csv")
    ap.add_argument("--out", default="config/test/all_systems")
    ap.add_argument(
        "--mapping",
        default="data/external/oc20dense_mapping.pkl",
        help="Optional oc20dense_mapping.pkl path to fill missing shift/top by system_id",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # system_id_stats.csv has NO header row; parse by position.
    df = pd.read_csv(csv_path, engine="python", header=None)
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected CSV format: expected >=6 columns, got {df.shape[1]}")
    
    # Try to detect if it is the augmented version (8 cols) or original (6 cols)
    if df.shape[1] >= 8:
        base_cols = [
            "system_id",
            "mpid",
            "bulk_symbol",
            "ads_smiles",
            "miller",
            "num_site",
            "shift",
            "top",
        ]
    else:
        base_cols = ["system_id", "mpid", "bulk_symbol", "ads_smiles", "miller", "num_site"]
    
    extra_cols = [f"extra_{i}" for i in range(df.shape[1] - len(base_cols))]
    df.columns = base_cols + extra_cols

    sid_to_shift_top = None
    if args.mapping:
        mapping_path = Path(args.mapping)
        if mapping_path.exists():
            print(f"Loading mapping from {mapping_path}...")
            with mapping_path.open("rb") as f:
                mapping = pickle.load(f)
            sid_to_shift_top = {}
            for _, raw in mapping.items():
                if not isinstance(raw, dict):
                    continue
                v = {str(k).strip(): vv for k, vv in raw.items()}
                sid = v.get("system_id")
                if not sid:
                    continue
                try:
                    shift = v.get("shift", None)
                    shift = float(shift) if shift is not None else None
                except Exception:
                    shift = None
                top = v.get("top", None)
                if isinstance(top, str):
                    top = top.strip().lower() in ("true", "1", "yes")
                if sid not in sid_to_shift_top:
                    sid_to_shift_top[sid] = (shift, bool(top) if top is not None else None)
        else:
            print(f"Warning: Mapping file {mapping_path} not found. Some shift/top values may be null.")

    print(f"Processing {len(df)} rows...")

    for i, row in df.iterrows():
        system_id = str(row["system_id"])
        bulk_id = str(row["mpid"])
        bulk_symbol = str(row["bulk_symbol"])
        ads_smiles = str(row["ads_smiles"])
        miller = _parse_miller(row["miller"])
        num_site = int(row["num_site"])

        # shift/top lookup
        shift = None
        top = None
        if "shift" in df.columns:
            try:
                shift = row.get("shift", None)
                shift = float(shift) if shift == shift and shift is not None else None
            except Exception:
                shift = None
        if "top" in df.columns:
            top = row.get("top", None)
            if isinstance(top, str):
                top = top.strip().lower() in ("true", "1", "yes")
            elif top is not None:
                top = bool(top)
        
        if shift is None and sid_to_shift_top is not None:
            shift, top2 = sid_to_shift_top.get(system_id, (None, None))
            if top is None:
                top = top2

        # Filename format: e.g. 000_34_928_12.yaml
        safe_system_id = system_id.replace("/", "_")
        fname = f"{i:03d}_{safe_system_id}.yaml"
        fpath = out_dir / fname

        lines = []
        lines.append("system_info:")
        lines.append("  system_id: null")
        lines.append(f"  num_site: {num_site}")
        lines.append(f"  bulk_id: {bulk_id}")
        lines.append(f"  bulk_symbol: {bulk_symbol}")
        lines.append(f"  ads_smiles: '{ads_smiles}'")
        if miller is not None:
            lines.append(f"  miller: ({miller[0]},{miller[1]},{miller[2]})")
        else:
            lines.append("  miller: null")
        if shift is None:
            lines.append("  shift: null")
        else:
            lines.append(f"  shift: {float(shift)}")
        if top is None:
            lines.append("  top: null")
        else:
            lines.append(f"  top: {str(bool(top)).lower()}")
        lines.append(f"  source_system_id: {system_id}")
        lines.append("")

        fpath.write_text("\n".join(lines), encoding="utf-8")

    # Write manifest
    (out_dir / "_manifest.txt").write_text(
        "\n".join([p.name for p in sorted(out_dir.glob("*.yaml")) if p.name != "_manifest.txt"]) + "\n",
        encoding="utf-8",
    )

    print(f"Successfully wrote {len(df)} system YAMLs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
