"""Rare Case Sampling by PDM Scores.

This script identifies rare / hard driving cases from PDM evaluation results
and generates NAVSIM-compatible YAML split files for targeted fine-tuning or
analysis. Three failure modes are extracted:

  1. **Low ego-progress** - bottom percentile of ego-progress among otherwise
     safe frames (no collision & on-road).
  2. **Collision** - frames where at-fault collision occurred.
  3. **Off-road** - frames where the vehicle left the drivable area.

Usage
-----
    python tools/rare_case_sampling_by_pdms.py \
        --pdm-result  work_dirs/path/to/<MODEL_NAME>/test/navtrain.csv \
        --base-split  configs/navsim_splits/navtrain_split/navtrain.yaml \
        --output-dir  configs/navsim_splits/navtrain_split/<MODEL_NAME> \
        [--ep-percentile 1]
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import yaml


def load_pdm_result(path: str) -> pd.DataFrame:
    """Load a PDM evaluation CSV and return a DataFrame."""
    df = pd.read_csv(path)
    required_cols = {"token", "no_at_fault_collisions",
                     "drivable_area_compliance", "ego_progress"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"PDM result CSV is missing columns: {missing}")
    return df


def load_navsim_split(path: str) -> dict:
    """Load a NAVSIM scene-filter YAML split file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_navsim_split(split: dict, path: str) -> None:
    """Save a NAVSIM scene-filter YAML split file."""
    with open(path, "w") as f:
        yaml.safe_dump(split, f)


def sample_low_ego_progress(df: pd.DataFrame, percentile: float = 1.0):
    """Return tokens with ego-progress in the bottom *percentile* among safe frames."""
    safe_mask = ((df["no_at_fault_collisions"] == 1) &
                 (df["drivable_area_compliance"] == 1))
    valid_mask = safe_mask & (df["ego_progress"] > 0.0)
    threshold = np.percentile(df["ego_progress"][valid_mask], percentile)
    tokens = df["token"][valid_mask & (df["ego_progress"] < threshold)].tolist()
    return tokens, threshold


def sample_collision(df: pd.DataFrame):
    """Return tokens where at-fault collision occurred."""
    mask = df["no_at_fault_collisions"] == 0
    return df["token"][mask].tolist()


def sample_off_road(df: pd.DataFrame):
    """Return tokens that left the drivable area."""
    mask = df["drivable_area_compliance"] == 0
    return df["token"][mask].tolist()


def make_split(base_split: dict, tokens: list) -> dict:
    """Create a new split dict with the given tokens."""
    new_split = copy.deepcopy(base_split)
    new_split["tokens"] = tokens
    return new_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rare case sampling by PDM scores.")
    parser.add_argument(
        "--pdm-result", required=True,
        help="Path to PDM evaluation CSV (must contain 'token', "
             "'no_at_fault_collisions', 'drivable_area_compliance', "
             "'ego_progress' columns).")
    parser.add_argument(
        "--base-split", required=True,
        help="Path to base NAVSIM scene-filter YAML split file.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write the output YAML split files.")
    parser.add_argument(
        "--ep-percentile", type=float, default=1.0,
        help="Percentile threshold for low ego-progress sampling "
             "(default: 1.0, i.e. bottom 1%%).")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load inputs -------------------------------------------------------
    df = load_pdm_result(args.pdm_result)
    base_split = load_navsim_split(args.base_split)
    prefix = os.path.basename(args.base_split).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    n_base = len(base_split.get("tokens", []))

    def _pct(n):
        return f"{n / n_base * 100:.2f}%" if n_base else "N/A"

    # ---- 1. Low ego-progress -----------------------------------------------
    ep_tokens, ep_thresh = sample_low_ego_progress(df, args.ep_percentile)
    print(f"[ego-progress < {args.ep_percentile}%ile]  "
          f"threshold={ep_thresh:.4f}  "
          f"tokens={len(ep_tokens)} ({_pct(len(ep_tokens))})")
    save_navsim_split(
        make_split(base_split, ep_tokens),
        os.path.join(args.output_dir, f"{prefix}_ep_{args.ep_percentile}pct.yaml"))

    # ---- 2. Collision ------------------------------------------------------
    col_tokens = sample_collision(df)
    print(f"[collision]  tokens={len(col_tokens)} ({_pct(len(col_tokens))})")
    save_navsim_split(
        make_split(base_split, col_tokens),
        os.path.join(args.output_dir, f"{prefix}_collision.yaml"))

    # ---- 3. Off-road -------------------------------------------------------
    offroad_tokens = sample_off_road(df)
    print(f"[off-road]   tokens={len(offroad_tokens)} ({_pct(len(offroad_tokens))})")
    save_navsim_split(
        make_split(base_split, offroad_tokens),
        os.path.join(args.output_dir, f"{prefix}_off_road.yaml"))

    print(f"\nAll splits written to: {args.output_dir}")


if __name__ == "__main__":
    main()
