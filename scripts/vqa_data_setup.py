#!/usr/bin/env python3
"""
Sample reproducible train / validation splits from the VQAv2 validation parquet.

Steps:
  1. Load the full VQAv2 validation parquet.
  2. Keep only rows where answer_type == 'other'.
  3. De-duplicate by image_id (keep first occurrence).
  4. Sample `n_train` rows for training   (seeded).
  5. From the remainder, sample `n_val` rows for validation (seeded).
  6. Save both splits as parquet files.

Usage:
  python3 vqa_data_setup.py                         # uses defaults
  python3 vqa_data_setup.py --seed 123              # custom seed
  python3 vqa_data_setup.py --n_train 1000 --n_val 100
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val splits from VQAv2 validation data"
    )
    parser.add_argument(
        "--input_parquet", type=str,
        default="/home/kalashkala/Datasets/VQAv2/lmms-lab_VQAv2_default_validation.parquet",
        help="Path to the full VQAv2 validation parquet",
    )
    parser.add_argument(
        "--output_train", type=str,
        default="/home/kalashkala/Datasets/VQAv2/vqa_train.parquet",
        help="Output path for the training split",
    )
    parser.add_argument(
        "--output_val", type=str,
        default="/home/kalashkala/Datasets/VQAv2/vqa_validation.parquet",
        help="Output path for the validation split",
    )
    parser.add_argument("--n_train", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=200, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"Loading {args.input_parquet} ...")
    df = pd.read_parquet(args.input_parquet)
    print(f"  Total rows: {len(df)}")

    # ── 2. Filter to answer_type == 'other' ──────────────────────────────
    df_other = df.loc[df["answer_type"] == "other"]
    print(f"  Rows with answer_type == 'other': {len(df_other)}")

    # ── 3. De-duplicate by image_id ──────────────────────────────────────
    df_unique = df_other.drop_duplicates(subset="image_id", keep="first")
    print(f"  Unique images: {len(df_unique)}")

    # ── 4. Sample train split (seeded) ───────────────────────────────────
    df_train = df_unique.sample(n=args.n_train, random_state=args.seed)
    df_train.to_parquet(args.output_train)
    print(f"  ✅ Train split ({len(df_train)} rows) → {args.output_train}")

    # ── 5. Sample val split from remainder (seeded) ──────────────────────
    df_remaining = df_unique.drop(df_train.index)
    df_val = df_remaining.sample(n=args.n_val, random_state=args.seed)
    df_val.to_parquet(args.output_val)
    print(f"  ✅ Val   split ({len(df_val)} rows) → {args.output_val}")

    print(f"\nDone. Seed used: {args.seed}")


if __name__ == "__main__":
    main()