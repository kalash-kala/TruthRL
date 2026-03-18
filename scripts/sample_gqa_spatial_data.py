import argparse
import pandas as pd
import re

def filter_gqa_open_spatial(df_gqa_data, verbose=True):
    """
    Filter GQA dataframe for open-ended spatial reasoning questions.

    Expected GQA columns:
        - question
        - answer
        - types
        - semantic
        - semanticStr

    Returns:
        Filtered pandas DataFrame
    """

    SPATIAL_RELATIONS = [
        "left", "right", "above", "below", "under", "underneath",
        "behind", "in front of", "front", "inside", "on", "on top of",
        "next to", "near", "beside", "between", "over"
    ]

    YES_NO_ANSWERS = {"yes", "no", "true", "false"}

    def is_open_query(types_col):
        if isinstance(types_col, dict):
            return str(types_col.get("structural", "")).lower() == "query"
        return False

    def has_spatial_relate(semantic_col):
        """
        Checks whether semantic program contains a spatial relation step.
        Works for list, tuple, numpy object arrays, etc.
        """
        try:
            for step in semantic_col:
                if isinstance(step, dict):
                    op = str(step.get("operation", "")).lower()
                    arg = str(step.get("argument", "")).lower()

                    if op == "relate":
                        # Example: 'blanket,under,s (-)'
                        if any(rel in arg for rel in SPATIAL_RELATIONS):
                            return True
            return False
        except Exception:
            return False

    def question_looks_spatial(q):
        q = str(q).lower()
        return bool(re.search(
            r"where|which side|left|right|above|below|under|behind|"
            r"in front|inside|on top|next to|near|between|beside|over",
            q
        ))

    df = df_gqa_data.copy()

    # 1. Open-ended only
    mask_open = df["types"].apply(is_open_query)

    # 2. Spatial semantics
    mask_sem = df["semantic"].apply(has_spatial_relate)

    # 3. Remove binary-answer rows
    mask_nonbinary = ~df["answer"].astype(str).str.strip().str.lower().isin(YES_NO_ANSWERS)

    # 4. Text-based spatial cue
    mask_qtext = df["question"].apply(question_looks_spatial)

    # Final filter
    df_filtered = df[mask_open & (mask_sem | mask_qtext) & mask_nonbinary].copy()
    df_filtered = df_filtered.reset_index(drop=True)

    if verbose:
        print("Total rows:", len(df))
        print("Open-ended query rows:", int(mask_open.sum()))
        print("Spatial semantic rows:", int(mask_sem.sum()))
        print("Spatial text rows:", int(mask_qtext.sum()))
        print("Non-binary answer rows:", int(mask_nonbinary.sum()))
        print("Final filtered rows:", len(df_filtered))

    return df_filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--input_path", type=str, default="/home/kalashkala/Datasets/GQA/val_instructions.parquet")
    parser.add_argument("--output_path", type=str, default="/home/kalashkala/Datasets/GQA/val_spatial_instructions_1k.parquet")
    args = parser.parse_args()

    df_gqa_data = pd.read_parquet(args.input_path)
    df_filtered = filter_gqa_open_spatial(df_gqa_data)

    sample_df = df_filtered.sample(n=args.n_samples, random_state=args.seed)

    sample_df.to_parquet(args.output_path)