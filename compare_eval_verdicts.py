#!/usr/bin/env python3
"""
compare_eval_verdicts.py

Compare verdict distributions between two evaluation JSONL files (e.g. vanilla vs trained).
Produces a verdict transition matrix showing how verdicts changed between the two runs,
plus a special analysis for rows where the baseline model output was "I don't know".

Usage:
    python compare_eval_verdicts.py \\
        --baseline /path/to/vanilla/evaluation_details.jsonl \\
        --trained  /path/to/trained/evaluation_details.jsonl \\
        [--baseline-label "Vanilla"] \\
        [--trained-label  "Perception-R1"] \\
        [--idk-string "I don't know"] \\
        [--output report.txt]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> dict[int, dict]:
    """Load a JSONL file and return a dict keyed by 'index'."""
    records = {}
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {lineno} in {path.name}: {exc}",
                      file=sys.stderr)
                continue
            idx = row.get("index")
            if idx is None:
                print(f"[WARN] Line {lineno} has no 'index' field – skipping.", file=sys.stderr)
                continue
            records[idx] = row
    return records


def fmt(count: int, total: int) -> str:
    """Return 'count (pct%)' string."""
    pct = (count / total * 100) if total > 0 else 0.0
    return f"{count} ({pct:.1f}%)"


def separator(width: int = 80, char: str = "─") -> str:
    return char * width


def print_transition_matrix(
    transitions: dict[str, dict[str, int]],
    baseline_verdicts: list[str],
    trained_verdicts: list[str],
    baseline_label: str,
    trained_label: str,
    title: str,
    out,
) -> None:
    """Pretty-print a transition matrix to `out`."""

    # Column width for cell content
    CELL_W = 18
    LABEL_W = 14

    def cell(text: str) -> str:
        return str(text).center(CELL_W)

    def label(text: str) -> str:
        return str(text).ljust(LABEL_W)

    lines = []
    lines.append(separator())
    lines.append(f"  {title}")
    lines.append(separator())
    lines.append(
        f"  Rows show {baseline_label} verdict → columns show {trained_label} verdict"
    )
    lines.append(
        "  Each cell: count (% of that baseline-verdict group)"
    )
    lines.append("")

    # Header row
    header = label(f"{baseline_label} ↓") + "".join(cell(v) for v in trained_verdicts) + cell("ROW TOTAL")
    lines.append(header)
    lines.append(separator())

    grand_total = sum(
        count
        for row_dict in transitions.values()
        for count in row_dict.values()
    )

    col_totals: dict[str, int] = defaultdict(int)

    for bv in baseline_verdicts:
        row_dict = transitions.get(bv, {})
        row_total = sum(row_dict.values())
        row_str = label(bv)
        for tv in trained_verdicts:
            cnt = row_dict.get(tv, 0)
            col_totals[tv] += cnt
            row_str += cell(fmt(cnt, row_total))
        row_str += cell(fmt(row_total, grand_total))
        lines.append(row_str)

    lines.append(separator())

    # Column totals row
    col_row = label("COL TOTAL")
    for tv in trained_verdicts:
        col_row += cell(fmt(col_totals[tv], grand_total))
    col_row += cell(fmt(grand_total, grand_total))
    lines.append(col_row)
    lines.append(separator())
    lines.append(f"  Grand total matched rows: {grand_total}")
    lines.append("")

    text = "\n".join(lines)
    print(text, file=out)


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse(
    baseline_path: Path,
    trained_path: Path,
    baseline_label: str,
    trained_label: str,
    idk_string: str,
    out,
) -> None:
    print(f"Loading baseline : {baseline_path}", file=sys.stderr)
    baseline = load_jsonl(baseline_path)
    print(f"Loading trained  : {trained_path}", file=sys.stderr)
    trained = load_jsonl(trained_path)

    # Find common indices
    common_indices = sorted(set(baseline.keys()) & set(trained.keys()))
    only_baseline = set(baseline.keys()) - set(trained.keys())
    only_trained  = set(trained.keys())  - set(baseline.keys())

    print(separator(), file=out)
    print("  EVALUATION COMPARISON REPORT", file=out)
    print(separator(), file=out)
    print(f"  Baseline : {baseline_path}", file=out)
    print(f"  Trained  : {trained_path}", file=out)
    print(f"  Baseline rows   : {len(baseline):,}", file=out)
    print(f"  Trained rows    : {len(trained):,}", file=out)
    print(f"  Matched rows    : {len(common_indices):,}", file=out)
    if only_baseline:
        print(f"  Only in baseline: {len(only_baseline):,}  (excluded from analysis)", file=out)
    if only_trained:
        print(f"  Only in trained : {len(only_trained):,}  (excluded from analysis)", file=out)
    print("", file=out)

    # ── Build transition matrix ───────────────────────────────────────────────
    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    idk_transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    idk_rows_in_baseline = 0

    for idx in common_indices:
        b_row = baseline[idx]
        t_row = trained[idx]

        bv = b_row.get("verdict", "MISSING")
        tv = t_row.get("verdict", "MISSING")
        transitions[bv][tv] += 1

        # Special "I don't know" case
        raw = (b_row.get("model_raw_output") or "").strip()
        if raw == idk_string.strip():
            idk_rows_in_baseline += 1
            idk_transitions[bv][tv] += 1

    # Collect all unique verdict labels (sorted for stable output)
    _VERDICT_ORDER = ["correct", "incorrect", "no_format", "abstention", "MISSING"]

    all_bv = set(transitions.keys())
    all_tv = set(v for d in transitions.values() for v in d.keys())
    all_verdicts = sorted(all_bv | all_tv, key=lambda x: (_VERDICT_ORDER.index(x) if x in _VERDICT_ORDER else 99, x))

    # ── Overall transition matrix ─────────────────────────────────────────────
    print_transition_matrix(
        transitions=transitions,
        baseline_verdicts=all_verdicts,
        trained_verdicts=all_verdicts,
        baseline_label=baseline_label,
        trained_label=trained_label,
        title="OVERALL VERDICT TRANSITION MATRIX",
        out=out,
    )

    # ── "I don't know" special case ───────────────────────────────────────────
    print(separator(), file=out)
    print(f'  SPECIAL CASE: Baseline model_raw_output == "{idk_string}"', file=out)
    print(separator(), file=out)
    print(
        f"  Total matched rows where baseline output is exactly \"{idk_string}\": "
        f"{idk_rows_in_baseline:,}",
        file=out,
    )
    print("", file=out)

    if idk_rows_in_baseline == 0:
        print('  No rows found with that exact raw output string.', file=out)
        print("", file=out)
    else:
        idk_all_bv = set(idk_transitions.keys())
        idk_all_tv = set(v for d in idk_transitions.values() for v in d.keys())
        idk_verdicts = sorted(
            idk_all_bv | idk_all_tv,
            key=lambda x: (_VERDICT_ORDER.index(x) if x in _VERDICT_ORDER else 99, x),
        )

        print_transition_matrix(
            transitions=idk_transitions,
            baseline_verdicts=idk_all_bv,
            trained_verdicts=idk_verdicts,
            baseline_label=f'{baseline_label} (IDK)',
            trained_label=trained_label,
            title=f'TRANSITION MATRIX FOR "I don\'t know" ROWS',
            out=out,
        )

        # Also print a flat summary for easy reading
        print(separator(char="·"), file=out)
        print(
            f'  Flat summary — what did the trained model predict for the {idk_rows_in_baseline} '
            f'"I don\'t know" rows?',
            file=out,
        )
        total_idk = idk_rows_in_baseline
        combined: dict[str, int] = defaultdict(int)
        for row_dict in idk_transitions.values():
            for tv, cnt in row_dict.items():
                combined[tv] += cnt
        for tv in idk_verdicts:
            cnt = combined.get(tv, 0)
            print(f"    {tv:<14} : {fmt(cnt, total_idk)}", file=out)
        print("", file=out)

    print(separator(), file=out)
    print("  END OF REPORT", file=out)
    print(separator(), file=out)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare verdict transitions between two evaluation JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        metavar="JSONL",
        help="Path to the baseline evaluation_details.jsonl (e.g. vanilla zeroshot).",
    )
    parser.add_argument(
        "--trained",
        required=True,
        type=Path,
        metavar="JSONL",
        help="Path to the trained model evaluation_details.jsonl.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Vanilla",
        metavar="LABEL",
        help="Human-readable label for the baseline model.",
    )
    parser.add_argument(
        "--trained-label",
        default="Trained",
        metavar="LABEL",
        help="Human-readable label for the trained model.",
    )
    parser.add_argument(
        "--idk-string",
        default="I don't know",
        metavar="STRING",
        help="Exact string to match in model_raw_output for the special-case analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write report to this file instead of stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.baseline.exists():
        sys.exit(f"[ERROR] Baseline file not found: {args.baseline}")
    if not args.trained.exists():
        sys.exit(f"[ERROR] Trained file not found: {args.trained}")

    if args.output:
        with args.output.open("w") as fh:
            analyse(
                baseline_path=args.baseline,
                trained_path=args.trained,
                baseline_label=args.baseline_label,
                trained_label=args.trained_label,
                idk_string=args.idk_string,
                out=fh,
            )
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        analyse(
            baseline_path=args.baseline,
            trained_path=args.trained,
            baseline_label=args.baseline_label,
            trained_label=args.trained_label,
            idk_string=args.idk_string,
            out=sys.stdout,
        )


if __name__ == "__main__":
    main()
