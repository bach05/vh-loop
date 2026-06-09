#!/usr/bin/env python3
"""
merge_compare_metrics_by_model.py

Merge multiple compare_sample.py metrics CSV files by stacking rows, then plot
one curve per unique value in the existing `model` column.

This is intentionally schema-agnostic: it does not care whether the metrics
come from schema v1, schema v2, or any other evaluation pipeline, as long as
the CSV files share the same metric columns.

Example:
    python merge_compare_metrics_by_model.py \
        --input /path/to/schema_v1/metrics_by_threshold.csv \
        --input /path/to/schema_v2/metrics_by_threshold.csv \
        --out-dir /path/to/merged_comparison

The script writes:
    merged_metrics_by_threshold.csv
    merged_summary_best.csv
    precision_by_threshold.png
    recall_by_threshold.png
    f1_by_threshold.png
    mean_iou_tp_by_threshold.png
    miou_gt_penalized_by_threshold.png
    best_f1_summary.png
    miou_gt_penalized_summary.png
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from scripts.utils import (
    load_and_stack_csvs,
    plot_metric_by_threshold,
    plot_summary_bar
)


DEFAULT_METRICS = [
    "precision",
    "recall",
    "f1",
    "mean_iou_tp",
    "miou_gt_penalized",
    "mean_iou_positive_matches",
]


def compute_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one summary row per model.

    The best row is selected by:
        highest f1, then highest threshold.

    If the CSV has no f1 column, the first row per model is used.
    """
    rows: list[dict] = []

    for model_name, group in sorted(df.groupby("model"), key=lambda x: x[0]):
        group = group.copy()

        if "f1" in group.columns:
            group["f1"] = pd.to_numeric(group["f1"], errors="coerce")
            group["threshold"] = pd.to_numeric(group["threshold"], errors="coerce")
            best = group.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        else:
            best = group.iloc[0]

        row = {
            "model": model_name,
            "best_threshold": best.get("threshold", None),
        }

        # Add common best-row metrics if available.
        for col in [
            "precision",
            "recall",
            "f1",
            "mean_iou_tp",
            "miou_gt_penalized",
            "mean_iou_positive_matches",
            "tp",
            "fp",
            "fn",
        ]:
            if col in group.columns:
                row[f"best_{col}" if col in {"precision", "recall", "f1", "mean_iou_tp"} else col] = best.get(col, None)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stack multiple metrics_by_threshold.csv files and plot one series "
            "per unique value in the existing `model` column."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Path to a metrics_by_threshold.csv file. "
            "Use multiple --input arguments to merge multiple files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for merged CSVs and plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to plot by threshold. Defaults to common compare_sample.py metrics.",
    )

    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_stack_csvs(args.input)

    merged_path = out_dir / "merged_metrics_by_threshold.csv"
    merged.to_csv(merged_path, index=False)

    summary = compute_best_summary(merged)
    summary_path = out_dir / "merged_summary_best.csv"
    summary.to_csv(summary_path, index=False)

    for metric in args.metrics:
        plot_metric_by_threshold(
            merged,
            metric=metric,
            out_path=out_dir / f"{metric}_by_threshold.png",
        )

    # Summary plots use the one-row-per-model summary table.
    plot_summary_bar(
        summary,
        value_col="best_f1",
        out_path=out_dir / "best_f1_summary.png",
    )

    plot_summary_bar(
        summary,
        value_col="miou_gt_penalized",
        out_path=out_dir / "miou_gt_penalized_summary.png",
    )

    print(f"Wrote merged metrics: {merged_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
