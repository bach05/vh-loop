from __future__ import annotations

import logging
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.data.canonical_schema.sample.base import DataSample
from scripts.utils.bbox import bbox_xyxy, match_boxes
from scripts.utils.io import extract_bbox_annotations, resolve_image_path


# ---------------------------------------------------------------------------
# Low-level drawing primitive
# ---------------------------------------------------------------------------

def draw_box(
    ax,
    bbox_px: tuple[float, float, float, float],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2,
    fontsize: int = 6,
    text_color: str = "white",
) -> None:
    """ Draw a single bounding-box rectangle with a text label on *ax*. """
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = bbox_px
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        max(0, y1),
        label,
        color=text_color,
        fontsize=fontsize,
        bbox={
            "facecolor": color,
            "alpha": 0.7,
            "edgecolor": "none",
            "boxstyle": "Round, pad=0.2",
        },
    )


# ---------------------------------------------------------------------------
# Per-sample grid (GT + one column per model)
# ---------------------------------------------------------------------------

def render_sample_grid(
    *,
    sample_id: str,
    gt_sample: DataSample,
    pred_samples_by_model: dict[str, dict[str, DataSample]],
    thresholds_by_model: dict[str, float],
    class_aware: bool,
    image_root: Optional[str | Path],
    out_path: Path,
) -> None:
    """
    Render a side-by-side GT / prediction comparison image for one sample.

    Columns:
      - Column 0: ground-truth boxes (black).
      - Columns 1…N: per-model predictions colour-coded as
        TP (limegreen), FP (tomato), FN (lightskyblue, dashed).

    The image is saved to *out_path* at 200 dpi.
    """
    image_path = resolve_image_path(gt_sample, image_root)
    if image_path is None or not image_path.exists():
        logging.warning(
            f"Cannot visualize sample {sample_id}: image not found: {image_path}"
        )
        return

    img = Image.open(image_path).convert("RGB")
    gt_anns = extract_bbox_annotations(gt_sample)
    model_names = list(pred_samples_by_model.keys())
    n_cols = 1 + len(model_names)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # --- GT column ---
    ax = axes[0]
    ax.imshow(img)
    ax.set_title(f"GT | sample {sample_id}")
    ax.axis("off")
    for gt in gt_anns:
        draw_box(ax, bbox_xyxy(gt), label=f"GT {gt.label_name}", color="black")

    # --- Model columns ---
    for ax, model_name in zip(axes[1:], model_names):
        pred_sample = pred_samples_by_model[model_name].get(sample_id)
        pred_anns = extract_bbox_annotations(pred_sample)
        thr = thresholds_by_model[model_name]

        matches, unmatched_gt, _ = match_boxes(
            gt_anns, pred_anns, threshold=thr, class_aware=class_aware
        )
        match_by_pred = {int(m["pred_idx"]): m for m in matches}

        ax.imshow(img)
        ax.set_title(f"{model_name} | IoU thr={thr:g}")
        ax.axis("off")

        for pi, pred in enumerate(pred_anns):
            if pi in match_by_pred:
                m = match_by_pred[pi]
                label = f"TP {pred.label_name} | IoU={float(m['iou']):.2f}"
                color = "limegreen"
            else:
                label = f"FP {pred.label_name}"
                color = "tomato"
            draw_box(ax, bbox_xyxy(pred), label=label, color=color)

        for gi in unmatched_gt:
            gt = gt_anns[gi]
            draw_box(
                ax,
                bbox_xyxy(gt),
                label=f"FN {gt.label_name}",
                color="lightskyblue",
                linestyle=":",
                linewidth=1.0,
                text_color="darkslategrey",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------

def _to_dataframe(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def plot_metric_by_threshold(
    data: pd.DataFrame | list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
) -> None:
    """ Line plot of *metric* vs IoU threshold, one line per model. """
    df = _to_dataframe(data)
    if df.empty or metric not in df.columns or "model" not in df.columns or "threshold" not in df.columns:
        return

    plot_df = df.copy()
    plot_df["threshold"] = pd.to_numeric(plot_df["threshold"], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=["model", "threshold", metric])
    if plot_df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for model_name, group in sorted(plot_df.groupby("model"), key=lambda x: str(x[0])):
        group = group.sort_values("threshold", kind="mergesort")
        plt.plot(
            group["threshold"].to_numpy(),
            group[metric].to_numpy(),
            marker="o",
            label=str(model_name),
        )

    plt.xlabel("IoU threshold")
    plt.ylabel(metric)
    plt.title(f"{metric} by IoU threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_bar(
    data: pd.DataFrame | list[dict[str, Any]],
    *,
    metric: str | None = None,
    value_col: str | None = None,
    out_path: Path,
) -> None:
    """
    Bar chart of a single scalar metric across all models.
    Preserves input order of rows/models.
    """
    col = metric if metric is not None else value_col
    if col is None:
        raise ValueError("plot_summary_bar requires either metric= or value_col=")
    if metric is not None and value_col is not None and metric != value_col:
        raise ValueError("metric and value_col refer to different columns")

    df = _to_dataframe(data)
    if df.empty or "model" not in df.columns or col not in df.columns:
        return

    plot_df = df.copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["model", col])

    if plot_df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"].astype(str).to_list(), plot_df[col].to_list())
    plt.ylabel(col)
    plt.title(col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()