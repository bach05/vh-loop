from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

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

def plot_metric_by_threshold(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
) -> None:
    """ Line plot of *metric* vs IoU threshold, one line per model. """
    models = sorted({r["model"] for r in rows})

    plt.figure(figsize=(10, 6))
    for model in tqdm(models, desc=f"Plotting {metric}", total=len(models)):
        model_rows = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["threshold"],
        )
        plt.plot(
            [r["threshold"] for r in model_rows],
            [r[metric] for r in model_rows],
            marker="o",
            label=model,
        )

    plt.xlabel("IoU threshold")
    plt.ylabel(metric)
    plt.title(f"{metric} by IoU threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_bar(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
) -> None:
    """ Bar chart of a single scalar *metric* across all models. """
    models = [r["model"] for r in summary_rows]
    values = [r[metric] for r in summary_rows]

    plt.figure(figsize=(10, 6))
    plt.bar(models, values)
    plt.ylabel(metric)
    plt.title(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()