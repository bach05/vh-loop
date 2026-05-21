from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import hydra
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image

from data.canonical_schema.schema import SampleRecord, VLMSample
from scripts.core.constants import NORM_SIZE


# -------------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxItem:
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in canonical norm coords
    ann_idx: int


@dataclass(frozen=True)
class Match:
    gt_idx: int
    pred_idx: int
    iou: float


# -------------------------------------------------------------------------
# JSONL loading
# -------------------------------------------------------------------------

def load_canonical_samples(jsonl_path: str | Path) -> tuple(dict[int, VLMSample], dict[Any, Any]):
    """
    Load canonical JSONL and return samples indexed by sample_id.
    Skips the dataset_info header line.
    """

    path = Path(to_absolute_path(str(jsonl_path)))
    samples: dict[int, VLMSample] = {}
    dataset_info = {}

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            record_type = record.get("record_type", "sample")

            if record_type == "dataset_info":
                dataset_info = record["info"]
                continue

            if record_type != "sample":
                raise ValueError(
                    f"Unsupported record_type='{record_type}' in {path} at line {line_idx}"
                )

            sample = SampleRecord.model_validate(record)
            samples[sample.sample_id] = sample

    return samples, dataset_info


def extract_bbox_items(sample: Optional[VLMSample]) -> list[BoxItem]:
    if sample is None or sample.target is None:
        return []

    items: list[BoxItem] = []

    for ann_idx, ann in enumerate(sample.target.instances):
        if ann.bbox is None:
            continue

        items.append(
            BoxItem(
                label=str(ann.label),
                bbox=(
                    float(ann.bbox.tl.x),
                    float(ann.bbox.tl.y),
                    float(ann.bbox.br.x),
                    float(ann.bbox.br.y),
                ),
                ann_idx=ann_idx,
            )
        )

    return items


# -------------------------------------------------------------------------
# BBox matching
# -------------------------------------------------------------------------

def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def match_boxes(
    gt_boxes: list[BoxItem],
    pred_boxes: list[BoxItem],
    *,
    threshold: float,
    class_aware: bool = True,
) -> tuple[list[Match], list[int], list[int]]:
    """
    Greedy one-to-one matching by descending IoU.

    Returns:
      matches
      unmatched_gt_indices
      unmatched_pred_indices
    """

    candidates: list[tuple[float, int, int]] = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if class_aware and gt.label != pred.label:
                continue

            iou = bbox_iou(gt.bbox, pred.bbox)
            if iou >= threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(Match(gt_idx=gi, pred_idx=pi, iou=iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]

    return matches, unmatched_gt, unmatched_pred


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def evaluate_prediction_file(
    *,
    model_name: str,
    gt_samples: dict[int, VLMSample],
    pred_samples: dict[int, VLMSample],
    thresholds: list[float],
    class_aware: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute metrics for one prediction JSONL.
    """

    sample_ids = sorted(set(gt_samples.keys()) | set(pred_samples.keys()))

    # Threshold-independent mIoU-like values.
    # miou_gt_penalized = sum IoU of best one-to-one positive matches / number of GT boxes.
    # Unmatched GT boxes contribute 0.
    total_gt_for_miou = 0
    total_iou_penalized = 0.0
    positive_match_ious: list[float] = []

    for sid in tqdm(sample_ids, total=len(sample_ids), desc="Evaluating..."):
        gt_boxes = extract_bbox_items(gt_samples.get(sid))
        pred_boxes = extract_bbox_items(pred_samples.get(sid))

        total_gt_for_miou += len(gt_boxes)

        matches, _, _ = match_boxes(
            gt_boxes,
            pred_boxes,
            threshold=1e-9,
            class_aware=class_aware,
        )

        total_iou_penalized += sum(m.iou for m in matches)
        positive_match_ious.extend([m.iou for m in matches])

    miou_gt_penalized = safe_div(total_iou_penalized, total_gt_for_miou)
    mean_iou_positive_matches = (
        float(np.mean(positive_match_ious)) if positive_match_ious else 0.0
    )

    rows: list[dict[str, Any]] = []

    for thr in thresholds:
        tp = 0
        fp = 0
        fn = 0
        matched_ious: list[float] = []

        for sid in sample_ids:
            gt_boxes = extract_bbox_items(gt_samples.get(sid))
            pred_boxes = extract_bbox_items(pred_samples.get(sid))

            matches, unmatched_gt, unmatched_pred = match_boxes(
                gt_boxes,
                pred_boxes,
                threshold=thr,
                class_aware=class_aware,
            )

            tp += len(matches)
            fp += len(unmatched_pred)
            fn += len(unmatched_gt)

            matched_ious.extend([m.iou for m in matches])

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "model": model_name,
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou_tp": float(np.mean(matched_ious)) if matched_ious else 0.0,
                "miou_gt_penalized": miou_gt_penalized,
                "mean_iou_positive_matches": mean_iou_positive_matches,
            }
        )

    best_row = max(rows, key=lambda r: (r["f1"], r["threshold"]))

    summary = {
        "model": model_name,
        "best_threshold": best_row["threshold"],
        "best_f1": best_row["f1"],
        "best_precision": best_row["precision"],
        "best_recall": best_row["recall"],
        "miou_gt_penalized": miou_gt_penalized,
        "mean_iou_positive_matches": mean_iou_positive_matches,
    }

    return rows, summary


# -------------------------------------------------------------------------
# Prediction file discovery
# -------------------------------------------------------------------------

def resolve_prediction_files(cfg: DictConfig) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []

    for item in cfg.get("predictions", []) or []:
        if isinstance(item, str):
            p = Path(to_absolute_path(item))
            found.append((p.stem, p))
        else:
            p = Path(to_absolute_path(item["path"]))
            name = item.get("name", p.stem)
            found.append((name, p))

    predictions_dir = cfg.get("predictions_dir", None)
    if predictions_dir is not None:
        pred_dir = Path(to_absolute_path(str(predictions_dir)))
        patterns = cfg.get("patterns", ["*.jsonl"])

        for pattern in patterns:
            for p in sorted(pred_dir.glob(pattern)):
                found.append((p.stem, p))

    # De-duplicate by absolute path.
    dedup: dict[Path, tuple[str, Path]] = {}
    for name, p in found:
        dedup[p.resolve()] = (name, p)

    return list(dedup.values())


# -------------------------------------------------------------------------
# CSV and plots
# -------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_by_threshold(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
) -> None:
    models = sorted({r["model"] for r in rows})

    plt.figure(figsize=(10, 6))

    for model in tqdm(models, desc=f"Plotting {metric}...", total=len(models)):
        model_rows = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["threshold"],
        )
        xs = [r["threshold"] for r in model_rows]
        ys = [r[metric] for r in model_rows]
        plt.plot(xs, ys, marker="o", label=model)

    plt.xlabel("IoU threshold")
    plt.ylabel(metric)
    plt.title(f"{metric} by IoU threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_bar(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
) -> None:
    models = [r["model"] for r in summary_rows]
    values = [r[metric] for r in summary_rows]

    plt.figure(figsize=(10, 6))
    plt.bar(models, values)
    plt.ylabel(metric)
    plt.title(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------

def resolve_image_path(sample: VLMSample, image_root: Optional[str | Path]) -> Optional[Path]:
    first_image = next(iter(sample.images.values()), None)
    if first_image is None:
        return None

    p = Path(first_image.path)

    if p.is_absolute():
        return p

    if image_root is not None:
        return Path(to_absolute_path(str(image_root))) / p

    return Path(to_absolute_path(str(p)))


def canonical_bbox_to_pixel(
    bbox: tuple[float, float, float, float],
    *,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:

    #scale = max(img_w, img_h) / 1000.0
    scale_x = img_w / NORM_SIZE
    scale_y = img_h / NORM_SIZE

    x1, y1, x2, y2 = bbox
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def draw_box(
    ax,
    bbox_px: tuple[float, float, float, float],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2,
    fontsize: int = 5,
    text_color = "white",
):
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = bbox_px
    w = x2 - x1
    h = y2 - y1

    rect = Rectangle(
        (x1, y1),
        w,
        h,
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


def render_sample_grid(
    *,
    sample_id: int,
    gt_sample: VLMSample,
    pred_samples_by_model: dict[str, dict[int, VLMSample]],
    thresholds_by_model: dict[str, float],
    class_aware: bool,
    image_root: Optional[str | Path],
    out_path: Path,
    id_to_category: dict[str, int] = None,
) -> None:
    image_path = resolve_image_path(gt_sample, image_root)
    if image_path is None or not image_path.exists():
        logging.warning(f"Cannot visualize sample {sample_id}: image not found: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    model_names = list(pred_samples_by_model.keys())
    n_cols = 1 + len(model_names)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    gt_boxes = extract_bbox_items(gt_sample)

    # GT tile
    ax = axes[0]
    ax.imshow(img)
    ax.set_title(f"GT | sample {sample_id}")
    ax.axis("off")

    for gt in gt_boxes:
        label = f"GT {gt.label}:{id_to_category[gt.label]}" if id_to_category and gt.label in id_to_category else f"GT {gt.label}"
        draw_box(
            ax,
            canonical_bbox_to_pixel(gt.bbox, img_w=img_w, img_h=img_h),
            label=label,
            color="black",
        )

    # Model tiles
    for ax, model_name in zip(axes[1:], model_names):
        pred_sample = pred_samples_by_model[model_name].get(sample_id)
        pred_boxes = extract_bbox_items(pred_sample)

        thr = thresholds_by_model[model_name]

        matches, unmatched_gt, unmatched_pred = match_boxes(
            gt_boxes,
            pred_boxes,
            threshold=thr,
            class_aware=class_aware,
        )

        match_by_pred = {m.pred_idx: m for m in matches}

        ax.imshow(img)
        ax.set_title(f"{model_name} | IoU thr={thr:g}")
        ax.axis("off")

        # TP / FP predicted boxes
        for pi, pred in enumerate(pred_boxes):
            if pi in match_by_pred:
                m = match_by_pred[pi]
                label = f"TP {pred.label}:{id_to_category[pred.label]}" if id_to_category and pred.label in id_to_category else f"TP {pred.label} | IoU={m.iou:.2f}"
                color = "limegreen"
            else:
                label = f"FP {pred.label}:{id_to_category[pred.label]}" if id_to_category and pred.label in id_to_category else f"FP {pred.label}"
                color = "tomato"

            draw_box(
                ax,
                canonical_bbox_to_pixel(pred.bbox, img_w=img_w, img_h=img_h),
                label=label,
                color=color,
                linestyle="-",
            )

        # FN GT boxes, drawn in model tile as dotted lines
        for gi in unmatched_gt:
            gt = gt_boxes[gi]
            draw_box(
                ax,
                canonical_bbox_to_pixel(gt.bbox, img_w=img_w, img_h=img_h),
                label=f"FN {gt.label}:{id_to_category[gt.label]}" if id_to_category and gt.label in id_to_category else f"FN {gt.label}",
                color="lightskyblue",
                linestyle=":",
                linewidth=1.0,
                text_color="darkslategrey"
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="compare_entrypoint")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    out_dir = Path(to_absolute_path(str(cfg.get("output_dir", "."))))
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(t) for t in cfg.get("thresholds", [0.1, 0.25, 0.5, 0.75, 0.9])]
    class_aware = bool(cfg.get("class_aware", True))

    gt_path = Path(to_absolute_path(str(cfg.gt_jsonl)))
    gt_samples, dataset_info = load_canonical_samples(gt_path)

    label_info = dataset_info["label_info"]
    category_to_id = {cat: id for id, cat in label_info.items()}
    id_to_category = label_info


    prediction_files = resolve_prediction_files(cfg)
    if not prediction_files:
        raise ValueError("No prediction JSONL files found. Use predictions or predictions_dir.")

    all_metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pred_samples_by_model: dict[str, dict[int, VLMSample]] = {}

    for model_name, pred_path in prediction_files:
        logging.info(f"Evaluating {model_name}: {pred_path}")

        pred_samples, _ = load_canonical_samples(pred_path)
        pred_samples_by_model[model_name] = pred_samples

        metric_rows, summary = evaluate_prediction_file(
            model_name=model_name,
            gt_samples=gt_samples,
            pred_samples=pred_samples,
            thresholds=thresholds,
            class_aware=class_aware,
        )

        all_metric_rows.extend(metric_rows)
        summary_rows.append(summary)

    write_csv(out_dir / "metrics_by_threshold.csv", all_metric_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    if cfg.get("plots", {}).get("enabled", True):
        plot_metric_by_threshold(
            all_metric_rows,
            metric="precision",
            out_path=out_dir / "precision_by_threshold.png",
        )
        plot_metric_by_threshold(
            all_metric_rows,
            metric="recall",
            out_path=out_dir / "recall_by_threshold.png",
        )
        plot_metric_by_threshold(
            all_metric_rows,
            metric="f1",
            out_path=out_dir / "f1_by_threshold.png",
        )
        plot_metric_by_threshold(
            all_metric_rows,
            metric="mean_iou_tp",
            out_path=out_dir / "mean_iou_tp_by_threshold.png",
        )
        plot_summary_bar(
            summary_rows,
            metric="miou_gt_penalized",
            out_path=out_dir / "miou_gt_penalized.png",
        )

    if cfg.get("visualization", {}).get("enabled", False):
        vis_cfg = cfg.visualization

        threshold_mode = vis_cfg.get("threshold_mode", "best_f1")
        fixed_threshold = float(vis_cfg.get("fixed_threshold", 0.5))

        if threshold_mode == "best_f1":
            thresholds_by_model = {
                r["model"]: float(r["best_threshold"])
                for r in summary_rows
            }
        elif threshold_mode == "fixed":
            thresholds_by_model = {
                r["model"]: fixed_threshold
                for r in summary_rows
            }
        else:
            raise ValueError(
                f"Unsupported visualization.threshold_mode='{threshold_mode}'. "
                "Use 'best_f1' or 'fixed'."
            )

        requested_sample_ids = list(vis_cfg.get("sample_ids", []) or [])
        max_samples = int(vis_cfg.get("max_samples", 25))

        if requested_sample_ids:
            sample_ids = [int(s) for s in requested_sample_ids]
        else:
            sample_ids = sorted(gt_samples.keys())[:max_samples]

        image_root = vis_cfg.get("image_root", None)

        for sid in tqdm(sample_ids, "Processing visualizations...", total=len(sample_ids)):
            gt_sample = gt_samples.get(sid)
            if gt_sample is None:
                logging.warning(f"Skipping visualization for missing GT sample_id={sid}")
                continue

            render_sample_grid(
                sample_id=sid,
                gt_sample=gt_sample,
                pred_samples_by_model=pred_samples_by_model,
                thresholds_by_model=thresholds_by_model,
                class_aware=class_aware,
                image_root=image_root,
                out_path=out_dir / "visualizations" / f"sample_{sid}.png",
                id_to_category=id_to_category
            )

    logging.info(f"Comparison complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()