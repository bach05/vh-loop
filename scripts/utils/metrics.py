from __future__ import annotations

import numpy as np
from tqdm import tqdm
from typing import Any

from scripts.data.canonical_schema.sample.base import DataSample
from scripts.utils.bbox import match_boxes
from scripts.utils.schema_helpers import extract_bbox_annotations


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    """ Return num/den, or 0.0 when den == 0. """
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_prediction_file(
    *,
    model_name: str,
    gt_samples: dict[str, DataSample],
    pred_samples: dict[str, DataSample],
    thresholds: list[float],
    class_aware: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute detection/localisation metrics for one prediction JSONL.

    Two mIoU variants are computed once (threshold-independent) and then
    attached to every per-threshold row:

    * ``miou_gt_penalized``       – total matched IoU / total GT boxes
      (unmatched GT boxes contribute 0, penalising missed detections).
    * ``mean_iou_positive_matches`` – mean IoU over matched pairs only
      (threshold=1e-9 to accept all overlapping pairs).

    Args:
        model_name:   Display name used in output rows.
        gt_samples:   Ground-truth samples keyed by sample_id.
        pred_samples: Predicted samples keyed by sample_id.
        thresholds:   IoU thresholds at which to evaluate P/R/F1.
        class_aware:  Whether class labels must agree for a match.

    Returns:
        rows:    One dict per threshold with keys
                 ``model``, ``threshold``, ``tp``, ``fp``, ``fn``,
                 ``precision``, ``recall``, ``f1``, ``mean_iou_tp``,
                 ``miou_gt_penalized``, ``mean_iou_positive_matches``.
        summary: Best-F1 row plus the two mIoU scalars.
    """
    sample_ids = sorted(set(gt_samples.keys()) | set(pred_samples.keys()))

    # --- Threshold-independent mIoU pass ---
    total_gt = 0
    total_iou_penalized = 0.0
    positive_ious: list[float] = []

    for sid in tqdm(sample_ids, desc=f"mIoU {model_name}"):
        gt_anns = extract_bbox_annotations(gt_samples.get(sid))
        pred_anns = extract_bbox_annotations(pred_samples.get(sid))
        total_gt += len(gt_anns)

        matches, _, _ = match_boxes(
            gt_anns, pred_anns, threshold=1e-9, class_aware=class_aware
        )
        total_iou_penalized += sum(float(m["iou"]) for m in matches)
        positive_ious.extend(float(m["iou"]) for m in matches)

    miou_gt_penalized = safe_div(total_iou_penalized, total_gt)
    mean_iou_positive = float(np.mean(positive_ious)) if positive_ious else 0.0

    # --- Per-threshold P/R/F1 ---
    rows: list[dict[str, Any]] = []

    for thr in thresholds:
        tp = fp = fn = 0
        matched_ious: list[float] = []

        for sid in sample_ids:
            gt_anns = extract_bbox_annotations(gt_samples.get(sid))
            pred_anns = extract_bbox_annotations(pred_samples.get(sid))

            matches, unmatched_gt, unmatched_pred = match_boxes(
                gt_anns, pred_anns, threshold=thr, class_aware=class_aware
            )
            tp += len(matches)
            fp += len(unmatched_pred)
            fn += len(unmatched_gt)
            matched_ious.extend(float(m["iou"]) for m in matches)

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
                "mean_iou_positive_matches": mean_iou_positive,
            }
        )

    best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    summary = {
        "model": model_name,
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "miou_gt_penalized": miou_gt_penalized,
        "mean_iou_positive_matches": mean_iou_positive,
    }

    return rows, summary