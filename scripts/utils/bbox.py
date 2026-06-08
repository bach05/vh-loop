from __future__ import annotations

from typing import Any
from scripts.data.canonical_schema import InstanceAnnotation


# ---------------------------------------------------------------------------
# Coordinate / label helpers
# ---------------------------------------------------------------------------

def bbox_xyxy(ann: InstanceAnnotation) -> tuple[float, float, float, float]:
    """ Return (x1, y1, x2, y2) pixel coords for an annotation bbox. """
    if ann.bbox is None:
        raise ValueError(f"Annotation {ann.instance_id!r} has no bbox")
    return (
        float(ann.bbox.tl.x),
        float(ann.bbox.tl.y),
        float(ann.bbox.br.x),
        float(ann.bbox.br.y),
    )


def annotation_label(ann: InstanceAnnotation) -> str:
    return str(ann.label_name)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def bbox_iou_from_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """ Compute IoU between two boxes given as (x1, y1, x2, y2) tuples. """
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
    return float(inter / union) if union > 0 else 0.0


def bbox_iou(a: InstanceAnnotation, b: InstanceAnnotation) -> float:
    """ Compute IoU between two InstanceAnnotation objects. """
    return bbox_iou_from_xyxy(bbox_xyxy(a), bbox_xyxy(b))


# ---------------------------------------------------------------------------
# Greedy one-to-one matching
# ---------------------------------------------------------------------------

def match_boxes(
    gt_anns: list[InstanceAnnotation],
    pred_anns: list[InstanceAnnotation],
    *,
    threshold: float,
    class_aware: bool = True,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """
    Greedy one-to-one bbox matching by descending IoU.

    Args:
        gt_anns:     Ground-truth annotation list.
        pred_anns:   Predicted annotation list.
        threshold:   Minimum IoU required to form a match.
        class_aware: When True, only annotations with the same label are
                     eligible to be matched.

    Returns:
        matches: List of dicts with keys gt_idx, pred_idx and iou.
        unmatched_gt_indices: Indices of gt_anns with no accepted match.
        unmatched_pred_indices: Indices of pred_anns with no accepted match.
    """
    candidates: list[tuple[float, int, int]] = []

    for gi, gt in enumerate(gt_anns):
        for pi, pred in enumerate(pred_anns):
            if class_aware and annotation_label(gt) != annotation_label(pred):
                continue

            iou = bbox_iou(gt, pred)
            if iou >= threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({"gt_idx": gi, "pred_idx": pi, "iou": float(iou)})

    unmatched_gt = [i for i in range(len(gt_anns)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_anns)) if i not in used_pred]

    return matches, unmatched_gt, unmatched_pred