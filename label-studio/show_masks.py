#!/usr/bin/env python3
"""
Debug script: load JSONL, decode COCO RLE masks, overlay them on images.
Saves one PNG per sample.

Usage:
    python show_masks.py \
        --jsonl path/to/dataset.jsonl \
        --images-root path/to/images \
        --output-dir path/to/output \
        [--max-samples 10] \
        [--alpha 0.45]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask


LABEL_MAP: Dict[int, str] = {1: "rotor", 2: "stator", 3: "shaft"}

LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "rotor":  (91,  155, 255),   # #5B9BFF
    "stator": (232, 89,  106),   # #E8596A
    "shaft":  (126, 211, 33),    # #7ED321
}

DEFAULT_COLOR = (200, 200, 200)


# -- JSONL I/O ---------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON error line {i}: {e}")
    return records


# -- Image lookup ------------------------------------------------------------
def find_image(sample_record: Dict[str, Any], images_root: Path) -> Optional[Path]:
    sample = sample_record.get("sample", sample_record)

    raw = None
    assets = sample.get("assets", [])
    if assets and isinstance(assets, list):
        asset = assets[0]
        if isinstance(asset, dict):
            raw = (
                asset.get("uri")
                or asset.get("metadata", {}).get("original_file_name")
            )

    raw = (
        raw
        or sample.get("images", {}).get("query", {}).get("path")
        or sample.get("image_path")
        or sample.get("image")
        or sample.get("file_path")
        or sample.get("metadata", {}).get("original_file_name")
    )

    if not raw:
        return None

    p = Path(raw)
    for candidate in [
        p if p.is_absolute() else None,
        images_root / p,
        images_root / p.name,
    ]:
        if candidate and candidate.exists():
            return candidate

    matches = list(images_root.rglob(p.name))
    return matches[0] if matches else None


# -- Mask decoding ------------------------------------------------------------
def decode_coco_rle(counts: str, size: List[int]) -> np.ndarray:
    """ Returns binary mask (H, W) with values 0/1. """
    return coco_mask.decode({"counts": counts, "size": size}).astype(np.uint8)


# -- Visualisation ------------------------------------------------------------
def overlay_masks(
    image: Image.Image,
    instances: List[Dict[str, Any]],
    alpha: float = 0.45,
) -> Tuple[Image.Image, List[str]]:
    img_w, img_h = image.size
    base = image.convert("RGBA")
    debug_msgs = []

    for idx, inst in enumerate(instances):
        label_name = inst.get("label_name")
        if not label_name:
            label_id = inst.get("label_id")
            label_name = LABEL_MAP.get(label_id, f"unknown_{label_id}")

        color = LABEL_COLORS.get(label_name, DEFAULT_COLOR)
        mask_data = inst.get("mask")

        if mask_data is None:
            debug_msgs.append(f"  instance {idx} ({label_name}): NO MASK - skipped")
            continue

        counts = mask_data.get("counts")
        rle_size = mask_data.get("size")

        if not counts or not rle_size:
            debug_msgs.append(f"  instance {idx} ({label_name}): invalid RLE - skipped")
            continue

        rle_h, rle_w = rle_size[0], rle_size[1]
        debug_msgs.append(
            f"  instance {idx} ({label_name}): "
            f"RLE size=[H={rle_h}, W={rle_w}], image size=(W={img_w}, H={img_h})"
        )

        if rle_h != img_h or rle_w != img_w:
            debug_msgs.append(
                f"    DIMENSION MISMATCH: RLE({rle_h}×{rle_w}) != image({img_h}×{img_w})"
            )

        binary = decode_coco_rle(counts, rle_size)
        area = int(binary.sum())
        debug_msgs.append(f"    mask area: {area} px  ({100*area/(img_h*img_w):.2f}% of image)")

        if area == 0:
            debug_msgs.append("    EMPTY MASK - zero foreground pixels")
            continue

        r, g, b = color
        alpha_val = int(alpha * 255)
        rgba_mask = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        fg = binary.astype(bool)
        rgba_mask[fg] = [r, g, b, alpha_val]

        overlay = Image.fromarray(rgba_mask, mode="RGBA")
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB"), debug_msgs


def draw_legend(image: Image.Image, labels_present: List[str]) -> Image.Image:
    """ Append a small legend at the bottom of the image. """
    draw    = ImageDraw.Draw(image)
    x, y    = 10, image.height - 30 * len(labels_present) - 10
    box_size = 20

    for label in labels_present:
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        draw.rectangle([x, y, x + box_size, y + box_size], fill=color)
        draw.text((x + box_size + 6, y + 2), label, fill=(255, 255, 255))
        y += 28

    return image


# -- Main ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SAM2 RLE masks from JSONL over images.")
    parser.add_argument("--jsonl",       type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, default=Path("mask_debug"))
    parser.add_argument("--max-samples", type=int,  default=20,
                        help="Max number of samples to process (0 = all)")
    parser.add_argument("--alpha",       type=float, default=0.45,
                        help="Mask overlay opacity (0–1)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.jsonl)
    print(f"Read {len(records)} records from {args.jsonl}")

    processed = 0
    for rec in records:
        if rec.get("record_type") != "sample":
            continue

        sample = rec.get("sample", {})
        assets = sample.get("assets", [])
        if not assets:
            continue

        asset = assets[0]
        instances = asset.get("annotations", [])

        has_masks = any(inst.get("mask") is not None for inst in instances)
        if not has_masks:
            continue

        sample_id = sample.get("sample_id", "?")
        image_path = find_image(rec, args.images_root)

        if image_path is None:
            print(f"[sample {sample_id}]: image not found - skipped")
            continue

        print(f"\n[sample {sample_id}] {image_path.name}")
        img = Image.open(image_path).convert("RGB")
        print(f"  PIL image size: W={img.width}, H={img.height}")

        composited, msgs = overlay_masks(img, instances, alpha=args.alpha)
        for m in msgs:
            print(m)

        labels = []
        for inst in instances:
            if inst.get("mask") is None:
                continue
            label_name = inst.get("label_name")
            if not label_name:
                label_name = LABEL_MAP.get(inst.get("label_id"), "?")
            labels.append(label_name)

        composited = draw_legend(composited, list(dict.fromkeys(labels)))

        out_path = args.output_dir / f"sample_{sample_id}_{image_path.stem}_debug.png"
        composited.save(out_path)
        print(f"  -> saved: {out_path}")

        processed += 1
        if args.max_samples and processed >= args.max_samples:
            print(f"\nReached --max-samples={args.max_samples}, stopping.")
            break

    print(f"\nDone. {processed} samples visualized -> {args.output_dir}/")


if __name__ == "__main__":
    main()