#!/usr/bin/env python3
"""
debug_grounder_samples.py

Randomly sample examples from each dataset entry defined in a canonical converter
run config YAML and render the ground-truth annotations on top of the images.

This is intended as a pre-conversion debugging tool: it uses the same config
parsing, EntrySpec collection, path resolution and grounder instantiation logic
as convert_datasets_to_canonical.py.

Example:
    python debug_grounder_samples.py \
        --config merge_datasets_to_canonical.yaml \
        --out-dir /tmp/grounder_debug \
        --num-samples 8 \
        --seed 42 \
        --draw-masks

Expected grounder sample format:
    {
        "image": "relative/or/absolute/path.jpg",
        "width": int,
        "height": int,
        "caption": optional asset-level caption,
        "annotations": [
            {
                "class": str,
                "bbox": [x, y, w, h],     # pixel COCO xywh
                "points": optional list of {x, y, is_positive},
                "caption": optional instance caption,
                "mask": optional CVAT-like mask payload,
            }
        ]
    }
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required. Install it with: pip install pillow") from exc


# -----------------------------------------------------------------------------
# Import your existing converter helpers
# -----------------------------------------------------------------------------


def import_converter(module_name: str):
    """Import the converter module.

    By default this expects convert_datasets_to_canonical.py to be importable
    from the current working directory or PYTHONPATH. If you place this script
    next to convert_datasets_to_canonical.py and run it from the project root,
    this should work directly.
    """
    try:
        return __import__(module_name)
    except Exception as first_exc:
        # Fallback: add this script's folder to sys.path and try again.
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        try:
            return __import__(module_name)
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not import converter module {module_name!r}. "
                "Run from the project root, place this script next to "
                "convert_datasets_to_canonical.py, or pass --converter-module.\n"
                f"First error: {first_exc}\nSecond error: {second_exc}"
            ) from second_exc


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


def reservoir_sample(iterable: Iterable[Any], k: int, rng: random.Random) -> tuple[list[tuple[int, Any]], int]:
    """Uniformly sample up to k elements from an iterable without loading all samples.

    Returns:
        selected: list of (1-based original sample index, sample)
        total: total number of items seen
    """
    selected: list[tuple[int, Any]] = []
    total = 0

    for total, item in enumerate(iterable, start=1):
        if len(selected) < k:
            selected.append((total, item))
        else:
            j = rng.randint(1, total)
            if j <= k:
                selected[j - 1] = (total, item)

    selected.sort(key=lambda x: x[0])
    return selected, total


# -----------------------------------------------------------------------------
# Geometry and annotation helpers
# -----------------------------------------------------------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xywh_to_xyxy(bbox: Any, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    try:
        x, y, w, h = map(float, bbox)
    except Exception:
        return None

    x1 = int(round(clamp(x, 0, img_w)))
    y1 = int(round(clamp(y, 0, img_h)))
    x2 = int(round(clamp(x + w, 0, img_w)))
    y2 = int(round(clamp(y + h, 0, img_h)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def label_to_color(label: str) -> tuple[int, int, int]:
    """Deterministic vivid RGB color for a label."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox(xy, text, font=font)
    w, h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
    x, y = xy
    return x, y, x + w, y + h


def normalize_points(raw_points: Any) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    if not raw_points:
        return out

    for p in raw_points:
        try:
            if isinstance(p, dict):
                x = float(p.get("x"))
                y = float(p.get("y"))
                is_positive = bool(p.get("is_positive", True))
            else:
                x = float(p[0])
                y = float(p[1])
                is_positive = True
            out.append((int(round(x)), int(round(y)), is_positive))
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# Optional CVAT mask overlay support
# -----------------------------------------------------------------------------


def parse_rle_counts(rle: Any) -> list[int]:
    if rle is None:
        return []
    if isinstance(rle, list):
        return [int(v) for v in rle]
    if isinstance(rle, str):
        rle = rle.strip()
        if not rle:
            return []
        return [int(tok.strip()) for tok in rle.replace(";", ",").split(",") if tok.strip()]
    return []


def decode_cvat_crop_mask(mask_payload: dict[str, Any]) -> tuple[Image.Image, int, int] | None:
    """Decode a CVAT XML <mask> payload into a crop alpha mask.

    Expected payload shape, as yielded by the CVAT grounder:
        {
            "type": "cvat_rle",
            "left": int,
            "top": int,
            "width": int,
            "height": int,
            "rle": "..."
        }

    The implementation assumes CVAT's alternating background/foreground runs
    over the cropped mask rectangle, starting with background.
    """
    if not isinstance(mask_payload, dict):
        return None

    rle = mask_payload.get("rle") or mask_payload.get("counts")
    counts = parse_rle_counts(rle)
    if not counts:
        return None

    try:
        left = int(round(float(mask_payload.get("left", 0))))
        top = int(round(float(mask_payload.get("top", 0))))
        width = int(round(float(mask_payload.get("width", 0))))
        height = int(round(float(mask_payload.get("height", 0))))
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    total = width * height
    data = bytearray(total)
    idx = 0
    foreground = False  # CVAT mask RLE starts with background

    for run in counts:
        run = max(0, int(run))
        end = min(total, idx + run)
        if foreground and end > idx:
            data[idx:end] = b"\xff" * (end - idx)
        idx = end
        foreground = not foreground
        if idx >= total:
            break

    mask = Image.frombytes("L", (width, height), bytes(data))
    return mask, left, top


def paste_mask_overlay(base: Image.Image, mask_payload: Any, color: tuple[int, int, int], alpha: int) -> None:
    decoded = decode_cvat_crop_mask(mask_payload)
    if decoded is None:
        return

    mask, left, top = decoded
    overlay = Image.new("RGBA", mask.size, (*color, alpha))
    base.paste(overlay, (left, top), mask)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------


def draw_sample(
    *,
    sample: dict[str, Any],
    image_path: Path,
    output_path: Path,
    draw_masks: bool,
    draw_captions: bool,
    max_side: int | None,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGBA")
    img_w, img_h = image.size

    # Draw mask overlays before boxes/text.
    if draw_masks:
        for ann in sample.get("annotations", []) or []:
            label = str(ann.get("class", "unknown"))
            color = label_to_color(label)
            paste_mask_overlay(image, ann.get("mask"), color=color, alpha=75)

    draw = ImageDraw.Draw(image)
    line_w = max(2, int(round(max(img_w, img_h) / 700)))
    font = get_font(max(12, int(round(max(img_w, img_h) / 140))))
    small_font = get_font(max(10, int(round(max(img_w, img_h) / 180))))
    point_r = max(4, line_w * 2)

    valid_annotations = 0
    invalid_annotations = 0

    for ann_idx, ann in enumerate(sample.get("annotations", []) or [], start=1):
        label = str(ann.get("class", "unknown"))
        color = label_to_color(label)
        xyxy = xywh_to_xyxy(ann.get("bbox"), img_w, img_h)
        if xyxy is None:
            invalid_annotations += 1
            continue
        valid_annotations += 1

        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        caption = str(ann.get("caption") or "").strip()
        text = f"{ann_idx}: {label}"
        if draw_captions and caption:
            text = f"{text} | {caption}"

        tb = text_bbox(draw, (x1, y1), text, font=font)
        text_h = tb[3] - tb[1]
        text_w = tb[2] - tb[0]
        bg_y1 = max(0, y1 - text_h - 2 * line_w)
        bg_y2 = bg_y1 + text_h + 2 * line_w
        bg_x2 = min(img_w, x1 + text_w + 2 * line_w)
        draw.rectangle([x1, bg_y1, bg_x2, bg_y2], fill=(*color, 210))
        draw.text((x1 + line_w, bg_y1 + line_w), text, fill=(255, 255, 255, 255), font=font)

        for px, py, is_positive in normalize_points(ann.get("points")):
            px = int(clamp(px, 0, img_w - 1))
            py = int(clamp(py, 0, img_h - 1))
            point_color = color if is_positive else (255, 255, 255)
            draw.ellipse(
                [px - point_r, py - point_r, px + point_r, py + point_r],
                fill=(*point_color, 255),
                outline=(0, 0, 0, 255),
                width=max(1, line_w // 2),
            )

    # Asset-level caption / dataset info footer.
    asset_caption = str(sample.get("caption") or sample.get("asset_caption") or "").strip()
    footer_lines = [f"image: {sample.get('image')}", f"instances: {valid_annotations}"]
    if asset_caption:
        footer_lines.append(f"asset_description: {asset_caption}")
    footer_text = "\n".join(footer_lines)
    fb = text_bbox(draw, (0, 0), footer_text, font=small_font)
    footer_h = fb[3] - fb[1] + 2 * line_w
    draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(0, 0, 0, 170))
    draw.text((line_w, img_h - footer_h + line_w), footer_text, fill=(255, 255, 255, 255), font=small_font)

    if max_side is not None and max_side > 0:
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)

    return {
        "valid_annotations": valid_annotations,
        "invalid_annotations": invalid_annotations,
        "image_width": img_w,
        "image_height": img_h,
    }


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------


def resolve_image_for_loading(conv, sample: dict[str, Any], entry: Any, common_data_folder: str | None) -> Path:
    sample_image = str(sample.get("image", ""))

    # First use the converter's own resolution logic.
    try:
        p = Path(conv.resolve_image_absolute_path(sample_image, entry=entry))
        if p.exists():
            return p
    except Exception:
        pass

    # Then try common_data_folder + sample image. Useful when image paths are
    # already relative to the global data root.
    if common_data_folder:
        p = Path(common_data_folder) / sample_image
        if p.exists():
            return p

    # Then try config folder + sample image.
    try:
        p = Path(entry.config_path).parent / sample_image
        if p.exists():
            return p
    except Exception:
        pass

    # Return the converter-resolved path anyway, so the caller can report it.
    try:
        return Path(conv.resolve_image_absolute_path(sample_image, entry=entry))
    except Exception:
        return Path(sample_image)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render random grounder samples from a converter run YAML for visual debugging."
    )
    parser.add_argument("--config", required=True, type=Path, help="Main converter run config YAML.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Folder where debug images and summary files are written.")
    parser.add_argument("--num-samples", "-n", type=int, default=8, help="Random samples per dataset entry.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--converter-module", default="convert_datasets_to_canonical")
    parser.add_argument("--draw-masks", action="store_true", help="Overlay masks when the grounder provides CVAT-like mask payloads.")
    parser.add_argument("--no-captions", action="store_true", help="Do not draw instance captions in labels.")
    parser.add_argument("--max-side", type=int, default=1800, help="Resize saved debug images so max side is <= this value. Use 0 to disable.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow writing samples without annotations.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    conv = import_converter(args.converter_module)
    run_cfg, _base_dir = conv.build_run_config_from_yaml(args.config)
    options = conv.run_config_to_options(run_cfg)
    entries = conv.collect_entry_specs(run_cfg.configs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"Loaded {len(entries)} dataset entries from {args.config}")

    for entry in entries:
        entry_id = conv.make_entry_id(entry)
        print(f"\n[{entry_id}] loading grounder: {entry.dataset_class_name}")

        try:
            grounder = conv.instantiate_grounder(entry)
            selected, total_seen = reservoir_sample(iter(grounder), args.num_samples, rng)
        except Exception as exc:
            errors.append({"entry_id": entry_id, "error": repr(exc)})
            print(f"  ERROR while sampling: {exc}")
            continue

        print(f"  total samples seen: {total_seen}; selected: {len(selected)}")

        entry_out_dir = args.out_dir / entry_id
        entry_out_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx, sample in selected:
            anns = sample.get("annotations", []) or []
            if not args.allow_empty and not anns:
                print(f"  skip sample {sample_idx}: no annotations")
                continue

            image_path = resolve_image_for_loading(conv, sample, entry, options.common_data_folder)
            original_name = Path(str(sample.get("image", f"sample_{sample_idx}"))).stem
            output_name = f"{sample_idx:06d}_{original_name}.jpg"
            output_path = entry_out_dir / output_name

            row: dict[str, Any] = {
                "entry_id": entry_id,
                "dataset_class": entry.dataset_class_name,
                "sample_index": sample_idx,
                "sample_image": sample.get("image"),
                "resolved_image_path": str(image_path),
                "output_path": str(output_path),
                "num_annotations_grounder": len(anns),
                "asset_caption": sample.get("caption") or sample.get("asset_caption"),
            }

            if not image_path.exists():
                row["status"] = "missing_image"
                errors.append({**row, "error": "Image file does not exist"})
                summary_rows.append(row)
                print(f"  missing image for sample {sample_idx}: {image_path}")
                continue

            try:
                stats = draw_sample(
                    sample=sample,
                    image_path=image_path,
                    output_path=output_path,
                    draw_masks=args.draw_masks,
                    draw_captions=not args.no_captions,
                    max_side=None if args.max_side == 0 else args.max_side,
                )
                row.update(stats)
                row["status"] = "ok"
                print(f"  wrote {output_path.name} ({len(anns)} anns)")
            except Exception as exc:
                row["status"] = "render_error"
                row["error"] = repr(exc)
                errors.append(row.copy())
                print(f"  ERROR rendering sample {sample_idx}: {exc}")

            summary_rows.append(row)

    # Write machine-readable and spreadsheet-friendly summaries.
    summary_json = args.out_dir / "debug_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows, "errors": errors}, f, indent=2, ensure_ascii=False)

    summary_csv = args.out_dir / "debug_summary.csv"
    fieldnames = sorted({k for row in summary_rows for k in row.keys()} | {"status"})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nDone. Debug images: {args.out_dir}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")
    if errors:
        print(f"Warnings/errors: {len(errors)}. Inspect debug_summary.json for details.")


if __name__ == "__main__":
    main()
