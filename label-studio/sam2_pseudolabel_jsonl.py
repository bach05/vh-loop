#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------
# JSONL utilities
# ----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


# ----------------------------
# Image lookup
# ----------------------------

def _candidate_paths(raw_path: str, images_root: Path) -> List[Path]:
    p = Path(raw_path)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((images_root / p).resolve())
        candidates.append((images_root / raw_path).resolve())
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())

    # Also try basename search later if needed.
    return candidates


def find_image_path(sample_record: Dict[str, Any], images_root: Path, recursive_search: bool = True) -> Path:
    """
    Resolve the local image path using JSONL sample.
    Priority:
      1) sample.assets[].uri
      2) sample.assets[].metadata.original_file_name
      3) basename search under images_root
    """
    sample = sample_record.get("sample", sample_record)
    raw_path = None

    assets = sample.get("assets", [])
    if assets and isinstance(assets, list):
        first_asset = assets[0]
        if isinstance(first_asset, dict):
            raw_path = (
                    first_asset.get("uri")
                    or first_asset.get("metadata", {}).get("original_file_name")
            )

    raw_path = (
            raw_path
            or sample.get("images", {}).get("query", {}).get("path")
            or sample.get("image")
            or sample.get("metadata", {}).get("original_file_name")
    )

    if not raw_path:
        raise FileNotFoundError(f"Sample {sample.get('sample_id')} has no image path field.")

    for candidate in _candidate_paths(str(raw_path), images_root):
        if candidate.exists():
            return candidate

    if recursive_search:
        basename = Path(str(raw_path)).name
        matches = list(images_root.rglob(basename))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate image for sample {sample.get('sample_id')} "
        f"from raw path '{raw_path}' under '{images_root}'."
    )


def load_image_rgb(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


# ----------------------------
# Geometry / mask encoding
# ----------------------------

def bbox_to_pixel_xyxy(bbox: Dict[str, Any], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """ Convert bbox to pixel xyxy. """
    tl = bbox.get("tl", {})
    br = bbox.get("br", {})

    x0 = max(0, min(round(float(tl["x"])), img_w - 1))
    y0 = max(0, min(round(float(tl["y"])), img_h - 1))
    x1 = max(0, min(round(float(br["x"])), img_w))
    y1 = max(0, min(round(float(br["y"])), img_h))

    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)

    return x0, y0, x1, y1


def binary_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """ Convert binary mask (H, W) to COCO-style RLE: {"counts": "...", "size": [H, W]} """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    rle = mask_utils.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")

    size = rle["size"]
    return {"counts": counts, "size": [int(size[0]), int(size[1])]}


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


# ----------------------------
# SAM2 inference
# ----------------------------

def load_sam2_predictor(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> SAM2ImagePredictor:
    sam_model = build_sam2(str(config_path), str(checkpoint_path), device=device)
    return SAM2ImagePredictor(sam_model)


def _predictor_device_type(predictor: SAM2ImagePredictor) -> str:
    try:
        model = getattr(predictor, "model", None)
        if model is not None:
            return next(model.parameters()).device.type
    except Exception:
        pass
    return "cpu"


@torch.inference_mode()
def predict_mask_from_bbox(
    predictor: SAM2ImagePredictor,
    image_rgb: np.ndarray,
    bbox_xyxy_px: Tuple[int, int, int, int],
    multimask_output: bool = False,
) -> np.ndarray:
    """
    Run SAM2 with a box prompt and return a single binary mask (H, W).
    """
    predictor.set_image(image_rgb)
    box = np.array(bbox_xyxy_px, dtype=np.float32)

    # SAM2 predictor signature may accept either (4,) or (1,4) depending on version.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if _predictor_device_type(predictor) == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        try:
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=multimask_output,
            )
        except Exception:
            masks, scores, logits = predictor.predict(
                box=np.asarray([bbox_xyxy_px], dtype=np.float32),
                multimask_output=multimask_output,
            )

    if hasattr(masks, "detach"):
        masks_np = masks.detach().float().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 2:
        mask = masks_np
    elif masks_np.ndim == 3:
        # shape could be [M, H, W] or [1, H, W]
        if multimask_output and masks_np.shape[0] > 1:
            if hasattr(scores, "detach"):
                scores_np = scores.detach().float().cpu().numpy()
            else:
                scores_np = np.asarray(scores)
            best_idx = int(np.argmax(scores_np))
            mask = masks_np[best_idx]
        else:
            mask = masks_np[0]
    elif masks_np.ndim == 4:
        # fallback shape [B, M, H, W], B is expected 1
        mask = masks_np[0, 0]
    else:
        raise RuntimeError(f"Unexpected mask tensor shape: {masks_np.shape}")

    return mask.astype(bool)


# ----------------------------
# Record transformation
# ----------------------------

def infer_and_attach_masks(
    records: List[Dict[str, Any]],
    images_root: Path,
    predictor: SAM2ImagePredictor,
    overwrite_existing_mask: bool = False,
    recursive_search: bool = True,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each sample record:
      - read image dimensions
      - for each instance with bbox, run SAM2
      - save mask as COCO RLE in sample.assets[].annotations[].mask
      - set metadata.mask_source = 'sam2_box_prompt_rle'
      - set dataset_info.info.has_semantic_masks = True
    """
    out_records: List[Dict[str, Any]] = []

    total = len(records)
    for idx, rec in enumerate(records, start=1):
        rec_type = rec.get("record_type")

        if rec_type == "dataset_info":
            info = rec.setdefault("info", {})
            if isinstance(info, dict):
                info["has_semantic_masks"] = True
            out_records.append(rec)
            if progress:
                print(f"[{idx}/{total}] dataset_info updated -> has_semantic_masks=True")
            continue

        if rec_type != "sample":
            out_records.append(rec)
            if progress:
                print(f"[{idx}/{total}] skipped non-sample record_type={rec_type!r}")
            continue

        sample = rec.get("sample", {})
        sample_id = sample.get("sample_id", rec.get("sample_id", f"idx_{idx}"))

        assets = sample.get("assets", [])
        first_asset = assets[0] if assets else {}
        instances = first_asset.get("annotations", []) if isinstance(first_asset, dict) else []
        if not isinstance(instances, list):
            instances = []

        if not instances:
            rec.setdefault("metadata", {})
            rec["metadata"]["mask_source"] = rec.get("metadata", {}).get("mask_source")
            out_records.append(rec)
            if progress:
                print(f"[{idx}/{total}] sample {sample_id}: no instances")
            continue

        try:
            image_path = find_image_path(rec, images_root=images_root, recursive_search=recursive_search)
            image_rgb = load_image_rgb(image_path)
        except Exception as exc:
            print(f"[{idx}/{total}] sample {sample_id}: image lookup failed: {exc}")
            out_records.append(rec)
            continue

        img_h, img_w = image_rgb.shape[:2]

        changed_any = False
        for inst_idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue

            bbox = inst.get("bbox")
            if not bbox or "tl" not in bbox or "br" not in bbox:
                continue

            if inst.get("mask") is not None and not overwrite_existing_mask:
                continue

            try:
                box_px = bbox_to_pixel_xyxy(bbox, img_w=img_w, img_h=img_h)
                mask_bool = predict_mask_from_bbox(
                    predictor=predictor,
                    image_rgb=image_rgb,
                    bbox_xyxy_px=box_px,
                    multimask_output=False,
                )

                if mask_bool.shape[:2] != (img_h, img_w):
                    raise RuntimeError(
                        f"Unexpected mask shape {mask_bool.shape}, expected {(img_h, img_w)}"
                    )

                rle = binary_mask_to_rle(mask_bool)

                inst["mask"] = rle
                inst["mask_format"] = "coco_rle"
                # provenance / bookkeeping
                inst["source"] = inst.get("source", "sam2_box_prompt_rle")

                changed_any = True

                if progress:
                    label = inst.get("label")
                    area = mask_area(mask_bool)
                    print(
                        f"[{idx}/{total}] sample {sample_id} instance {inst_idx}: "
                        f"label={label}, box_px={box_px}, mask_area={area}"
                    )

            except Exception as exc:
                print(
                    f"[{idx}/{total}] sample {sample_id} instance {inst_idx}: SAM2 failed: {exc}"
                )

        asset_metadata = first_asset.setdefault("metadata", {}) if isinstance(first_asset, dict) else {}
        if changed_any:
            asset_metadata["mask_source"] = "sam2_box_prompt_rle"

        out_records.append(rec)

    return out_records


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SAM2 box-prompt pseudolabeling on a JSONL dataset and write masks back as RLE."
    )
    p.add_argument("--jsonl", type=Path, required=True, help="Input JSONL")
    p.add_argument("--images-root", type=Path, required=True, help="Root folder where images live")
    p.add_argument("--output-jsonl", type=Path, default=None, help="Output JSONL (default: <input>.sam2.jsonl)")
    p.add_argument("--sam2-config", type=Path, required=True, help="SAM2 config YAML, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-checkpoint", type=Path, required=True, help="SAM2 checkpoint .pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    p.add_argument("--overwrite-existing-mask", action="store_true", help="Overwrite non-null instance.mask")
    p.add_argument("--no-recursive-search", action="store_true", help="Disable basename recursive search under images-root")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if not args.quiet:
        print(f"Loading SAM2 on device={device}")
        print(f"Config: {args.sam2_config}")
        print(f"Checkpoint: {args.sam2_checkpoint}")

    predictor = load_sam2_predictor(
        config_path=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=device,
    )

    records = read_jsonl(args.jsonl)

    out_records = infer_and_attach_masks(
        records=records,
        images_root=args.images_root,
        predictor=predictor,
        overwrite_existing_mask=args.overwrite_existing_mask,
        recursive_search=not args.no_recursive_search,
        progress=not args.quiet,
    )

    output_jsonl = args.output_jsonl or args.jsonl.with_name(f"{args.jsonl.stem}.sam2.jsonl")
    write_jsonl(output_jsonl, out_records)

    if not args.quiet:
        print(f"Written output JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
