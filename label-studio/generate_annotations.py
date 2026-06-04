#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from sam2_pseudolabel_jsonl import read_jsonl, write_jsonl, binary_mask_to_rle


# ----------------------------------------------------------------------
# Label Studio Official RLE Bitstream Decoder (Brush Tool)
# ----------------------------------------------------------------------

class InputStream:
    def __init__(self, data: List[int]):
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        out = self.data[self.i: self.i + size]
        self.i += size
        return int("".join(map(str, out)), 2)


def bytes2bit(data: List[int]) -> List[int]:
    """ Converts an array of bytes into a list of bits. """
    out = []
    for byte in data:
        if byte < 0:
            byte += 256
        bits = bin(byte)[2:].zfill(8)
        out.extend([int(b) for b in bits])
    return out


def decode_rle(rle: List[int], width: int, height: int) -> np.ndarray:
    """
    Decodes the compressed RLE data of the Label Studio brush
    returning a Boolean binary mask of size (H, W).
    """
    bits = bytes2bit(rle)
    rle_input = InputStream(bits)

    # rle-pack compressed format
    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    out = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                out[i] = rle_input.read(word_size)
                i += 1

    # Label Studio exports the brush image in RGBA format
    # The Alpha channel contains the actual mask drawn by the user
    image = np.reshape(out, [height, width, 4])
    alpha_channel = image[:, :, 3]

    # Consider all pixels with an opacity > 0 to be active
    return alpha_channel > 0


# --------------------
# Mask computation
# --------------------

def polygon_to_mask(points: List[List[float]], width: int, height: int) -> np.ndarray:
    """ Converts the points of a Label Studio percentage polygon into a bitmap. """
    pixel_points = []
    for pt in points:
        px = (pt[0] / 100.0) * width
        py = (pt[1] / 100.0) * height
        pixel_points.append((px, py))

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    if len(pixel_points) >= 3:
        draw.polygon(pixel_points, fill=1)
    return np.array(img, dtype=bool)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """ Compute bounding box [xmin, ymin, xmax, ymax] """
    rows = np.any(mask, axis=1)  # search rows (H)
    cols = np.any(mask, axis=0)  # search columns (W)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return int(xmin), int(ymin), int(xmax + 1), int(ymax + 1)


def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float] | None:
    """ Compute the geometric centroid (x, y) of a binary mask. """
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return float(np.mean(x_indices)), float(np.mean(y_indices))


# --------------------------------
# Path Resolution & Alignment
# --------------------------------

def get_sample_uri(sample_rec: Dict[str, Any]) -> str | None:
    sample = sample_rec.get("sample", sample_rec)
    assets = sample.get("assets", [])
    if assets and isinstance(assets, list):
        first_asset = assets[0]
        if isinstance(first_asset, dict):
            uri = first_asset.get("uri") or first_asset.get("metadata", {}).get("original_file_name")
            if uri:
                return str(uri)

    return (
            sample.get("images", {}).get("query", {}).get("path")
            or sample.get("image")
            or sample.get("metadata", {}).get("original_file_name")
    )


def match_ls_task_to_jsonl(ls_task: Dict[str, Any], jsonl_samples: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """ Find the mapping between a Label Studio task and a JSONL record. """
    data = ls_task.get("data", {})
    ls_path = data.get("image") or data.get("img") or data.get("uri") or ""
    if not ls_path:
        return None

    ls_name = Path(ls_path).name
    ls_sample_id = data.get("sample_id") or data.get("id")

    # 1) Explicit ID match
    if ls_sample_id is not None:
        for sample in jsonl_samples:
            s_data = sample.get("sample", sample)
            if str(s_data.get("sample_id")) == str(ls_sample_id):
                return sample

    # 2) Match by exact image file name
    for sample in jsonl_samples:
        uri = get_sample_uri(sample)
        if not uri:
            continue
        orig_name = Path(uri).name
        if ls_name == orig_name:
            return sample

    # 3) Partial match to handle any random prefixes added by Label Studio
    for sample in jsonl_samples:
        uri = get_sample_uri(sample)
        if not uri:
            continue
        orig_name = Path(uri).name
        if orig_name in ls_name or ls_name in orig_name:
            return sample

    return None


# -----------------
# Main Pipeline
# -----------------

def sync_pipeline(
        jsonl_path: Path,
        ls_path: Path,
        output_jsonl_path: Path | None,
        output_coco_path: Path | None,
        write_jsonl_enabled: bool,
        write_coco_enabled: bool,
) -> None:
    print(f"Reading original JSONL file: {jsonl_path}")
    jsonl_records = read_jsonl(jsonl_path)

    print(f"Reading Label Studio annotations: {ls_path}")
    with ls_path.open("r", encoding="utf-8") as f:
        ls_tasks = json.load(f)
        if not isinstance(ls_tasks, list):
            ls_tasks = [ls_tasks]

    # Separate sample records from descriptive records
    sample_records = [r for r in jsonl_records if r.get("record_type") == "sample"]
    other_records = [r for r in jsonl_records if r.get("record_type") != "sample"]

    # Map of the categories recorded in dataset_info
    label_name_to_id: Dict[str, int] = {}
    label_descriptions: Dict[str, str] = {}

    for rec in other_records:
        if rec.get("record_type") == "dataset_info":
            info = rec.setdefault("info", {})
            if isinstance(info, dict):
                info["has_semantic_masks"] = True
                label_info = info.get("label_info", {})
                for lname, ldict in label_info.items():
                    if isinstance(ldict, dict):
                        label_name_to_id[lname] = ldict.get("label_id", len(label_name_to_id) + 1)
                        if "description" in ldict:
                            label_descriptions[lname] = ldict["description"]

    next_label_id = max(label_name_to_id.values()) + 1 if label_name_to_id else 1

    # Task Label Studio -> JSONL
    matched_ls_data: Dict[str, Dict[str, Any]] = {}
    print(f"Matching {len(ls_tasks)} Label Studio tasks with {len(sample_records)} JSONL records...")

    for task in ls_tasks:
        matched_sample = match_ls_task_to_jsonl(task, sample_records)
        if matched_sample:
            s_data = matched_sample.get("sample", matched_sample)
            s_id = str(s_data.get("sample_id"))
            matched_ls_data[s_id] = task
        else:
            print(f"Warning: Unable to associate Label Studio task ID {task.get('id')} with any JSONL record.")

    updated_samples: List[Dict[str, Any]] = []

    for record in sample_records:
        sample = record.setdefault("sample", {})
        sample_id = str(sample.get("sample_id", ""))

        # If the sample is not present in the Label Studio export, keep it unchanged
        if sample_id not in matched_ls_data:
            updated_samples.append(record)
            continue

        task = matched_ls_data[sample_id]

        # Retrieve image resolution
        assets = sample.setdefault("assets", [{}])
        first_asset = assets[0] if assets else {}
        size = first_asset.get("size", [0, 0])  # [W, H]
        img_w, img_h = size[0], size[1]

        annotations = task.get("annotations", [])
        if not annotations:
            # L'utente ha svuotato le annotazioni per questa immagine
            first_asset["annotations"] = []
            if "metadata" in first_asset:
                first_asset["metadata"]["mask_source"] = "label_studio_manual"
            updated_samples.append(record)
            continue

        ls_annotation = annotations[0]
        results = ls_annotation.get("result", [])

        # Group Label Studio results by instance ID
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for res in results:
            res_id = res.get("id") or f"gen_{len(grouped_results)}"
            grouped_results.setdefault(res_id, []).append(res)

        new_instances: List[Dict[str, Any]] = []

        for idx, (inst_id, res_list) in enumerate(grouped_results.items()):
            label_name = None
            mask_bool = None
            bbox_xyxy = None
            keypoint = None

            # If image dimensions are missing in the JSONL, extract them from Label Studio metadata
            if img_w == 0 or img_h == 0:
                for res in res_list:
                    img_w = res.get("original_width") or img_w
                    img_h = res.get("original_height") or img_h
                size = [img_w, img_h]
                first_asset["size"] = size

            for res in res_list:
                val = res.get("value", {})

                # Extract class name
                for label_key in ["brushlabels", "polygonlabels", "rectanglelabels", "labels", "keypointlabels"]:
                    if label_key in val and val[label_key]:
                        label_name = val[label_key][0]
                        break

                # Case 1: Brush (RLE) -> Correct decoding with bitstream
                if "rle" in val:
                    mask_bool = decode_rle(val["rle"], img_w, img_h)

                # Case 2: Polygon
                elif "points" in val:
                    mask_bool = polygon_to_mask(val["points"], img_w, img_h)

                # Case 3: Manual rectangular bounding box (percentage-based)
                elif all(k in val for k in ["x", "y", "width", "height"]):
                    x0 = (val["x"] / 100.0) * img_w
                    y0 = (val["y"] / 100.0) * img_h
                    w = (val["width"] / 100.0) * img_w
                    h = (val["height"] / 100.0) * img_h
                    bbox_xyxy = [int(round(x0)), int(round(y0)), int(round(x0 + w)), int(round(y0 + h))]

                # Case 4: Keypoints
                elif "x" in val and "y" in val:
                    kp_x = (val["x"] / 100.0) * img_w
                    kp_y = (val["y"] / 100.0) * img_h
                    keypoint = {"x": kp_x, "y": kp_y}

            if not label_name:
                continue

            # Deterministic recomputation of real bounding box from corrected mask
            if mask_bool is not None:
                computed_bbox = bbox_from_mask(mask_bool)
                if computed_bbox is not None:
                    bbox_xyxy = computed_bbox

                # If no manual keypoint exists, use exact mask centroid
                if not keypoint:
                    centroid = get_mask_centroid(mask_bool)
                    if centroid:
                        keypoint = {"x": centroid[0], "y": centroid[1]}

            # Dynamically handle new classes created directly in Label Studio
            if label_name not in label_name_to_id:
                label_name_to_id[label_name] = next_label_id
                next_label_id += 1

            lbl_id = label_name_to_id[label_name]

            inst_obj: Dict[str, Any] = {
                "instance_id": f"ls_{inst_id}" if not inst_id.startswith("gen_") else f"inst_{idx:04d}",
                "label_id": lbl_id,
                "label_name": label_name,
                "attributes": {},
            }

            if bbox_xyxy:
                inst_obj["bbox"] = {
                    "tl": {"x": bbox_xyxy[0], "y": bbox_xyxy[1]},
                    "br": {"x": bbox_xyxy[2], "y": bbox_xyxy[3]},
                    "format": "xyxy"
                }

            if mask_bool is not None:
                inst_obj["mask"] = binary_mask_to_rle(mask_bool)
                inst_obj["mask_format"] = "coco_rle"
                inst_obj["source"] = "label_studio_manual"

            if keypoint:
                inst_obj["center_point"] = keypoint

            new_instances.append(inst_obj)

        first_asset["annotations"] = new_instances
        first_asset.setdefault("metadata", {})["mask_source"] = "label_studio_manual"
        updated_samples.append(record)

    # Rebuild final dataset with updated category metadata
    final_records: List[Dict[str, Any]] = []
    for rec in other_records:
        if rec.get("record_type") == "dataset_info":
            info = rec.setdefault("info", {})
            label_info = {}
            for lname, lid in label_name_to_id.items():
                label_info[lname] = {
                    "label_id": lid,
                    "label_name": lname,
                    "description": label_descriptions.get(lname, f"Validated class {lname}"),
                    "aliases": [],
                    "parent_label": "root"
                }
            info["label_info"] = label_info

        final_records.append(rec)
    final_records.extend(updated_samples)

    # Write updated JSONL file
    if write_jsonl_enabled:
        out_jsonl = output_jsonl_path or jsonl_path.with_name(f"{jsonl_path.stem}.updated.jsonl")
        print(f"Saving synchronized JSONL file: {out_jsonl}")
        write_jsonl(out_jsonl, final_records)

    # Generate default COCO v1 export
    if write_coco_enabled:
        out_coco = output_coco_path or jsonl_path.with_name("coco_instances.json")
        print(f"Generating COCO v1 annotations: {out_coco}")

        coco_output = {
            "info": {
                "description": "Instance segmentation exported from Label Studio",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "Validation Pipeline",
                "date_created": "2026-06-04"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        for lname, lid in label_name_to_id.items():
            coco_output["categories"].append({
                "id": lid,
                "name": lname,
                "supercategory": "object"
            })

        image_id_counter = 1
        ann_id_counter = 1

        for record in updated_samples:
            sample = record.get("sample", {})
            uri = get_sample_uri(record)
            if not uri:
                continue

            assets = sample.get("assets", [])
            first_asset = assets[0] if assets else {}
            size = first_asset.get("size", [0, 0])

            coco_img = {
                "id": image_id_counter,
                "width": size[0],
                "height": size[1],
                "file_name": str(uri),
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_output["images"].append(coco_img)

            instances = first_asset.get("annotations", [])
            for inst in instances:
                bbox_data = inst.get("bbox")
                if not bbox_data:
                    continue

                tl = bbox_data.get("tl", {})
                br = bbox_data.get("br", {})
                x0, y0 = tl.get("x", 0), tl.get("y", 0)
                x1, y1 = br.get("x", 0), br.get("y", 0)

                # COCO bounding boxes use the format: [x_min, y_min, width, height]
                coco_bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

                segmentation = {}
                area = int(coco_bbox[2] * coco_bbox[3])

                if "mask" in inst:
                    rle = inst["mask"]
                    segmentation = {
                        "counts": rle.get("counts"),
                        "size": rle.get("size")
                    }
                    try:
                        area = int(mask_utils.area(rle))
                    except Exception:
                        pass

                coco_ann = {
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": inst.get("label_id"),
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": coco_bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(coco_ann)
                ann_id_counter += 1

            image_id_counter += 1

        with out_coco.open("w", encoding="utf-8") as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)

    print("Synchronization completed successfully!")


# ----------------------------
# CLI Parser
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synchronize Label Studio validated/edited annotation streams back into dataset JSONL files "
                    "and generate compliant COCO v1 instance segmentation outputs."
    )
    p.add_argument("--jsonl", type=Path, required=True, help="Path to original dataset input JSONL file")
    p.add_argument("--ls", type=Path, required=True, help="Path to Label Studio JSON export annotations file")
    p.add_argument("--out-jsonl", type=Path, default=None, help="Output destination for the updated JSONL file")
    p.add_argument("--out-coco", type=Path, default=None, help="Output destination for standard COCO annotations file")
    p.add_argument("--no-jsonl", action="store_true", help="Disable generation/updating of the updated JSONL file")
    p.add_argument("--no-coco", action="store_true", help="Disable generation of the standard COCO annotations file")

    args = p.parse_args()

    # Enforce constraint: cannot disable both output pipelines
    if args.no_jsonl and args.no_coco:
        p.error(
            "Operation aborted: Both --no-jsonl and --no-coco were requested. At least one output pipeline must remain active.")
    return args


def main() -> None:
    args = parse_args()
    sync_pipeline(
        jsonl_path=args.jsonl,
        ls_path=args.ls,
        output_jsonl_path=args.out_jsonl,
        output_coco_path=args.out_coco,
        write_jsonl_enabled=not args.no_jsonl,
        write_coco_enabled=not args.no_coco,
    )


if __name__ == "__main__":
    main()