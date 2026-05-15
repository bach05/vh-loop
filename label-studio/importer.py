# importer.py
# Handles reading the JSONL dataset and importing tasks into a Label Studio project.
# Can be run standalone to parse and import tasks only (assumes LS is already running).

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote
from pycocotools import mask as coco_mask
import numpy as np

from label_studio_sdk import LabelStudio
from label_studio_sdk.converter import brush


LABEL_MAP: Dict[int, str] = {
    1: "rotor",
    2: "stator",
    3: "shaft",
}


def coco_rle_to_ls_rle(counts: str, size: list) -> list:
    """ Convert COCO RLE string [H, W] to LS brush RLE via label_studio_sdk. """
    binary_mask = coco_mask.decode({"counts": counts, "size": size})
    ls_mask = (binary_mask * 255).astype(np.uint8)
    return brush.mask2rle(ls_mask)


def read_samples(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parses a JSONL file and returns (samples, dataset_id).
    Uses utf-8-sig to handle Windows BOM characters.
    """
    samples: List[Dict[str, Any]] = []
    dataset_id = "unknown"

    if not jsonl_path.exists():
        print(f"ERROR: File not found at path: {jsonl_path.absolute()}")
        return samples, dataset_id

    print(f"Reading file: {jsonl_path.absolute()}")
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    print(f"Total lines found in file: {len(lines)}")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)

            # skip dataset_info row
            if data.get("record_type") == "dataset_info":
                info = data.get("info", {})
                dataset_id = info.get("dataset_id", dataset_id)
                continue

            samples.append(data)

            if "dataset_id" in data:
                dataset_id = data["dataset_id"]

        except json.JSONDecodeError as e:
            print(f"JSON decode error line {i + 1}: {e}")
            print(f"Problematic line content: {line[:100]}...")

    return samples, dataset_id


def build_predictions(instances: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict]:
    """ Converts JSONL target instances into Label Studio prediction results. """
    results = []
    for inst in instances:
        label_id = inst.get("label")
        label_name = LABEL_MAP.get(label_id)
        if not label_name:
            print(f"WARNING: label ID {label_id} not found within LABEL_MAP, skip.")
            continue

        mask = inst.get("mask")
        if not mask:
            continue

        rle_counts = mask.get("counts")
        rle_size = mask.get("size") # [H, W]

        if not rle_counts or not rle_size:
            print("WARNING: invalid RLE mask")
            continue

        ls_rle = coco_rle_to_ls_rle(rle_counts, rle_size)
        if not ls_rle:
            print("WARNING: empty RLE after conversion")
            continue

        results.append({
            "id": f"mask_{len(results)}",
            "type": "brushlabels",
            "from_name": "mask",
            "to_name": "image",
            "original_width": img_w,
            "original_height": img_h,
            "image_rotation": 0,
            "origin": "prediction",
            "value": {
                "format": "rle",
                "rle": ls_rle,
                "brushlabels": [label_name],
            },
        })
    return results


def build_tasks(samples: List[Dict[str, Any]], storage_path: Path) -> List[Dict]:
    """
    Converts raw JSONL samples into Label Studio task dicts.
    Maps to the $image and $sample_id variables in XML config.
    """
    tasks = []
    for i, s in enumerate(samples):
        raw_path = (
            s.get("image_path")
            or s.get("image")
            or s.get("file_path")
            or (s.get("images", {}).get("query", {}) or {}).get("path")
        )

        if not raw_path:
            print(
                f"WARNING: Skipping line {i}, no valid image key found. "
                f"Keys: {list(s.keys())}"
            )
            continue

        img_path = Path(raw_path)
        if not img_path.is_absolute():
            img_path = (storage_path / raw_path).resolve()

        try:
            rel_path = img_path.relative_to(storage_path).as_posix()
        except ValueError:
            rel_path = img_path.as_posix()

        image_url = f"/data/local-files/?d={quote(rel_path)}"
        # Image dimensions for LS coordinate context
        img_size = s.get("images", {}).get("query", {}).get("size", [0, 0])
        img_w, img_h = img_size[0], img_size[1]

        # Build pre-annotations from target instances
        instances = s.get("target", {}).get("instances", [])
        prediction_results = build_predictions(instances, img_w, img_h)

        task: Dict[str, Any] = {
            "data": {
                "image": image_url,
                "sample_id": s.get("sample_id", f"idx_{i}"),
                "storage_path": str(img_path),
            }
        }

        if prediction_results:
            task["predictions"] = [{
                "model_version": "sam2_rle_masks",
                "result": prediction_results,
            }]

        tasks.append(task)
    return tasks


def import_tasks(client: LabelStudio, project_id: int, tasks: List[Dict]) -> None:
    """ Sends the task list to the Label Studio project. """
    print(f"Importing {len(tasks)} tasks into project {project_id}...")
    client.projects.import_tasks(id=project_id, request=tasks)
    print("Import completed.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from connection import make_client

    parser = argparse.ArgumentParser(description="Parse a JSONL file and import tasks into Label Studio.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--ls-url", default="http://localhost:8080")
    args = parser.parse_args()

    client = make_client(args.api_key, args.ls_url)

    samples, dataset_id = read_samples(args.jsonl)
    print(f"Read {len(samples)} samples  |  dataset_id: {dataset_id}")

    if not samples:
        print("Critical error: no samples extracted from JSONL file.")
        raise SystemExit(1)

    tasks = build_tasks(samples, args.storage_path)
    print(f"Valid tasks generated: {len(tasks)}")

    if not tasks:
        print("No valid tasks generated (check previous warning logs).")
        raise SystemExit(1)

    import_tasks(client, args.project_id, tasks)
