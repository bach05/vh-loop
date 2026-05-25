#!/usr/bin/env python3
"""
convert_datasets_to_canonical.py

Build vh_loop canonical_schema v2 JSONL datasets directly from heterogeneous
dataset grounders.

The grounder layer is responsible only for loading samples from a source format.
Each yielded sample must look like:

    {
        "image": "relative/or/absolute/path.jpg",
        "width":  int,
        "height": int,
        "annotations": [
            {"class": "class_name", "bbox": [x, y, w, h]},
            ...
        ],
    }

This script converts those samples into canonical_schema v2 records:

    DatasetInfoRecord
    DataRecord(sample=SISimpleDataSample(...))

It can be configured in two ways:

1. CLI-only:
    python merge_datasets_to_canonical.py cosmari.yaml warpD.yaml \
        --out-dir /tmp/canonical \
        --dataset-id waste_merged_v1

2. YAML run config:
    python merge_datasets_to_canonical.py \
        --config merge_datasets_to_canonical.yaml

The YAML mode is Hydra-friendly: relative paths are resolved using
hydra.utils.to_absolute_path when Hydra is available, otherwise relative
to the YAML file location.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install it with `pip install pyyaml`.") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

import detection_dataset_grounders

from data.canonical_schema import (
    AnnotationInfo,
    BoundingBox,
    DataRecord,
    DatasetInfo,
    DatasetInfoRecord,
    ImageAsset,
    InstanceAnnotation,
    MessageBuildInfo,
    Point,
    SISimpleDataSample,
)


# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------


@dataclass
class EntrySpec:
    config_name: str
    entry_key: str
    dataset_class_name: str
    source_path: str
    folder_path: str | None
    global_root: str | None
    classes: dict[str, Any]
    config_path: Path


@dataclass
class CanonicalBuildOptions:
    dataset_id: str
    description: str | None
    domain: str | None
    split: str | None
    annotation_source: str
    annotation_quality: str
    sample_type: str
    prompt_template_version: str
    answer_format: str
    normalization_factor: int
    uri_mode: str
    common_data_folder: str | None
    drop_empty: bool
    skip_invalid_bboxes: bool
    generate_center_points: bool
    include_instance_metadata: bool
    sanitize_label_names: bool


@dataclass
class RunConfig:
    """Top-level run configuration for YAML mode."""

    configs: list[str]
    out_dir: str
    common_data_folder: str | None = None
    dataset_id: str = "merged_grounded_dataset_v1"
    description: str | None = "Merged canonical dataset from dataset grounders."
    domain: str | None = None
    split: str | None = None

    annotation_source: str = "imported"
    annotation_quality: str = "auto"

    sample_type: str = "si_simple_data"
    prompt_template_version: str = "single_image_dataset_level_prompt_generation"
    answer_format: str = "tag_bbox_list"
    normalization_factor: int = 1000

    uri_mode: str = "absolute"

    drop_empty: bool = False
    skip_invalid_bboxes: bool = True
    generate_center_points: bool = True
    include_instance_metadata: bool = False
    sanitize_label_names: bool = True


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


_LABEL_ALLOWED_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping/dict, got {type(data)} in {path}")

    return data


def resolve_config_path(path_like: str | Path | None, *, base_dir: Path | None = None) -> str | None:
    """Resolve a path from YAML or CLI.

    Resolution order:
    - URL-like paths are returned unchanged.
    - Absolute paths are returned unchanged.
    - If Hydra is available, hydra.utils.to_absolute_path() is used.
    - Otherwise, relative paths are resolved against base_dir if provided.
    - Otherwise, relative paths are resolved against the current working directory.
    """
    if path_like is None:
        return None

    s = os.path.expanduser(str(path_like))
    if s.startswith(("http://", "https://", "s3://", "gs://")):
        return s

    p = Path(s)
    if p.is_absolute():
        return str(p)

    try:
        from hydra.utils import to_absolute_path  # type: ignore
        hp = Path(to_absolute_path(s))
        if hp.exists():
            return str(hp)
    except Exception:
        pass

    if base_dir is not None:
        return str((base_dir / p).resolve())

    return str(p.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_like: str | Path | None, *, base_dir: Path | None = None) -> str | None:
    if path_like is None:
        return None

    s = os.path.expanduser(str(path_like))
    if s.startswith(("http://", "https://", "s3://", "gs://")):
        return s

    p = Path(s)
    if p.is_absolute():
        return str(p)

    if base_dir is not None:
        p = base_dir / p

    return str(p.resolve() if p.exists() else p)


def sanitize_label_name(name: str) -> str:
    name = str(name).strip()
    name = _LABEL_ALLOWED_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("Cannot sanitize an empty label name")
    return name


def normalize_description(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        desc = value.get("description", None)
        if desc is not None:
            return normalize_description(desc)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def compact_classes_dict(classes: Any) -> dict[str, Any]:
    if classes is None:
        return {}

    if isinstance(classes, dict):
        return {str(k): v for k, v in classes.items()}

    if isinstance(classes, list):
        return {str(v): None for v in classes}

    raise TypeError(f"Unsupported CLASSES type: {type(classes)}")


def get_dataset_class(name: str):
    cls = getattr(detection_dataset_grounders, name, None)
    if cls is None:
        raise ValueError(f"Unknown dataset grounder class: {name!r}")
    return cls


def extract_data_entries(cfg: dict[str, Any], *, cfg_path: Path) -> list[tuple[str, str, str | None]]:
    raw = cfg.get("data_dicts") or cfg.get("data_dict")
    if raw is None:
        return []

    entries: list[tuple[str, str, str | None]] = []

    def add_entry(key: str, val: Any) -> None:
        if isinstance(val, str):
            entries.append((key, val, None))
            return

        if isinstance(val, dict):
            source = val.get("labels") or val.get("path") or val.get("p")
            folder = val.get("folder_path") or val.get("img_root") or val.get("folder")
            if source is None:
                raise ValueError(f"Entry {key!r} in {cfg_path} has no labels/path/p field.")
            entries.append((key, str(source), str(folder) if folder is not None else None))
            return

        raise TypeError(f"Unsupported data_dicts entry type for {key!r}: {type(val)}")

    if isinstance(raw, dict):
        for key, val in raw.items():
            add_entry(str(key), val)

    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                key = Path(item).stem
                add_entry(key, item)
            elif isinstance(item, dict):
                for key, val in item.items():
                    add_entry(str(key), val)
            else:
                raise TypeError(f"Unsupported item in data_dicts list: {type(item)}")

    elif isinstance(raw, str):
        add_entry("dataset", raw)

    else:
        raise TypeError(f"Unsupported data_dicts/data_dict type: {type(raw)}")

    return entries


# -----------------------------------------------------------------------------
# YAML run config
# -----------------------------------------------------------------------------


def normalize_run_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat configs and configs with an `options:` section."""

    out = dict(data)

    # Optional nesting:
    #   configs: [...]
    #   out_dir: ...
    #   options:
    #     dataset_id: ...
    options = out.pop("options", None)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError("The 'options' field must be a mapping/dict.")
        merged = dict(options)
        for key, value in out.items():
            if key not in merged:
                merged[key] = value
        out = merged

    if "config_paths" in out and "configs" not in out:
        out["configs"] = out.pop("config_paths")

    return out


def build_run_config_from_yaml(path: str | Path) -> tuple[RunConfig, Path]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path(resolve_config_path(cfg_path))
    base_dir = cfg_path.parent

    data = normalize_run_config_dict(load_yaml(cfg_path))

    valid_keys = {f.name for f in fields(RunConfig)}
    unknown = set(data.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown keys in run config {cfg_path}: {sorted(unknown)}")

    if "configs" not in data:
        raise ValueError(f"Run config {cfg_path} must define 'configs' or 'config_paths'.")
    if "out_dir" not in data:
        raise ValueError(f"Run config {cfg_path} must define 'out_dir'.")

    data["configs"] = [
        resolve_config_path(p, base_dir=base_dir)
        for p in data["configs"]
    ]
    data["out_dir"] = resolve_config_path(data["out_dir"], base_dir=base_dir)

    return RunConfig(**data), base_dir


def run_config_to_options(cfg: RunConfig) -> CanonicalBuildOptions:
    return CanonicalBuildOptions(
        dataset_id=cfg.dataset_id,
        description=cfg.description,
        domain=cfg.domain,
        split=cfg.split,
        annotation_source=cfg.annotation_source,
        annotation_quality=cfg.annotation_quality,
        sample_type=cfg.sample_type,
        prompt_template_version=cfg.prompt_template_version,
        answer_format=cfg.answer_format,
        normalization_factor=cfg.normalization_factor,
        uri_mode=cfg.uri_mode,
        common_data_folder=cfg.common_data_folder,
        drop_empty=cfg.drop_empty,
        skip_invalid_bboxes=cfg.skip_invalid_bboxes,
        generate_center_points=cfg.generate_center_points,
        include_instance_metadata=cfg.include_instance_metadata,
        sanitize_label_names=cfg.sanitize_label_names,
    )


# -----------------------------------------------------------------------------
# Grounder instantiation
# -----------------------------------------------------------------------------


def instantiate_grounder(entry: EntrySpec):
    DatasetClass = get_dataset_class(entry.dataset_class_name)

    if entry.dataset_class_name == "CocoWasteDataset":
        coco_path = resolve_path(entry.source_path, base_dir=entry.config_path.parent)
        assert coco_path is not None
        coco = load_json(coco_path)
        return DatasetClass(
            coco,
            entry.classes,
            detailed_ratio=0.0,
            img_root=entry.folder_path,
        )

    if entry.dataset_class_name == "WarpDataset":
        labels_dir = resolve_path(entry.source_path, base_dir=entry.config_path.parent)
        entry_global_root = resolve_path(entry.global_root, base_dir=entry.config_path.parent)

        if entry_global_root and entry.folder_path:
            images_dir = str(Path(entry_global_root) / entry.folder_path)
        elif entry.folder_path:
            images_dir = resolve_path(entry.folder_path, base_dir=entry.config_path.parent)
        elif entry_global_root:
            images_dir = entry_global_root
        else:
            images_dir = ""

        assert labels_dir is not None
        return DatasetClass(
            images_dir or "",
            labels_dir,
            entry.classes,
            detailed_ratio=0.0,
            img_root=entry.folder_path,
        )

    source_path = resolve_path(entry.source_path, base_dir=entry.config_path.parent)
    if source_path is None:
        raise ValueError(f"Cannot resolve source path for entry {entry.entry_key!r}")

    try:
        source_obj = load_json(source_path)
        return DatasetClass(
            source_obj,
            entry.classes,
            detailed_ratio=0.0,
            img_root=entry.folder_path,
        )
    except Exception:
        return DatasetClass(
            source_path,
            entry.classes,
            detailed_ratio=0.0,
            img_root=entry.folder_path,
        )


def collect_entry_specs(config_paths: Iterable[str | Path]) -> list[EntrySpec]:
    specs: list[EntrySpec] = []

    for cfg_path_like in config_paths:
        cfg_path = Path(str(cfg_path_like)).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = cfg_path.resolve()

        cfg = load_config(cfg_path)
        cfg_name = cfg_path.stem
        dataset_class_name = str(cfg.get("dataset") or "CocoWasteDataset")
        classes = compact_classes_dict(cfg.get("CLASSES"))
        global_root = cfg.get("GLOBAL_ROOT")

        entries = extract_data_entries(cfg, cfg_path=cfg_path)
        if not entries:
            print(f"Warning: no data entries found in {cfg_path}")
            continue

        for entry_key, source_path, folder_path in entries:
            specs.append(
                EntrySpec(
                    config_name=cfg_name,
                    entry_key=str(entry_key),
                    dataset_class_name=dataset_class_name,
                    source_path=source_path,
                    folder_path=folder_path,
                    global_root=str(global_root) if global_root is not None else None,
                    classes=classes,
                    config_path=cfg_path,
                )
            )

    return specs


# -----------------------------------------------------------------------------
# Canonical conversion
# -----------------------------------------------------------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def coco_xywh_to_xyxy_pixel(
    bbox_xywh: list[float] | tuple[float, ...],
    *,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = map(float, bbox_xywh)

    x1 = clamp(x, 0.0, float(img_width))
    y1 = clamp(y, 0.0, float(img_height))
    x2 = clamp(x + w, 0.0, float(img_width))
    y2 = clamp(y + h, 0.0, float(img_height))

    return (
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    )


def valid_xyxy(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 < x2 and y1 < y2


def resolve_image_absolute_path(
    sample_image: str,
    *,
    entry: EntrySpec,
) -> str:
    """Resolve a grounder-provided image path to an absolute path when possible.

    Grounders may return:
    - an absolute path;
    - a path already prefixed by folder_path;
    - a path relative to GLOBAL_ROOT;
    - a path relative to the dataset config folder.
    """
    if sample_image.startswith(("http://", "https://", "s3://", "gs://")):
        return sample_image

    p = Path(sample_image)
    if p.is_absolute():
        return str(p)

    global_root = resolve_path(entry.global_root, base_dir=entry.config_path.parent)
    if global_root is not None:
        return str((Path(global_root) / p).resolve())

    return str((entry.config_path.parent / p).resolve())


def make_image_uri(
    sample_image: str,
    *,
    entry: EntrySpec,
    uri_mode: str,
    common_data_folder: str | None = None,
) -> str:
    """Build the URI stored in ImageAsset.

    If ``uri_mode == "absolute"``, store the resolved absolute image path.

    If ``uri_mode == "relative"`` and ``common_data_folder`` is provided,
    store the image path relative to common_data_folder. At training/loading
    time, pass ``dataset_root=common_data_folder`` so Asset.resolve_path()
    reconstructs the absolute path.

    If ``uri_mode == "relative"`` and no common_data_folder is provided,
    keep the path as returned by the grounder.
    """
    if sample_image.startswith(("http://", "https://", "s3://", "gs://")):
        return sample_image

    if uri_mode == "absolute":
        return resolve_image_absolute_path(sample_image, entry=entry)

    if uri_mode != "relative":
        raise ValueError(f"Unsupported uri_mode={uri_mode!r}")

    if common_data_folder is None:
        return str(Path(sample_image))

    common_root = Path(resolve_config_path(common_data_folder, base_dir=entry.config_path.parent) or common_data_folder)
    abs_image = Path(resolve_image_absolute_path(sample_image, entry=entry))

    try:
        return str(abs_image.resolve().relative_to(common_root.resolve()))
    except ValueError:
        # Still make the URI relative to common_data_folder, even if the image is
        # outside that folder. This may produce '../...' segments, which remain
        # resolvable with dataset_root=common_data_folder.
        return os.path.relpath(str(abs_image), start=str(common_root))


def build_label_tables(
    *,
    entries: list[EntrySpec],
    samples_by_entry: dict[str, list[dict[str, Any]]],
    sanitize_labels: bool,
) -> tuple[dict[str, int], dict[str, str | None]]:
    label_order: list[str] = []
    descriptions: dict[str, str | None] = {}

    def add_label(raw_name: str, desc_source: Any = None) -> str:
        label_name = sanitize_label_name(raw_name) if sanitize_labels else str(raw_name)
        if label_name not in label_order:
            label_order.append(label_name)
        if label_name not in descriptions:
            descriptions[label_name] = normalize_description(desc_source)
        return label_name

    for entry in entries:
        for raw_name, desc in entry.classes.items():
            add_label(raw_name, desc)

    for entry in entries:
        key = make_entry_id(entry)
        for sample in samples_by_entry.get(key, []):
            for ann in sample.get("annotations", []) or []:
                raw_name = ann.get("class")
                if raw_name is not None:
                    add_label(str(raw_name), None)

    label_to_id = {label_name: idx + 1 for idx, label_name in enumerate(label_order)}
    return label_to_id, descriptions


def make_entry_id(entry: EntrySpec) -> str:
    return f"{entry.config_name}_{entry.entry_key}"


def sample_to_data_record(
    sample: dict[str, Any],
    *,
    sample_id: str,
    entry: EntrySpec,
    label_to_id: dict[str, int],
    options: CanonicalBuildOptions,
) -> tuple[DataRecord | None, int]:
    width = int(sample["width"])
    height = int(sample["height"])

    annotations: list[InstanceAnnotation] = []
    invalid_count = 0

    for ann_idx, ann in enumerate(sample.get("annotations", []) or []):
        raw_label = ann.get("class")
        raw_bbox = ann.get("bbox")

        if raw_label is None or raw_bbox is None:
            invalid_count += 1
            continue

        label_name = sanitize_label_name(raw_label) if options.sanitize_label_names else str(raw_label)
        if label_name not in label_to_id:
            raise ValueError(f"Label {label_name!r} is not present in label_to_id mapping.")

        try:
            xyxy = coco_xywh_to_xyxy_pixel(raw_bbox, img_width=width, img_height=height)
        except Exception:
            invalid_count += 1
            if options.skip_invalid_bboxes:
                continue
            raise

        if not valid_xyxy(xyxy):
            invalid_count += 1
            if options.skip_invalid_bboxes:
                continue
            raise ValueError(f"Invalid bbox for sample_id={sample_id}: {raw_bbox!r} -> {xyxy!r}")

        x1, y1, x2, y2 = xyxy
        bbox = BoundingBox(tl=Point(x=x1, y=y1), br=Point(x=x2, y=y2))

        points = None
        if options.generate_center_points:
            points = [
                Point(
                    x=round((x1 + x2) / 2),
                    y=round((y1 + y2) / 2),
                    is_positive=True,
                )
            ]

        metadata_attrs: dict[str, str] = {}
        if options.include_instance_metadata:
            metadata_attrs = {
                "raw_label": str(raw_label),
                "raw_bbox_xywh": json.dumps(raw_bbox, separators=(",", ":")),
                "source_entry": make_entry_id(entry),
            }

        annotations.append(
            InstanceAnnotation(
                instance_id=f"obj_{ann_idx:05d}",
                label_id=label_to_id[label_name],
                label_name=label_name,
                bbox=bbox,
                points=points,
                mask=None,
                caption=None,
                attributes=metadata_attrs,
            )
        )

    if options.drop_empty and not annotations:
        return None, invalid_count

    image_uri = make_image_uri(
        str(sample["image"]),
        entry=entry,
        uri_mode=options.uri_mode,
        common_data_folder=options.common_data_folder,
    )

    asset_metadata = {
        "source_entry": make_entry_id(entry),
        "source_config": str(entry.config_path),
        "dataset_class": entry.dataset_class_name,
        "global_root": entry.global_root,
        "folder_path": entry.folder_path,
        "num_instances": len(annotations),
        "num_invalid_annotations_skipped": invalid_count,
        "bbox_source": "grounder_bbox_xywh_pixel",
        "bbox_coordinate_space": "pixel_xyxy",
    }

    asset = ImageAsset(
        type="image",
        uri=image_uri,
        size=(width, height),
        annotations=annotations,
        metadata=asset_metadata,
    )

    data_sample = SISimpleDataSample(
        sample_type=options.sample_type,
        sample_id=sample_id,
        assets=[asset],
    )

    return DataRecord(sample=data_sample), invalid_count


def make_dataset_info_record(
    *,
    dataset_id: str,
    label_to_id: dict[str, int],
    label_descriptions: dict[str, str | None],
    entries: list[EntrySpec],
    options: CanonicalBuildOptions,
    subset_label_names: set[str] | None = None,
) -> DatasetInfoRecord:
    now = datetime.now().isoformat(timespec="seconds")

    if subset_label_names is None:
        selected = list(label_to_id.keys())
    else:
        selected = [name for name in label_to_id.keys() if name in subset_label_names]

    label_info = {
        label_name: {
            "label_id": label_to_id[label_name],
            "label_name": label_name,
            "description": label_descriptions.get(label_name),
            "aliases": [],
            "parent_label": None,
        }
        for label_name in selected
    }

    info = DatasetInfo(
        dataset_id=dataset_id,
        description=options.description,
        annotation_info=AnnotationInfo(
            source_type=options.annotation_source,  # type: ignore[arg-type]
            quality=options.annotation_quality,  # type: ignore[arg-type]
            notes="Converted from heterogeneous dataset grounders.",
        ),
        domain=options.domain,
        split=options.split,
        date_collected=None,
        date_last_update=now,
        label_info=label_info,
        message_build_info=MessageBuildInfo(
            prompt_template_version=options.prompt_template_version,
            answer_format=options.answer_format,  # type: ignore[arg-type]
            normalization_factor=options.normalization_factor,
            metadata={
                "sample_type": options.sample_type,
                "target_encoding": options.answer_format,
                "bbox_storage_coordinates": "pixel_xyxy",
                "bbox_message_coordinates": f"normalized_{options.normalization_factor}",
            },
        ),
        metadata={
            "source_format": "grounded_dataset_samples",
            "schema_conversion": "grounders_to_canonical_schema_v2",
            "date_converted": now,
            "uri_mode": options.uri_mode,
            "common_data_folder": options.common_data_folder,
            "entries": [
                {
                    "entry_id": make_entry_id(entry),
                    "dataset_class": entry.dataset_class_name,
                    "source_path": entry.source_path,
                    "folder_path": entry.folder_path,
                    "global_root": entry.global_root,
                    "config_path": str(entry.config_path),
                }
                for entry in entries
            ],
        },
    )

    return DatasetInfoRecord(info=info)


def write_jsonl_record(f, obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    f.write("\n")


def write_canonical_jsonl(
    path: str | Path,
    *,
    dataset_info_record: DatasetInfoRecord,
    records: list[DataRecord],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        write_jsonl_record(f, dataset_info_record)
        for record in records:
            write_jsonl_record(f, record)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def validate_options(options: CanonicalBuildOptions) -> None:
    if options.sample_type != "si_simple_data":
        raise ValueError("Only sample_type='si_simple_data' is currently supported.")
    if options.answer_format != "tag_bbox_list":
        raise ValueError("Only answer_format='tag_bbox_list' is currently supported.")
    if options.normalization_factor <= 0:
        raise ValueError("normalization_factor must be > 0")
    if options.uri_mode not in {"absolute", "relative"}:
        raise ValueError("uri_mode must be either 'absolute' or 'relative'.")


def build_options(args: argparse.Namespace) -> CanonicalBuildOptions:
    options = CanonicalBuildOptions(
        dataset_id=args.dataset_id,
        description=args.description,
        domain=args.domain,
        split=args.split,
        annotation_source=args.annotation_source,
        annotation_quality=args.annotation_quality,
        sample_type=args.sample_type,
        prompt_template_version=args.prompt_template_version,
        answer_format=args.answer_format,
        normalization_factor=args.normalization_factor,
        uri_mode=args.uri_mode,
        drop_empty=args.drop_empty,
        skip_invalid_bboxes=args.skip_invalid_bboxes,
        generate_center_points=args.generate_center_points,
        include_instance_metadata=args.include_instance_metadata,
        sanitize_label_names=args.sanitize_label_names,
    )
    validate_options(options)
    return options


def run_conversion(config_paths: list[str | Path], options: CanonicalBuildOptions, out_dir: Path) -> dict[str, Any]:
    validate_options(options)

    entries = collect_entry_specs(config_paths)
    if not entries:
        raise RuntimeError("No dataset entries found.")

    samples_by_entry: dict[str, list[dict[str, Any]]] = {}

    for entry in tqdm(entries, desc="Loading entries"):
        entry_id = make_entry_id(entry)
        grounder = instantiate_grounder(entry)
        samples = list(grounder)
        samples_by_entry[entry_id] = samples

    label_to_id, label_descriptions = build_label_tables(
        entries=entries,
        samples_by_entry=samples_by_entry,
        sanitize_labels=options.sanitize_label_names,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    merged_records: list[DataRecord] = []
    per_entry_files: list[str] = []
    total_invalid = 0

    for entry in tqdm(entries, desc="Writing per-entry canonical JSONL"):
        entry_id = make_entry_id(entry)
        source_samples = samples_by_entry[entry_id]
        entry_records: list[DataRecord] = []
        entry_label_names: set[str] = set()

        for idx, sample in enumerate(source_samples, start=1):
            sample_id = f"{entry_id}_{idx:06d}"
            record, invalid = sample_to_data_record(
                sample,
                sample_id=sample_id,
                entry=entry,
                label_to_id=label_to_id,
                options=options,
            )
            total_invalid += invalid

            if record is None:
                continue

            entry_records.append(record)
            merged_records.append(record)

            for ann in record.sample.assets[0].annotations:
                entry_label_names.add(ann.label_name)

        entry_info = make_dataset_info_record(
            dataset_id=f"{options.dataset_id}_{entry_id}",
            label_to_id=label_to_id,
            label_descriptions=label_descriptions,
            entries=[entry],
            options=options,
            subset_label_names=entry_label_names,
        )

        entry_out = out_dir / f"{entry_id}.canonical.jsonl"
        write_canonical_jsonl(entry_out, dataset_info_record=entry_info, records=entry_records)
        per_entry_files.append(str(entry_out))

    merged_info = make_dataset_info_record(
        dataset_id=options.dataset_id,
        label_to_id=label_to_id,
        label_descriptions=label_descriptions,
        entries=entries,
        options=options,
        subset_label_names=None,
    )

    merged_out = out_dir / "merged.canonical.jsonl"
    write_canonical_jsonl(merged_out, dataset_info_record=merged_info, records=merged_records)

    return {
        "output_dir": str(out_dir),
        "merged_output": str(merged_out),
        "per_entry_outputs": per_entry_files,
        "num_entries": len(entries),
        "num_samples_written": len(merged_records),
        "num_invalid_annotations_skipped": total_invalid,
        "num_labels": len(label_to_id),
        "labels": label_to_id,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert heterogeneous dataset grounders directly to canonical_schema v2 JSONL."
    )

    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            "Dataset YAML config files, e.g. cosmari.yaml warpD.yaml zerowaste.yaml. "
            "Not required when --config is used."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional main YAML run config containing configs/out_dir/options.",
    )

    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--common-data-folder",
        type=str,
        default=None,
        help="Common root folder used to store relative asset.uri paths when uri_mode=relative.",
    )
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)

    parser.add_argument("--annotation-source", type=str, default=None)
    parser.add_argument("--annotation-quality", type=str, default=None)

    parser.add_argument("--sample-type", type=str, default=None)
    parser.add_argument("--prompt-template-version", type=str, default=None)
    parser.add_argument("--answer-format", type=str, default=None)
    parser.add_argument("--normalization-factor", type=int, default=None)

    parser.add_argument(
        "--uri-mode",
        choices=["absolute", "relative"],
        default=None,
    )

    # Boolean CLI overrides. They default to None so YAML can control them.
    parser.add_argument("--drop-empty", dest="drop_empty", action="store_true", default=None)
    parser.add_argument("--no-drop-empty", dest="drop_empty", action="store_false")

    parser.add_argument("--skip-invalid-bboxes", dest="skip_invalid_bboxes", action="store_true", default=None)
    parser.add_argument("--no-skip-invalid-bboxes", dest="skip_invalid_bboxes", action="store_false")

    parser.add_argument("--generate-center-points", dest="generate_center_points", action="store_true", default=None)
    parser.add_argument("--no-generate-center-points", dest="generate_center_points", action="store_false")

    parser.add_argument("--include-instance-metadata", dest="include_instance_metadata", action="store_true", default=None)
    parser.add_argument("--no-include-instance-metadata", dest="include_instance_metadata", action="store_false")

    parser.add_argument("--sanitize-label-names", dest="sanitize_label_names", action="store_true", default=None)
    parser.add_argument("--no-sanitize-label-names", dest="sanitize_label_names", action="store_false")

    return parser


def apply_cli_overrides(run_cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    data = dict(run_cfg.__dict__)

    scalar_keys = [
        "dataset_id",
        "description",
        "domain",
        "split",
        "annotation_source",
        "annotation_quality",
        "sample_type",
        "prompt_template_version",
        "answer_format",
        "normalization_factor",
        "uri_mode",
        "common_data_folder",
        "drop_empty",
        "skip_invalid_bboxes",
        "generate_center_points",
        "include_instance_metadata",
        "sanitize_label_names",
    ]

    for key in scalar_keys:
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value

    if args.out_dir is not None:
        data["out_dir"] = str(args.out_dir)

    if args.configs:
        data["configs"] = list(args.configs)

    return RunConfig(**data)


def args_to_run_config(args: argparse.Namespace) -> RunConfig:
    if args.config is not None:
        run_cfg, base_dir = build_run_config_from_yaml(args.config)
        run_cfg = apply_cli_overrides(run_cfg, args)

        # CLI configs/out_dir may be relative to current cwd, not YAML dir.
        run_cfg.configs = [
            resolve_config_path(p, base_dir=base_dir) for p in run_cfg.configs
        ]
        run_cfg.out_dir = resolve_config_path(run_cfg.out_dir, base_dir=base_dir) or run_cfg.out_dir
        return run_cfg

    if not args.configs:
        raise ValueError("Provide dataset config paths positionally or use --config main_run.yaml.")
    if args.out_dir is None:
        raise ValueError("--out-dir is required when --config is not used.")

    # CLI-only defaults.
    run_cfg = RunConfig(
        configs=[resolve_config_path(p) for p in args.configs],
        out_dir=str(args.out_dir),
    )
    return apply_cli_overrides(run_cfg, args)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    run_cfg = args_to_run_config(args)
    options = run_config_to_options(run_cfg)
    validate_options(options)

    summary = run_conversion(run_cfg.configs, options, Path(run_cfg.out_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()