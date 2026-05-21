#!/usr/bin/env python3
"""
Convert a COCO instance-annotation JSON into the vh_loop canonical_schema v2 JSONL format.

Output JSONL format:

Line 1:
    DatasetInfoRecord

Lines 2+:
    DataRecord(sample=SISimpleDataSample(...))

Important conventions:
- COCO bbox coordinates are stored in pixel space inside BoundingBox(tl, br).
- Normalization to a 1000x1000, or any other configured grid, is deferred to
  SISimpleDataSample.sample_to_message() through MessageBuildInfo.normalization_factor.
- The generated sample type is currently "si_simple_data": one image asset with
  active instance annotations attached to the asset.
- Reusable class descriptions can be loaded from an external YAML file via
  label_descriptions_yaml.

Example:
    python data_mining/converters/coco_to_canonical_dataset.py \
        --config configs/data/panizzolo_coco2canonical.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

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


# =============================================================================
# Config
# =============================================================================


@dataclass
class CocoToCanonicalConfig:
    coco_json: Path | None = None
    output_jsonl: Path | None = None

    dataset_id: str | None = None
    split: str | None = None
    description: str | None = None
    domain: str | None = None

    # Must be compatible with AnnotationInfo.source_type / quality.
    annotation_source: str = "human"
    annotation_quality: str = "gold"

    # Current converter target.
    sample_type: str = "si_simple_data"

    # Message generation metadata stored in DatasetInfo.message_build_info.
    prompt_template_version: str = "single_image_dataset_level_prompt_generation"
    answer_format: str = "tag_bbox_list"
    normalization_factor: int = 1000

    disable_pbar: bool = False
    drop_empty: bool = False
    include_categories: set[int] | None = None
    skip_invalid_bboxes: bool = True
    include_instance_metadata: bool = False
    generate_center_points: bool = True

    # Label-name handling.
    # The canonical schema requires compact label names without spaces.
    sanitize_label_names: bool = True
    label_name_overrides: dict[int, str] | None = None

    # Inline descriptions:
    #   label_descriptions:
    #     PET_bottle: "..."
    # or richer entries:
    #   label_descriptions:
    #     PET_bottle:
    #       description: "..."
    #       aliases: [...]
    #       parent_label: plastic
    label_descriptions: dict[str, Any] | None = None

    # External reusable taxonomy / labels file:
    #   taxonomy_id: waste_labels_v1
    #   labels:
    #     PET_bottle:
    #       description: "..."
    #       aliases: [...]
    #       parent_label: plastic
    label_descriptions_yaml: Path | None = None

    # Dataset-specific description override by canonical label_name.
    label_description_overrides: dict[str, str] | None = None

    # If true, every selected canonical label must have a non-empty description.
    require_label_descriptions: bool = False

    # Sample ids.
    sample_id_prefix: str | None = None
    sample_id_strategy: str = "image_id"  # image_id | sequential | file_stem


# =============================================================================
# YAML / config loading
# =============================================================================


def load_yaml_config(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if isinstance(data, dict) or isinstance(data, list):
        return data

    raise TypeError(f"YAML config root must be a mapping or a list of mappings, got {type(data)}")


def _resolve_yaml_path(value: Any, yaml_base_dir: Path | None) -> Path:
    path = Path(value)
    if yaml_base_dir is not None and not path.is_absolute():
        path = yaml_base_dir / path
    return path


def normalize_include_categories(value: Any) -> set[int] | None:
    if value is None or value == "":
        return None

    if isinstance(value, str):
        return {int(v.strip()) for v in value.split(",") if v.strip()}

    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}

    raise TypeError("include_categories must be null, a list of ints, or a comma-separated string")


def normalize_int_key_dict(value: Any) -> dict[int, str] | None:
    """Normalize YAML maps whose keys may be strings into dict[int, str]."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping/dict")
    return {int(k): str(v) for k, v in value.items()}


def normalize_str_description_dict(value: Any) -> dict[str, str] | None:
    """Normalize a description override map into dict[str, str]."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping/dict")
    return {str(k): str(v) for k, v in value.items()}


def normalize_label_descriptions(value: Any) -> dict[str, Any] | None:
    """Normalize inline label_descriptions.

    Supports both:
        label_name: "description"
    and:
        label_name:
          description: "..."
          aliases: [...]
          parent_label: "..."
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("label_descriptions must be a mapping/dict")
    return {str(k): v for k, v in value.items()}


def build_config_from_yaml_and_cli(
    args: argparse.Namespace,
    yaml_item: dict[str, Any] | None = None,
    yaml_base_dir: Path | None = None,
) -> CocoToCanonicalConfig:
    cfg = CocoToCanonicalConfig()

    if yaml_item is not None:
        if not isinstance(yaml_item, dict):
            raise TypeError("Each YAML entry must be a mapping/dict")

        valid_keys = set(asdict(cfg).keys())
        unknown_keys = set(yaml_item.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(f"Unknown keys in YAML config entry: {sorted(unknown_keys)}")

        path_keys = {"coco_json", "output_jsonl", "label_descriptions_yaml"}

        for key, value in yaml_item.items():
            if key in path_keys and value is not None:
                value = _resolve_yaml_path(value, yaml_base_dir)
            elif key == "include_categories":
                value = normalize_include_categories(value)
            elif key == "label_name_overrides":
                value = normalize_int_key_dict(value)
            elif key == "label_descriptions":
                value = normalize_label_descriptions(value)
            elif key == "label_description_overrides":
                value = normalize_str_description_dict(value)

            setattr(cfg, key, value)

    cli_overrides = {
        "coco_json": args.coco_json,
        "output_jsonl": args.output_jsonl,
        "dataset_id": args.dataset_id,
        "split": args.split,
        "description": args.description,
        "domain": args.domain,
        "annotation_source": args.annotation_source,
        "annotation_quality": args.annotation_quality,
        "sample_type": args.sample_type,
        "prompt_template_version": args.prompt_template_version,
        "answer_format": args.answer_format,
        "normalization_factor": args.normalization_factor,
        "disable_pbar": args.disable_pbar,
        "drop_empty": args.drop_empty,
        "include_categories": args.include_categories,
        "skip_invalid_bboxes": args.skip_invalid_bboxes,
        "include_instance_metadata": args.include_instance_metadata,
        "generate_center_points": args.generate_center_points,
        "sanitize_label_names": args.sanitize_label_names,
        "label_descriptions_yaml": args.label_descriptions_yaml,
        "require_label_descriptions": args.require_label_descriptions,
        "sample_id_prefix": args.sample_id_prefix,
        "sample_id_strategy": args.sample_id_strategy,
    }

    path_keys = {"coco_json", "output_jsonl", "label_descriptions_yaml"}

    for key, value in cli_overrides.items():
        if value is None:
            continue
        if key in path_keys:
            value = Path(value)
        elif key == "include_categories":
            value = normalize_include_categories(value)
        setattr(cfg, key, value)

    missing = []
    if cfg.coco_json is None:
        missing.append("coco_json")
    if cfg.output_jsonl is None:
        missing.append("output_jsonl")
    if cfg.dataset_id is None:
        missing.append("dataset_id")

    if missing:
        raise ValueError(
            "Missing required config values after merging YAML and CLI: "
            f"{missing}. Provide them in YAML or through CLI."
        )

    if cfg.sample_type != "si_simple_data":
        raise ValueError(
            "This converter currently emits SISimpleDataSample records only, "
            "therefore sample_type must be 'si_simple_data'."
        )

    if cfg.answer_format != "tag_bbox_list":
        raise ValueError(
            "This converter currently supports answer_format='tag_bbox_list' only, "
            f"got {cfg.answer_format!r}."
        )

    if cfg.normalization_factor <= 0:
        raise ValueError("normalization_factor must be > 0")

    return cfg


def build_configs_from_yaml_and_cli(args: argparse.Namespace) -> list[CocoToCanonicalConfig]:
    yaml_root = None
    yaml_base_dir = None

    if args.config is not None:
        yaml_path = Path(args.config)
        yaml_root = load_yaml_config(yaml_path)
        yaml_base_dir = yaml_path.parent

    if isinstance(yaml_root, list):
        return [
            build_config_from_yaml_and_cli(
                args,
                yaml_item=(item or {}),
                yaml_base_dir=yaml_base_dir,
            )
            for item in yaml_root
        ]

    return [
        build_config_from_yaml_and_cli(
            args,
            yaml_item=(yaml_root if isinstance(yaml_root, dict) else None),
            yaml_base_dir=yaml_base_dir,
        )
    ]


# =============================================================================
# External label descriptions / reusable taxonomy
# =============================================================================


def normalize_label_entry(entry: Any) -> dict[str, Any]:
    """Normalize one label entry.

    Accepted forms:
        PET_bottle: "description"
        PET_bottle:
          description: "..."
          aliases: [...]
          parent_label: plastic
    """
    if entry is None:
        return {}
    if isinstance(entry, str):
        return {"description": entry}
    if isinstance(entry, dict):
        return dict(entry)
    raise TypeError(f"Unsupported label description entry: {type(entry)}")


def load_label_description_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError("label_descriptions_yaml root must be a mapping/dict.")

    if "labels" in data:
        labels = data.get("labels") or {}
    else:
        # Allow a compact file that is itself a label mapping.
        # Ignore common taxonomy metadata keys if present.
        metadata_keys = {"taxonomy_id", "description", "version", "metadata"}
        labels = {k: v for k, v in data.items() if k not in metadata_keys}

    if not isinstance(labels, dict):
        raise TypeError("label_descriptions_yaml must contain a 'labels' mapping.")

    return {str(label_name): normalize_label_entry(entry) for label_name, entry in labels.items()}


def load_external_label_info(cfg: CocoToCanonicalConfig) -> dict[str, Any]:
    """Load and merge reusable and dataset-specific label metadata.

    Precedence:
        external label_descriptions_yaml
            < inline label_descriptions
            < label_description_overrides
    """
    labels: dict[str, Any] = {}

    # 1. External reusable taxonomy file.
    labels.update(load_label_description_yaml(cfg.label_descriptions_yaml))

    # 2. Inline label descriptions or rich entries.
    if cfg.label_descriptions:
        for label_name, entry in cfg.label_descriptions.items():
            current = normalize_label_entry(labels.get(label_name))
            incoming = normalize_label_entry(entry)
            current.update(incoming)
            labels[label_name] = current

    # 3. Dataset-specific description overrides.
    if cfg.label_description_overrides:
        for label_name, description in cfg.label_description_overrides.items():
            current = normalize_label_entry(labels.get(label_name))
            current["description"] = description
            labels[label_name] = current

    return labels


# =============================================================================
# COCO helpers
# =============================================================================


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def coco_xywh_to_clipped_xyxy(
    bbox_xywh: list[float],
    *,
    img_width: int,
    img_height: int,
) -> list[int]:
    """Convert COCO [x, y, w, h] to clipped pixel xyxy.

    The bottom-right corner is interpreted as exclusive. We use
    floor for top-left and ceil for bottom-right to avoid collapsing
    very small but valid boxes.
    """
    x, y, w, h = bbox_xywh

    x1 = clamp(float(x), 0.0, float(img_width))
    y1 = clamp(float(y), 0.0, float(img_height))
    x2 = clamp(float(x + w), 0.0, float(img_width))
    y2 = clamp(float(y + h), 0.0, float(img_height))

    return [
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    ]


def is_valid_xyxy(box: list[int] | list[float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 < x2 and y1 < y2


def build_category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}

    for cat in coco.get("categories", []):
        cat_id = int(cat["id"])
        name = str(cat["name"])
        id_to_name[cat_id] = name

    name_to_id = {name: cat_id for cat_id, name in id_to_name.items()}
    return id_to_name, name_to_id


def group_annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        anns_by_image[image_id].append(ann)

    return anns_by_image


# =============================================================================
# Label handling
# =============================================================================


_LABEL_ALLOWED_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def sanitize_label_name(name: str) -> str:
    name = name.strip()
    name = _LABEL_ALLOWED_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("Cannot sanitize an empty label name")
    return name


def build_label_name_map(
    id_to_name: dict[int, str],
    cfg: CocoToCanonicalConfig,
    *,
    selected_category_ids: list[int],
) -> dict[int, str]:
    overrides = cfg.label_name_overrides or {}
    out: dict[int, str] = {}

    for cat_id in selected_category_ids:
        raw_name = id_to_name[cat_id]
        if cat_id in overrides:
            label_name = overrides[cat_id]
        elif cfg.sanitize_label_names:
            label_name = sanitize_label_name(raw_name)
        else:
            label_name = raw_name
        out[cat_id] = label_name

    # Guard against collisions after sanitization/overrides among selected classes only.
    inv: dict[str, list[int]] = defaultdict(list)
    for cat_id, label_name in out.items():
        inv[label_name].append(cat_id)
    collisions = {name: ids for name, ids in inv.items() if len(ids) > 1}
    if collisions:
        raise ValueError(
            "Label-name collisions after sanitization/overrides: "
            f"{collisions}. Use label_name_overrides in the YAML."
        )

    return out


# =============================================================================
# Conversion to canonical_schema v2
# =============================================================================


def coerce_attr_value(value: Any) -> str:
    """InstanceAnnotation.attributes is dict[str, str], so values are stringified."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def convert_coco_annotation_to_instance_annotation(
    ann: dict[str, Any],
    *,
    img_width: int,
    img_height: int,
    id_to_name: dict[int, str],
    label_name_by_id: dict[int, str],
    cfg: CocoToCanonicalConfig,
) -> tuple[InstanceAnnotation, dict[str, Any] | None] | None:
    category_id = int(ann["category_id"])
    class_name = label_name_by_id.get(category_id)
    if class_name is None:
        raise ValueError(f"Unknown or unselected category_id={category_id}")

    raw_class_name = id_to_name.get(category_id, str(category_id))

    xyxy = coco_xywh_to_clipped_xyxy(
        ann["bbox"],
        img_width=img_width,
        img_height=img_height,
    )

    if not is_valid_xyxy(xyxy):
        if cfg.skip_invalid_bboxes:
            return None
        raise ValueError(f"Invalid COCO bbox after clipping. ann_id={ann.get('id')} box={xyxy}")

    x1, y1, x2, y2 = xyxy
    bbox = BoundingBox(
        tl=Point(x=x1, y=y1),
        br=Point(x=x2, y=y2),
    )

    points = None
    if cfg.generate_center_points:
        points = [
            Point(
                x=round((x1 + x2) / 2),
                y=round((y1 + y2) / 2),
                is_positive=True,
            )
        ]

    attributes: dict[str, str] = {}
    if isinstance(ann.get("attributes"), dict):
        attributes = {str(k): coerce_attr_value(v) for k, v in ann["attributes"].items()}

    instance = InstanceAnnotation(
        instance_id=f"obj_{ann.get('id', 'unknown')}",
        label_id=category_id,
        label_name=class_name,
        bbox=bbox,
        points=points,
        mask=None,
        caption=None,
        attributes=attributes,
    )

    instance_metadata = None
    if cfg.include_instance_metadata:
        instance_metadata = {
            "coco_annotation_id": ann.get("id"),
            "coco_category_id": category_id,
            "raw_class_name": raw_class_name,
            "label_name": class_name,
            "coco_bbox_xywh": ann.get("bbox"),
            "bbox_xyxy_pixel": xyxy,
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd"),
        }

    return instance, instance_metadata


def make_sample_id(
    *,
    image: dict[str, Any],
    sequential_id: int,
    cfg: CocoToCanonicalConfig,
) -> str:
    strategy = cfg.sample_id_strategy

    if strategy == "image_id":
        base = str(image.get("id"))
    elif strategy == "sequential":
        base = f"{sequential_id:06d}"
    elif strategy == "file_stem":
        base = Path(str(image.get("file_name", sequential_id))).stem
    else:
        raise ValueError(f"Unsupported sample_id_strategy={strategy!r}")

    return f"{cfg.sample_id_prefix}{base}" if cfg.sample_id_prefix else base


def convert_coco_image_to_data_record(
    image: dict[str, Any],
    anns: list[dict[str, Any]],
    *,
    sequential_id: int,
    id_to_name: dict[int, str],
    label_name_by_id: dict[int, str],
    cfg: CocoToCanonicalConfig,
    source_file: str,
) -> tuple[DataRecord, int]:
    assert cfg.dataset_id is not None

    img_width = int(image["width"])
    img_height = int(image["height"])
    image_id = int(image["id"])

    instance_annotations: list[InstanceAnnotation] = []
    instances_metadata: list[dict[str, Any]] = []
    skipped_invalid = 0

    for ann in anns:
        converted = convert_coco_annotation_to_instance_annotation(
            ann,
            img_width=img_width,
            img_height=img_height,
            id_to_name=id_to_name,
            label_name_by_id=label_name_by_id,
            cfg=cfg,
        )

        if converted is None:
            skipped_invalid += 1
            continue

        instance, instance_metadata = converted
        instance_annotations.append(instance)
        if instance_metadata is not None:
            instances_metadata.append(instance_metadata)

    asset_metadata: dict[str, Any] = {
        "source_format": "COCO",
        "source_file": source_file,
        "split": cfg.split,
        "coco_image_id": image_id,
        "original_file_name": image.get("file_name"),
        "num_instances": len(instance_annotations),
        "num_invalid_annotations_skipped": skipped_invalid,
        "bbox_source": "coco_bbox_xywh",
        "bbox_coordinate_space": "pixel_xyxy",
        "points_source": "bbox_center_generated" if cfg.generate_center_points else None,
        "mask_source": None,
    }
    if instances_metadata:
        asset_metadata["instances_metadata"] = instances_metadata

    asset = ImageAsset(
        type="image",
        uri=str(image["file_name"]),
        size=(img_width, img_height),
        annotations=instance_annotations,
        metadata=asset_metadata,
    )

    sample = SISimpleDataSample(
        sample_type=cfg.sample_type,
        sample_id=make_sample_id(image=image, sequential_id=sequential_id, cfg=cfg),
        assets=[asset],
    )

    record = DataRecord(sample=sample)
    return record, skipped_invalid


# =============================================================================
# Dataset-level metadata
# =============================================================================


def build_label_info(
    *,
    selected_category_ids: list[int],
    label_name_by_id: dict[int, str],
    external_labels: dict[str, Any],
    require_label_descriptions: bool,
) -> dict[str, dict[str, Any]]:
    label_info: dict[str, dict[str, Any]] = {}

    print()
    for cat_id in selected_category_ids:
        label_name = label_name_by_id[cat_id]
        external = normalize_label_entry(external_labels.get(label_name))

        description = external.get("description")
        aliases = external.get("aliases", [])
        parent_label = external.get("parent_label")

        if require_label_descriptions and not description:
            raise ValueError(
                f"Missing label description for {label_name!r}. "
                "Add it to label_descriptions_yaml, label_descriptions, "
                "or label_description_overrides."
            )

        label_info[label_name] = {
            "label_id": cat_id,
            "label_name": label_name,
            "description": description,
            "aliases": aliases,
            "parent_label": parent_label,
        }

    return label_info


def build_dataset_info_record(
    *,
    cfg: CocoToCanonicalConfig,
    coco: dict[str, Any],
    id_to_name: dict[int, str],
    label_name_by_id: dict[int, str],
    selected_category_ids: list[int],
    external_labels: dict[str, Any],
) -> DatasetInfoRecord:
    assert cfg.dataset_id is not None

    coco_info = coco.get("info", {}) or {}

    label_info = build_label_info(
        selected_category_ids=selected_category_ids,
        label_name_by_id=label_name_by_id,
        external_labels=external_labels,
        require_label_descriptions=cfg.require_label_descriptions,
    )

    conversion_time = datetime.now().isoformat(timespec="seconds")

    info = DatasetInfo(
        dataset_id=cfg.dataset_id,
        description=cfg.description or coco_info.get("description") or "",
        annotation_info=AnnotationInfo(
            source_type=cfg.annotation_source,  # type: ignore[arg-type]
            quality=cfg.annotation_quality,  # type: ignore[arg-type]
            notes="Converted from COCO instance annotations.",
        ),
        domain=cfg.domain,
        split=cfg.split,
        date_collected=None,
        date_last_update=conversion_time,
        label_info=label_info,
        message_build_info=MessageBuildInfo(
            prompt_template_version=cfg.prompt_template_version,
            answer_format=cfg.answer_format,  # type: ignore[arg-type]
            normalization_factor=cfg.normalization_factor,
            metadata={
                "sample_type": cfg.sample_type,
                "target_encoding": "tag_bbox_list",
                "bbox_storage_coordinates": "pixel_xyxy",
                "bbox_message_coordinates": f"normalized_{cfg.normalization_factor}",
            },
        ),
        metadata={
            "source_format": "COCO",
            "schema_conversion": "coco_to_canonical_schema_v2",
            "date_converted": conversion_time,
            "selected_category_ids": selected_category_ids,
            "raw_category_names": {str(cat_id): id_to_name[cat_id] for cat_id in selected_category_ids},
            "label_name_by_id": {str(cat_id): label_name_by_id[cat_id] for cat_id in selected_category_ids},
            "label_descriptions_yaml": str(cfg.label_descriptions_yaml) if cfg.label_descriptions_yaml else None,
            "drop_empty": cfg.drop_empty,
            "skip_invalid_bboxes": cfg.skip_invalid_bboxes,
            "generate_center_points": cfg.generate_center_points,
        },
    )

    return DatasetInfoRecord(info=info)


# =============================================================================
# JSONL writer
# =============================================================================


def write_jsonl_record(f, obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    f.write("\n")


def convert_coco_to_canonical_jsonl(cfg: CocoToCanonicalConfig) -> dict[str, Any]:
    assert cfg.coco_json is not None
    assert cfg.output_jsonl is not None
    assert cfg.dataset_id is not None

    coco_path = Path(cfg.coco_json)
    output_path = Path(cfg.output_jsonl)

    coco = load_json(coco_path)
    id_to_name, _ = build_category_maps(coco)
    anns_by_image = group_annotations_by_image(coco)

    if not id_to_name:
        raise ValueError(f"No COCO categories found in {coco_path}")

    if cfg.include_categories is not None:
        unknown = sorted(cfg.include_categories - set(id_to_name.keys()))
        if unknown:
            raise ValueError(
                f"Unknown category ids in include_categories: {unknown}. "
                f"Available categories: {sorted(id_to_name.keys())}"
            )

    selected_category_ids = (
        sorted(cfg.include_categories)
        if cfg.include_categories is not None
        else sorted(id_to_name.keys())
    )

    label_name_by_id = build_label_name_map(
        id_to_name,
        cfg,
        selected_category_ids=selected_category_ids,
    )

    external_labels = load_external_label_info(cfg)

    dataset_info_record = build_dataset_info_record(
        cfg=cfg,
        coco=coco,
        id_to_name=id_to_name,
        label_name_by_id=label_name_by_id,
        selected_category_ids=selected_category_ids,
        external_labels=external_labels,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_coco_images = 0
    num_written_samples = 0
    num_empty_images_seen = 0
    num_instances_written = 0
    num_invalid_annotations_skipped = 0

    next_sample_idx = 1
    images = coco.get("images", [])

    with output_path.open("w", encoding="utf-8") as f:
        write_jsonl_record(f, dataset_info_record)

        for image in tqdm(
            images,
            desc=f"Processing {cfg.dataset_id}",
            total=len(images),
            disable=cfg.disable_pbar,
        ):
            num_coco_images += 1

            image_id = int(image["id"])
            anns = anns_by_image.get(image_id, [])

            if cfg.include_categories is not None:
                anns = [ann for ann in anns if int(ann["category_id"]) in cfg.include_categories]

            if not anns:
                num_empty_images_seen += 1
                if cfg.drop_empty:
                    continue

            record, skipped_invalid = convert_coco_image_to_data_record(
                image,
                anns,
                sequential_id=next_sample_idx,
                id_to_name=id_to_name,
                label_name_by_id=label_name_by_id,
                cfg=cfg,
                source_file=str(coco_path),
            )

            num_invalid_annotations_skipped += skipped_invalid

            # If all annotations were skipped and drop_empty=True, skip the sample.
            if cfg.drop_empty and len(record.sample.assets[0].annotations) == 0:
                continue

            write_jsonl_record(f, record)

            next_sample_idx += 1
            num_written_samples += 1
            num_instances_written += len(record.sample.assets[0].annotations)

    return {
        "output_path": str(output_path),
        "dataset_id": cfg.dataset_id,
        "split": cfg.split,
        "schema": "vh_loop.data_schema.v2",
        "sample_type": cfg.sample_type,
        "num_coco_images": num_coco_images,
        "num_written_samples": num_written_samples,
        "num_empty_images_seen": num_empty_images_seen,
        "num_instances_written": num_instances_written,
        "num_invalid_annotations_skipped": num_invalid_annotations_skipped,
        "drop_empty": cfg.drop_empty,
        "selected_categories": {str(cat_id): id_to_name[cat_id] for cat_id in selected_category_ids},
        "canonical_label_names": {str(cat_id): label_name_by_id[cat_id] for cat_id in selected_category_ids},
        "label_descriptions_yaml": str(cfg.label_descriptions_yaml) if cfg.label_descriptions_yaml else None,
    }


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert COCO instance annotations to canonical_schema v2 JSONL."
    )

    parser.add_argument("--config", default=None, type=Path, help="Optional YAML config file.")
    parser.add_argument("--coco-json", default=None, type=Path, help="Path to COCO annotation JSON.")
    parser.add_argument("--output-jsonl", default=None, type=Path, help="Output canonical JSONL path.")
    parser.add_argument("--dataset-id", default=None, type=str, help="Dataset identifier.")
    parser.add_argument("--split", default=None, type=str, help="Optional split name.")
    parser.add_argument("--description", default=None, type=str, help="Optional dataset description.")
    parser.add_argument("--domain", default=None, type=str, help="Optional domain, e.g. waste-sorting.")

    parser.add_argument("--annotation-source", default=None, type=str, help="human, ai, ai_human_reviewed, synthetic, imported, web_scrap.")
    parser.add_argument("--annotation-quality", default=None, type=str, help="raw, weak, auto, reviewed, gold.")

    parser.add_argument("--sample-type", default=None, type=str, help="Currently only si_simple_data.")
    parser.add_argument("--prompt-template-version", default=None, type=str)
    parser.add_argument("--answer-format", default=None, type=str, help="Currently tag_bbox_list.")
    parser.add_argument("--normalization-factor", default=None, type=int)

    parser.add_argument("--include-categories", default=None, type=str, help="Comma-separated category ids to include, or a YAML list. By default, all categories are included.")

    parser.add_argument("--disable-pbar", dest="disable_pbar", action="store_true", default=None)

    parser.add_argument("--drop-empty", dest="drop_empty", action="store_true", default=None)

    parser.add_argument("--disable-skip-invalid-bboxes", dest="skip_invalid_bboxes", action="store_false")

    parser.add_argument("--include-instance-metadata", dest="include_instance_metadata", action="store_true", default=None)

    parser.add_argument("--disable-generate-center-points", dest="generate_center_points", action="store_false")

    parser.add_argument("--disable-sanitize-label-names", dest="sanitize_label_names", action="store_false")

    parser.add_argument("--label-descriptions-yaml", default=None, type=Path)
    parser.add_argument("--require-label-descriptions", dest="require_label_descriptions", action="store_true", default=None)

    parser.add_argument("--sample-id-prefix", default=None, type=str)
    parser.add_argument("--sample-id-strategy", default=None, choices=["image_id", "sequential", "file_stem"])

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    configs = build_configs_from_yaml_and_cli(args)

    summaries = []
    for cfg in configs:
        summaries.append(convert_coco_to_canonical_jsonl(cfg))

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()