from __future__ import annotations

import csv
import json
import pandas as pd
from pathlib import Path
from typing import Any, Optional, Iterable

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from scripts.data.canonical_schema import DatasetInfo, DatasetInfoRecord, DataRecord, InstanceAnnotation
from scripts.data.canonical_schema.assets import ImageAsset
from scripts.data.canonical_schema.sample.base import DataSample


# ---------------------------------------------------------------------------
# Schema asset helpers
# ---------------------------------------------------------------------------

def get_image_assets(sample: Optional[DataSample]) -> list[ImageAsset]:
    """ Return all ImageAsset objects from *sample*, or an empty list. """
    if sample is None:
        return []
    assets = getattr(sample, "assets", None)
    if not assets:
        return []
    return [a for a in assets if isinstance(a, ImageAsset)]


def get_primary_image_asset(sample: Optional[DataSample]) -> Optional[ImageAsset]:
    """ Return the first ImageAsset from *sample*, or None. """
    assets = get_image_assets(sample)
    return assets[0] if assets else None


def extract_bbox_annotations(sample: Optional[DataSample]) -> list[InstanceAnnotation]:
    """ Return every InstanceAnnotation that has a bounding-box across all image assets. """
    if sample is None:
        return []
    return [
        ann
        for asset in get_image_assets(sample)
        for ann in asset.annotations
        if ann.bbox is not None
    ]


def resolve_image_path(sample: DataSample, image_root: Optional[str | Path]) -> Optional[Path]:
    """ Resolve the filesystem path to the primary image of *sample*. """
    asset = get_primary_image_asset(sample)
    if asset is None:
        return None
    resolved = asset.resolve_path(
        to_absolute_path(str(image_root)) if image_root is not None else None
    )
    return Path(resolved)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_canonical_samples(jsonl_path: str | Path) -> tuple[dict[str, DataSample], DatasetInfo]:
    """
    Load a canonical_schema v2 JSONL file.

    The file is expected to contain exactly one ``dataset_info`` record
    (usually the first line) followed by any number of ``sample`` records.

    Returns:
        samples:      Dict mapping ``sample_id`` → ``DataSample``.
        dataset_info: The ``DatasetInfo`` extracted from the header record.

    Raises:
        RuntimeError: If parsing fails on any line, or if no
                      ``dataset_info`` record is present.
    """
    path = Path(to_absolute_path(str(jsonl_path)))
    samples: dict[str, DataSample] = {}
    dataset_info: DatasetInfo | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            record_type = record.get("record_type", "sample")

            try:
                if record_type == "dataset_info":
                    dataset_info = DatasetInfoRecord.model_validate(record).info
                    continue

                if record_type == "sample":
                    data_record = DataRecord.model_validate(record)
                    sample = data_record.sample
                    samples[str(sample.sample_id)] = sample
                    continue

                raise ValueError(f"Unsupported record_type={record_type!r}")

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse record at line {line_idx} of {path}: {exc}"
                ) from exc

    if dataset_info is None:
        raise RuntimeError(
            f"Manifest {path} does not contain a dataset_info record."
        )

    return samples, dataset_info


# ---------------------------------------------------------------------------
# Prediction-file discovery
# ---------------------------------------------------------------------------

def resolve_prediction_files(cfg: DictConfig) -> list[tuple[str, Path]]:
    """
    Collect (name, path) pairs from the Hydra config.

    Sources (both may be combined; duplicates are removed):

    * ``cfg.predictions`` – explicit list of paths or
      ``{path: ..., name: ...}`` dicts.
    * ``cfg.predictions_dir`` + ``cfg.patterns`` – glob patterns applied to a
      directory.
    """
    found: list[tuple[str, Path]] = []

    for item in cfg.get("predictions", []) or []:
        if isinstance(item, str):
            p = Path(to_absolute_path(item))
            found.append((p.stem, p))
        else:
            p = Path(to_absolute_path(str(item["path"])))
            found.append((str(item.get("name", p.stem)), p))

    predictions_dir = cfg.get("predictions_dir", None)
    if predictions_dir is not None:
        pred_dir = Path(to_absolute_path(str(predictions_dir)))
        for pattern in cfg.get("patterns", ["*.jsonl"]):
            for p in sorted(pred_dir.glob(pattern)):
                found.append((p.stem, p))

    # Deduplicate by resolved path, preserving last-seen name.
    dedup: dict[Path, tuple[str, Path]] = {}
    for name, p in found:
        dedup[p.resolve()] = (name, p)

    return list(dedup.values())


# ---------------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------------

def load_and_stack_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path)
        df["_source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("No input CSV files were provided.")

    merged = pd.concat(frames, axis=0, ignore_index=True)

    required = {"model", "threshold"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(
            f"Merged CSV is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(merged.columns)}"
        )

    merged["model"] = merged["model"].astype(str)
    merged["threshold"] = pd.to_numeric(merged["threshold"], errors="coerce")

    if merged["threshold"].isna().any():
        bad_rows = merged[merged["threshold"].isna()]
        raise ValueError(
            "Some rows have invalid threshold values:\n"
            f"{bad_rows[['model', 'threshold', '_source_file']].head()}"
        )

    return merged


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """ Write *rows* as a CSV to *path*, creating parent directories as needed. """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)