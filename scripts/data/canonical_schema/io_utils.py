from __future__ import annotations

"""Utilities for reading schema-v2 JSONL manifests.

Expected JSONL layout:

Line 1:
    {"record_type": "dataset_info", "schema_version": "...", "info": {...}}

Lines 2+:
    {"record_type": "sample", "sample": {"sample_type": "...", ...}}

The concrete sample object is deserialized through DataRecord.sample, which
uses the Pydantic discriminated union defined in scripts.data.schema.sample.registry.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .dataset_header import DatasetInfo
from .records import DataRecord, DatasetInfoRecord
from .sample.base import DataSample

def _load_json_line(line: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at line {line_no} of {path}") from exc

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Manifest record at line {line_no} of {path} must be a JSON object, "
            f"got {type(obj).__name__}"
        )
    return obj


def read_dataset_info(path: str | Path) -> DatasetInfo:
    """Read and validate the first dataset_info record from a JSONL dataset."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = _load_json_line(line, path=path, line_no=line_no)

            if obj.get("record_type") != "dataset_info":
                raise RuntimeError(
                    f"Manifest {path} must start with a dataset_info record. "
                    f"Found record_type={obj.get('record_type')!r} at line {line_no}."
                )

            try:
                return DatasetInfoRecord.model_validate(obj).info
            except Exception as exc:
                raise RuntimeError(f"Failed to parse dataset_info at line {line_no} of {path}") from exc

    raise RuntimeError(f"Empty manifest: {path}")

def iter_data_records(path: str | Path) -> Iterator[DataRecord]:
    """Yield parsed DataRecord objects from a schema-v2 JSONL manifest.

    The first non-empty line must be the dataset_info record. All following
    non-empty records must have record_type="sample".
    """
    path = Path(path)
    seen_info = False

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = _load_json_line(line, path=path, line_no=line_no)
            record_type = obj.get("record_type")

            if not seen_info:
                if record_type != "dataset_info":
                    raise RuntimeError(
                        f"First non-empty line of {path} must be dataset_info, "
                        f"got {record_type!r} at line {line_no}."
                    )
                # Validate once so malformed headers fail early.
                try:
                    DatasetInfoRecord.model_validate(obj)
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse dataset_info at line {line_no} of {path}") from exc
                seen_info = True
                continue

            if record_type != "sample":
                raise RuntimeError(
                    f"Unsupported record_type={record_type!r} at line {line_no} of {path}. "
                    "Expected 'sample'."
                )

            try:
                yield DataRecord.model_validate(obj)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse sample record at line {line_no} of {path}") from exc

    if not seen_info:
        raise RuntimeError(f"Manifest {path} has no dataset_info record.")

def iter_samples_from_jsonl(path: str | Path) -> Iterator[DataSample]:
    """Yield concrete DataSample objects from a schema-v2 JSONL manifest."""
    for record in iter_data_records(path):
        yield record.sample

def write_jsonl_to_record(f, obj: dict[str, Any]) -> None:
    """Write one compact JSONL record to an already-open text file."""
    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    f.write("\n")
