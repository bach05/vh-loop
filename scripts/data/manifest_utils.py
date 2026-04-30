import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field
from typing import Literal, Optional

from scripts.data.schema import VLMSample, DatasetInfo, DatasetInfoRecord

def read_manifest_info(path: str | Path) -> DatasetInfo:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise RuntimeError(f"Empty manifest: {path}")

    obj = json.loads(first_line)

    if obj.get("record_type") != "dataset_info":
        raise RuntimeError(
            f"Manifest {path} must start with a dataset_info record."
        )

    return DatasetInfoRecord.model_validate(obj).info

def iter_vlm_samples_from_manifest(
    path: str | Path,
    *,
    attach_dataset_info: bool = False,
) -> Iterator[VLMSample]:
    path = Path(path)
    dataset_info = read_manifest_info(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            record_type = obj.get("record_type")

            if line_no == 1:
                if record_type != "dataset_info":
                    raise RuntimeError(
                        f"First line of {path} must be dataset_info."
                    )
                continue

            if record_type != "sample":
                raise RuntimeError(
                    f"Unsupported record_type={record_type!r} at line {line_no} of {path}"
                )

            obj.pop("record_type", None)

            if attach_dataset_info:
                obj.setdefault("metadata", {})
                obj["metadata"]["dataset_info"] = dataset_info.model_dump()

            try:
                yield VLMSample.model_validate(obj)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse sample at line {line_no} of {path}"
                ) from e

