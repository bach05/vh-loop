# src/data/canonical_dataset.py
from __future__ import annotations

"""PyTorch dataset wrappers for vh_loop canonical_schema v2 manifests.

The canonical JSONL format is expected to contain:

Line 1:
    DatasetInfoRecord

Lines 2+:
    DataRecord(sample=SampleUnion(...))

This module intentionally keeps the canonical dataset independent from the
Hugging Face SFT conversion layer. The dataset returns canonical DataSample
objects. InferenceCanonicalDataset converts them to prompt rows by calling the
sample's own sample_to_message() method.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from scripts.data.canonical_schema import DatasetInfo
from scripts.data.canonical_schema.records import DatasetInfoRecord, DataRecord
from scripts.data.canonical_schema.sample import DataSample, PromptingSchema

from scripts.data.hf_dataset import _apply_transform

class CanonicalDataset(Dataset):
    """Load a canonical_schema v2 JSONL manifest as canonical samples.

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL manifest. The first non-empty line must be a
        DatasetInfoRecord. Subsequent non-empty lines must be DataRecord rows.

    Attributes
    ----------
    info:
        Parsed DatasetInfo object.
    samples:
        List of parsed DataSample objects. Concrete subclasses are created
        through the Pydantic discriminated union inside DataRecord.
    """

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.info: DatasetInfo | None = None
        self.samples: list[DataSample] = []
        self.records: list[DataRecord] = []

        self._load_data()

    def _load_data(self) -> None:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.jsonl_path}")

        seen_info = False

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Invalid JSON at line {line_no} of {self.jsonl_path}"
                    ) from e

                record_type = obj.get("record_type")

                if record_type == "dataset_info":
                    if seen_info:
                        raise RuntimeError(
                            f"Multiple dataset_info records found in {self.jsonl_path}; "
                            f"second occurrence at line {line_no}."
                        )
                    try:
                        self.info = DatasetInfoRecord.model_validate(obj).info
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to parse dataset_info at line {line_no} of "
                            f"{self.jsonl_path}"
                        ) from e
                    seen_info = True
                    continue

                if not seen_info:
                    raise RuntimeError(
                        f"The JSONL file {self.jsonl_path} must start with a "
                        f"dataset_info record before sample records."
                    )

                if record_type != "sample":
                    raise RuntimeError(
                        f"Unsupported record_type={record_type!r} at line {line_no} "
                        f"of {self.jsonl_path}"
                    )

                try:
                    record = DataRecord.model_validate(obj)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse sample record at line {line_no} of "
                        f"{self.jsonl_path}"
                    ) from e

                self.records.append(record)
                self.samples.append(record.sample)

        if self.info is None:
            raise RuntimeError(f"The JSONL file {self.jsonl_path} has no dataset_info record.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]

    def get_record(self, idx: int) -> DataRecord:
        """Return the original DataRecord wrapper for a sample."""
        return self.records[idx]

    def get_dataset_info(self) -> DatasetInfo:
        """Return the DatasetInfo object parsed from the manifest."""
        if self.info is None:
            raise RuntimeError(f"Dataset info not loaded for {self.jsonl_path}")
        return self.info


class InferenceCanonicalDataset(Dataset):
    """Prompt-only wrapper for canonical_schema v2 datasets.

    The wrapper calls sample.sample_to_message(..., include_target=False), then
    opens the referenced image paths and replaces example["images"] with loaded
    PIL images or transformed image objects.
    """

    def __init__(
        self,
        canonical_dataset: CanonicalDataset,
        *,
        prompting_schema: PromptingSchema = "conversational",
        dataset_root: str | Path | None = None,
        transform: Optional[Callable] = None,
    ):
        self.canonical_dataset = canonical_dataset
        self.prompting_schema = prompting_schema
        self.dataset_root = dataset_root
        self.transform = transform

        dataset_info = canonical_dataset.get_dataset_info()
        self.dataset_info: DatasetInfo = dataset_info

    def __len__(self) -> int:
        return len(self.canonical_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.canonical_dataset[idx]

        if not hasattr(sample, "sample_to_message"):
            raise TypeError(
                f"Expected a DataSample-like object with sample_to_message(), got {type(sample)}"
            )

        example = sample.sample_to_message(
            dataset_info=self.dataset_info,
            prompting_schema=self.prompting_schema,
            include_target=False,
            dataset_root=self.dataset_root,
        )

        image_paths = list(example.get("images", []))
        images = []

        for image_path in image_paths:
            # Remote URLs are kept as strings because PIL.Image.open cannot open
            # them directly without a separate downloader.
            if str(image_path).startswith(("http://", "https://", "s3://", "gs://")):
                images.append(image_path)
                continue

            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = _apply_transform(self.transform, np.array(image))

            images.append(image)

        example["image_paths"] = image_paths
        example["images"] = images
        example["_sample_idx"] = idx
        example["_sample_id"] = getattr(sample, "sample_id", None)

        return example