# src/data/canonical_dataset.py
import json

from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from scripts.data.hf_dataset import sample_to_messages, _apply_transform
from data.canonical_schema.schema import VLMSample, SampleRecord, DatasetInfo


class CanonicalVLMDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.samples: list[VLMSample] = []
        self.info: DatasetInfo | None = None

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                record_type = record.get("record_type", "sample")

                if record_type == "dataset_info":
                    self.info = DatasetInfo.model_validate(record["info"])
                    continue

                if record_type != "sample":
                    raise ValueError(f"Unsupported record_type: {record_type}")

                self.samples.append(SampleRecord.model_validate(record))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> VLMSample:
        return self.samples[idx]



class InferenceVLMDataset(Dataset):
    def __init__(
        self,
        canonical_dataset: Dataset,
        *,
        dataset_schema: str = "conversational",
        dataset_root: str | Path | None = None,
        transform: Optional[Callable] = None,
    ):
        self.canonical_dataset = canonical_dataset
        self.dataset_schema = dataset_schema
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self):
        return len(self.canonical_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.canonical_dataset[idx]

        example = sample_to_messages(
            sample,
            dataset_schema=self.dataset_schema,
            dataset_root=self.dataset_root,
            include_target=False,
        )

        images = []
        for image_path in example["images"]:
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = _apply_transform(self.transform, np.array(image))

            images.append(image)

        example["images"] = images
        example["_sample_idx"] = idx

        return example