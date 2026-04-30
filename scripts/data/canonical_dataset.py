# src/data/canonical_dataset.py

from pathlib import Path
import json
from torch.utils.data import Dataset
from scripts.data.schema import VLMSample


class CanonicalVLMDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.samples = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(VLMSample.model_validate_json(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> VLMSample:
        return self.samples[idx]