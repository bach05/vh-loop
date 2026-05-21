from __future__ import annotations

"""Hugging Face dataset conversion utilities for schema-v2 manifests.

This module no longer assumes the old VLMSample structure. Instead, it reads
DatasetInfo once from the manifest, parses each DataRecord into a concrete
DataSample, and delegates prompt/target construction to
DataSample.sample_to_message().
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional
import hashlib
import json

from datasets import Dataset as HFDataset
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from scripts.core.constants import SFT_CONVERTER_VERSION
except Exception:  # pragma: no cover - fallback for lightweight tests
    SFT_CONVERTER_VERSION = "schema_v2_sft_converter_v1"

from scripts.data.canonical_schema.io_utils import read_dataset_info, iter_samples_from_jsonl
from scripts.data.canonical_schema.dataset_header import DatasetInfo
from scripts.data.canonical_schema.sample.base import DataSample
from scripts.data.canonical_schema.sample.single_image import PromptingSchema

def sample_to_messages(
    sample: DataSample,
    dataset_info: DatasetInfo,
    *,
    prompting_schema: PromptingSchema = "conversational",
    dataset_root: str | Path | None = None,
    include_target: bool = True,
) -> dict[str, Any]:
    """Convert one schema-v2 DataSample into a training/inference row.

    The concrete sample class owns the message construction logic. This wrapper
    exists so HF conversion code can remain independent from sample subclasses.
    """
    return sample.sample_to_message(
        dataset_info=dataset_info,
        prompting_schema=prompting_schema,
        include_target=include_target,
        dataset_root=dataset_root,
    )

def iter_data_rows_from_jsonl(
    jsonl_path: str | Path,
    *,
    prompting_schema: PromptingSchema = "conversational",
    dataset_root: str | Path | None = None,
    include_target: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield HF-ready SFT rows from a schema-v2 JSONL manifest."""
    dataset_info = read_dataset_info(jsonl_path)

    for sample in iter_samples_from_jsonl(jsonl_path):
        yield sample_to_messages(
            sample,
            dataset_info=dataset_info,
            prompting_schema=prompting_schema,
            dataset_root=dataset_root,
            include_target=include_target,
        )


def manifest_fingerprint(
    jsonl_path: str | Path,
    *,
    prompting_schema: PromptingSchema = "conversational",
    dataset_root: Optional[str | Path] = None,
    include_target: bool = True,
    converter_version: str = SFT_CONVERTER_VERSION,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Build a deterministic HF cache fingerprint for manifest conversion.

    The fingerprint includes both file identity and conversion options that
    affect the generated rows, including dataset-level message_build_info.
    """
    path = Path(jsonl_path)
    stat = path.stat()

    dataset_info = read_dataset_info(path)

    payload = {
        "jsonl_path": str(path.resolve()),
        "manifest_mtime_ns": stat.st_mtime_ns,
        "manifest_size": stat.st_size,
        "schema_dataset_id": dataset_info.dataset_id,
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        "converter_version": converter_version,
        "extra": extra or {},
    }

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def canonical_jsonl_to_hf_dataset(
    jsonl_path: str | Path,
    *,
    prompting_schema: PromptingSchema = "conversational",
    dataset_root: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    include_target: bool = True,
    fingerprint_extra: Optional[dict[str, Any]] = None,
) -> HFDataset:
    """Create an HF Dataset from a schema-v2 canonical JSONL manifest.

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL manifest file.
    prompting_schema:
        Preferred name for the output row format, e.g. "conversational" or
        "prompt-completion".
    dataset_root:
        Optional root directory for resolving relative asset paths in the jsonl file.
    cache_dir:
        Optional directory for Hugging Face to store cached datasets.
    include_target:
        True for SFT training rows. False for inference/prompt-only rows.
    fingerprint_extra:
        Optional dictionary of extra parameters to include in the cache fingerprint.
    """

    fingerprint = manifest_fingerprint(
        jsonl_path=jsonl_path,
        prompting_schema=prompting_schema,
        dataset_root=dataset_root,
        include_target=include_target,
        extra=fingerprint_extra,
    )

    return HFDataset.from_generator(
        lambda: iter_data_rows_from_jsonl(
            jsonl_path,
            prompting_schema=prompting_schema,
            dataset_root=dataset_root,
            include_target=include_target,
        ),
        fingerprint=fingerprint,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def _apply_transform(transform: Optional[Callable], img: np.ndarray):
    """Apply either an Albumentations-style or PIL-style transform."""
    if transform is None:
        return img

    try:
        out = transform(image=img)
        if isinstance(out, dict):
            if "image" in out:
                return out["image"]
            if len(out) == 1:
                return next(iter(out.values()))
        return out
    except TypeError:
        pil_img = Image.fromarray(img)
        out = transform(pil_img)
        if isinstance(out, dict):
            if "image" in out:
                return out["image"]
            if len(out) == 1:
                return next(iter(out.values()))
        return out


class TransformedVLMHFDataset(Dataset):
    """PyTorch wrapper that loads image paths from an HF row into images.

    This class keeps the row structure produced by sample_to_message(), but
    replaces example["images"] paths with loaded RGB PIL images or transformed
    arrays/tensors.

    Note: the transform is applied to the image only. Bbox and points are normalized in the prompt construction.
    """

    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = dict(self.hf_dataset[idx])

        images = []
        for image_path in example.get("images", []):
            # PIL cannot open remote URLs directly; keep the error explicit.
            if isinstance(image_path, str) and image_path.startswith(("http://", "https://", "s3://", "gs://")):
                raise ValueError(
                    f"TransformedVLMHFDataset cannot load remote image URI directly: {image_path!r}. "
                    "Download/cache the image first or provide a custom dataset wrapper."
                )

            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = _apply_transform(self.transform, np.array(image))

            images.append(image)

        example["images"] = images
        return example
