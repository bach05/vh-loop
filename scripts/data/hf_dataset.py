from typing import Any, Optional, Callable

from datasets import Dataset as HFDataset
import hashlib

from scripts.data.schema import VLMSample
from scripts.data.manifest_utils import iter_vlm_samples_from_manifest
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from scripts.core.constants import SUPPORTED_DATASET_SCHEMAS, SFT_CONVERTER_VERSION

import numpy as np

#Convert VLMSample to a conversation
def sample_to_messages(
    sample: VLMSample,
    dataset_schema: str = "conversational",
    dataset_root: str | None = None,
    include_target: bool = True,
) -> dict[str, Any]:

    messages = []
    images = []

    for msg in sample.messages:
        content = []

        for part in msg.content:
            if part.type == "text":
                content.append(
                    {
                        "type": "text",
                        "text": part.text,
                    }
                )

            elif part.type == "image":
                image_id = part.image_id
                img_ref = sample.images.get(image_id, None)

                if img_ref is None:
                    raise ValueError(f"Image ID {image_id} not found in sample images")

                img_path = img_ref.path
                full_image_path = Path(dataset_root) / img_path if dataset_root else Path(img_path)
                full_image_path = str(full_image_path)

                content.append(
                    {
                        "type": "image",
                        "path": full_image_path,
                    }
                )
                images.append(full_image_path)

            else:
                raise ValueError(f"Unsupported content part type: {part.type}")

        role = msg.role.value if hasattr(msg.role, "value") else msg.role

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    if dataset_schema == "conversational":
        if include_target:
            if sample.target is None or sample.target.text is None:
                raise ValueError(
                    f"Sample {sample.sample_id} has no target, but include_target=True"
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": sample.target.text,
                        }
                    ],
                }
            )

        return {
            "messages": messages,
            "images": images,
        }

    elif dataset_schema == "prompt-completion":
        if include_target:
            if sample.target is None or sample.target.text is None:
                raise ValueError(
                    f"Sample {sample.sample_id} has no target, but include_target=True"
                )

            return {
                "prompt": messages,
                "completion": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": sample.target.text,
                            }
                        ],
                    }
                ],
                "images": images,
            }

        # For inference, return prompt-only format.
        # Adjust this if your adapter expects another key.
        return {
            "messages": messages,
            "images": images,
        }

    else:
        raise ValueError(
            f"Unsupported dataset schema: {dataset_schema}, "
            f"supported schemas: {SUPPORTED_DATASET_SCHEMAS}"
        )


def iter_sft_rows_from_manifest(manifest_path, dataset_schema="conversational", dataset_root=None):
    for sample in iter_vlm_samples_from_manifest(
        manifest_path,
        attach_dataset_info=False,
    ):
        yield sample_to_messages(
            sample,
            dataset_schema=dataset_schema,
            dataset_root=dataset_root,
            include_target=True,
        )

def manifest_fingerprint(
    manifest_path: str | Path,
    dataset_schema: str = "conversational",
    dataset_root: Optional[str | Path] = None,
    *,
    converter_version: str = SFT_CONVERTER_VERSION,
    extra: Optional[dict] = None,
) -> str:
    path = Path(manifest_path)
    stat = path.stat()

    import json

    payload = {
        # Data source identity
        "manifest_path": str(path.resolve()),
        "manifest_mtime_ns": stat.st_mtime_ns,
        "manifest_size": stat.st_size,

        # Conversion options that change generated rows
        "dataset_schema": dataset_schema,
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,

        # Manual invalidation when sample_to_messages / parsing logic changes
        "converter_version": converter_version,

        # Optional project/config-specific invalidation
        "extra": extra or {},
    }

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def canonical_manifest_to_hf_sft(
    manifest_path: str | Path,
    dataset_schema: str = 'conversational',
    dataset_root: Optional[str] | Optional[Path] = None,
    cache_dir: Optional[str | Path] = None,
    fingerprint_extra: Optional[dict] = None,
) -> HFDataset:

    fingerprint = manifest_fingerprint(
        manifest_path=manifest_path,
        dataset_schema=dataset_schema,
        dataset_root=dataset_root,
        extra=fingerprint_extra,
    )

    return HFDataset.from_generator(
        lambda: iter_sft_rows_from_manifest(manifest_path, dataset_schema=dataset_schema, dataset_root=dataset_root),
        fingerprint=fingerprint, #guarantee to recompute the dataset when path changes or file is updated
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

def _apply_transform(transform: Optional[Callable], img: np.ndarray):
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
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        images = []

        for image_path in example["images"]:
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = _apply_transform(self.transform, np.array(image))

            images.append(image)

        example["images"] = images
        return example