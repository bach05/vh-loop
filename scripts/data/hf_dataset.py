from typing import Any, Optional, Callable

from datasets import Dataset as HFDataset
import hashlib

from scripts.data.schema import VLMSample
from scripts.data.manifest_utils import iter_vlm_samples_from_manifest
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import numpy as np

#Convert VLMSample to a conversation
def sample_to_messages(sample: VLMSample, dataset_root: str = None) -> (int, list[dict[str, Any]]):

    messages = []
    images = []

    # Parse messages
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
                else:
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

    #Add target
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

    ret_dict = \
    {
        "messages": messages,
        "images": images,
    }

    return ret_dict


def iter_sft_rows_from_manifest(manifest_path, dataset_root=None):

    for sample in iter_vlm_samples_from_manifest(
        manifest_path,
        attach_dataset_info=False,
    ):
        yield sample_to_messages(sample, dataset_root)


def manifest_fingerprint(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    stat = path.stat()

    raw = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:sft-v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def canonical_manifest_to_hf_sft(
    manifest_path: str | Path,
    dataset_root: str | Path = None,
) -> HFDataset:
    return HFDataset.from_generator(
        lambda: iter_sft_rows_from_manifest(manifest_path, dataset_root),
        fingerprint=manifest_fingerprint(manifest_path), #guarantee to recompute the dataset when path changes or file is updated
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