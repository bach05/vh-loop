from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
import hashlib

from scripts.data.schema import VLMSample
from scripts.data.manifest_utils import iter_vlm_samples_from_manifest

#Convert VLMSample to a conversation
def sample_to_messages(sample: VLMSample) -> (int, list[dict[str, Any]]):

    messages = []

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

                if image_id == "query":
                    img_path = sample.query_image.path
                else:
                    img_ref = sample.images.get(image_id, None)
                    if img_ref is None:
                        raise ValueError(f"Image ID {image_id} not found in sample images")
                    else:
                        img_path = img_ref.path

                content.append(
                    {
                        "type": "image",
                        "path": img_path,
                    }
                )

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
    }

    return ret_dict


def iter_sft_rows_from_manifest(manifest_path):

    for sample in iter_vlm_samples_from_manifest(
        manifest_path,
        attach_dataset_info=False,
    ):
        yield sample_to_messages(sample)


def manifest_fingerprint(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    stat = path.stat()

    raw = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:sft-v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def canonical_manifest_to_hf_sft(
    manifest_path: str | Path,
) -> HFDataset:
    return HFDataset.from_generator(
        lambda: iter_sft_rows_from_manifest(manifest_path),
        fingerprint=manifest_fingerprint(manifest_path), #guarantee to recompute the dataset when path changes or file is updated
    )