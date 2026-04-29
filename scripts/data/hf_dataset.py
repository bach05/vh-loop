from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from src.data.schema import VLMSample

#Convert VLMSample to a conversation
def sample_to_messages(sample: VLMSample) -> list[dict[str, Any]]:
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
                content.append(
                    {
                        "type": "image",
                        "path": part.image.path,
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

    return messages


def iter_sft_rows_from_manifest(
    manifest_path: str | Path,
):
    manifest_path = Path(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                sample = VLMSample.model_validate_json(line)

                yield {
                    "sample_id": sample.sample_id,
                    "messages": sample_to_messages(sample),
                }

            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert sample at line {line_idx} "
                    f"of manifest {manifest_path}"
                ) from e


def canonical_manifest_to_hf_sft(
    manifest_path: str | Path,
) -> HFDataset:
    return HFDataset.from_generator(
        lambda: iter_sft_rows_from_manifest(manifest_path)
    )