# scripts/training/dataset_statistics.py

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from pathlib import Path


def select_dataset_indices(
    n_total: int,
    max_samples: Optional[int],
) -> list[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= n_total:
        return list(range(n_total))

    # Uniform sampling over the full dataset
    step = n_total / max_samples
    return [min(int(i * step), n_total - 1) for i in range(max_samples)]


def batched(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def inspect_sample_structure(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample.get("messages", [])
    images = sample.get("images", {})
    target = sample.get("target", None)

    num_messages = len(messages) if isinstance(messages, list) else 0
    num_text_parts = 0
    num_message_image_parts = 0
    num_image_assets = len(images) if isinstance(images, dict) else 0

    labels = []
    num_target_instances = 0

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", [])

            if isinstance(content, str):
                num_text_parts += 1

            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type")

                    if part_type == "text":
                        num_text_parts += 1

                    elif part_type in {"image", "image_url"}:
                        num_message_image_parts += 1

    if isinstance(target, dict):
        instances = target.get("instances", [])

        if isinstance(instances, list):
            num_target_instances = len(instances)

            for inst in instances:
                if not isinstance(inst, dict):
                    continue

                label = inst.get("label", None)
                if label is not None:
                    labels.append(str(label))

    return {
        "num_messages": num_messages,
        "num_text_parts": num_text_parts,
        "num_image_assets": num_image_assets,
        "num_message_image_parts": num_message_image_parts,
        "num_target_instances": num_target_instances,
        "labels": labels,
    }

def build_text_from_sample(
    sample: dict[str, Any],
    processor,
    tokenizer,
) -> Optional[str]:
    """
    Convert several common SFT dataset schemas into a single training string.
    """

    # ------------------------------------------------------------------
    # 1. Chat/messages format
    # ------------------------------------------------------------------
    if "messages" in sample:
        messages = sample["messages"]

        if hasattr(processor, "apply_chat_template"):
            try:
                return processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass

        return flatten_messages(messages)

    # ------------------------------------------------------------------
    # 2. Prompt/completion format
    # ------------------------------------------------------------------
    if "prompt" in sample and "completion" in sample:
        return f"{sample['prompt']}{sample['completion']}"

    if "prompt" in sample and "response" in sample:
        return f"{sample['prompt']}{sample['response']}"

    if "prompt" in sample and "answer" in sample:
        return f"{sample['prompt']}{sample['answer']}"

    # ------------------------------------------------------------------
    # 3. Instruction tuning format
    # ------------------------------------------------------------------
    if "instruction" in sample:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = (
            sample.get("output")
            or sample.get("response")
            or sample.get("answer")
            or ""
        )

        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output_text}"
            )

        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )

    # ------------------------------------------------------------------
    # 4. Canonical target text fallback
    # ------------------------------------------------------------------
    if "query" in sample and "target" in sample:
        query = sample.get("query", "")
        target = sample.get("target", "")

        if isinstance(target, dict):
            target_text = target.get("text", "")
        else:
            target_text = str(target)

        return f"{query}{target_text}"

    if "target" in sample and isinstance(sample["target"], dict):
        target_text = sample["target"].get("text", None)
        if target_text is not None:
            return str(target_text)

    # ------------------------------------------------------------------
    # 5. Plain text format
    # ------------------------------------------------------------------
    if "text" in sample:
        return sample["text"]

    return None

def flatten_messages(messages: list[dict[str, Any]]) -> str:
    chunks = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        chunks.append(f"{role}:")

        if isinstance(content, str):
            chunks.append(content)

        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type")

                if part_type == "text":
                    chunks.append(part.get("text", ""))

                elif part_type in {"image", "image_url"}:
                    chunks.append("<image>")

    return "\n".join(chunks)

def extract_images_from_sample(sample: dict[str, Any]) -> list[Any]:
    """
    Extract images from common VLM dataset schemas.

    Supports:
    - sample["image"]
    - sample["images"]
    - sample["messages"][...]["content"] image parts
    - canonical image assets like {"path": "..."}
    """

    images = []

    # ------------------------------------------------------------------
    # sample["image"]
    # ------------------------------------------------------------------
    if "image" in sample:
        img = normalize_image_object(sample["image"])
        if img is not None:
            images.append(img)

    # ------------------------------------------------------------------
    # sample["images"]
    # Can be:
    # - list
    # - dict[str, image]
    # - dict[str, {"path": "..."}]
    # ------------------------------------------------------------------
    image_registry = {}

    if "images" in sample:
        raw_images = sample["images"]

        if isinstance(raw_images, dict):
            for key, value in raw_images.items():
                img = normalize_image_object(value)
                if img is not None:
                    image_registry[key] = img
                    images.append(img)

        elif isinstance(raw_images, list):
            for value in raw_images:
                img = normalize_image_object(value)
                if img is not None:
                    images.append(img)

        else:
            img = normalize_image_object(raw_images)
            if img is not None:
                images.append(img)

    # ------------------------------------------------------------------
    # Images referenced inside messages
    # ------------------------------------------------------------------
    messages = sample.get("messages", [])

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", [])

            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get("type") not in {"image", "image_url"}:
                    continue

                # Case: {"type": "image", "image": ...}
                if "image" in part:
                    img = normalize_image_object(part["image"])
                    if img is not None:
                        images.append(img)
                    continue

                # Case: {"type": "image", "path": "..."}
                if "path" in part:
                    img = normalize_image_object(part["path"])
                    if img is not None:
                        images.append(img)
                    continue

                # Case: {"type": "image", "image_id": "query"}
                image_id = part.get("image_id", None)
                if image_id is not None and image_id in image_registry:
                    images.append(image_registry[image_id])

    # Avoid duplicated object references where possible
    unique_images = []
    seen = set()

    for img in images:
        key = id(img)
        if key not in seen:
            unique_images.append(img)
            seen.add(key)

    return unique_images

def normalize_image_object(value: Any) -> Optional[Any]:
    """
    Convert an image-like object into something the HF processor can consume.

    Returns:
    - PIL.Image.Image if a path is provided
    - the original object if it already looks like an image/tensor/array
    """

    if value is None:
        return None

    # Canonical ImageAsset-like dict
    if isinstance(value, dict):
        if "path" in value:
            return load_image(value["path"])

        if "image" in value:
            return normalize_image_object(value["image"])

        if "url" in value:
            # Avoid downloading here. Dataset stats should not depend on network.
            return None

        return None

    # Path-like
    if isinstance(value, (str, Path)):
        return load_image(value)

    # PIL image, numpy array, torch tensor, etc.
    return value

def load_image(path: str | Path) -> Optional[Any]:
    try:
        from PIL import Image

        path = Path(path)
        if not path.exists():
            return None

        return Image.open(path).convert("RGB")

    except Exception:
        return None

def try_processor_encode(
    processor,
    text: str,
    images: list[Any],
) -> Optional[Any]:
    """
    Try common Hugging Face VLM processor call patterns.
    Different processors expect slightly different text/image nesting.
    """

    attempts = [
        lambda: processor(
            text=text,
            images=images,
            return_tensors="pt",
            truncation=False,
        ),
        lambda: processor(
            text=[text],
            images=images,
            return_tensors="pt",
            truncation=False,
        ),
        lambda: processor(
            images=images,
            text=text,
            return_tensors="pt",
            truncation=False,
        ),
        lambda: processor(
            images=images,
            text=[text],
            return_tensors="pt",
            truncation=False,
        ),
    ]

    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue

    return None

def extract_input_ids(encoded: Any) -> Optional[Any]:
    if encoded is None:
        return None

    if isinstance(encoded, dict):
        return encoded.get("input_ids", None)

    return getattr(encoded, "input_ids", None)

def flatten_single_input_ids(input_ids: Any) -> list[int]:
    """
    Normalize tokenizer/processor input_ids into a single list[int].
    Handles:
    - torch.Tensor [seq]
    - torch.Tensor [1, seq]
    - list[int]
    - list[list[int]]
    """

    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().tolist()

    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    return list(input_ids)

def estimate_raw_token_length(
    sample: dict[str, Any],
    processor,
    include_image_tokens: bool = True,
    image_token_fallback: int = 0,
) -> Optional[int]:
    """
    Estimate sequence length before collation.

    Supports:
    - chat/messages style datasets
    - prompt/completion style datasets
    - instruction/input/output style datasets
    - plain text datasets
    - VLM samples with images

    If possible, this function uses the actual processor with both text and images.
    This is the most reliable way to include image tokens.

    If image processing fails, it falls back to tokenizer-only length plus
    image_token_fallback * num_images.
    """

    tokenizer = getattr(processor, "tokenizer", processor)

    text = build_text_from_sample(
        sample=sample,
        processor=processor,
        tokenizer=tokenizer,
    )

    if text is None:
        return None

    images = extract_images_from_sample(sample) if include_image_tokens else []

    # ------------------------------------------------------------------
    # Best path for VLMs: let the processor build input_ids from text+images
    # ------------------------------------------------------------------
    if include_image_tokens and len(images) > 0 and callable(processor):
        encoded = try_processor_encode(
            processor=processor,
            text=text,
            images=images,
        )

        if encoded is not None:
            input_ids = extract_input_ids(encoded)
            if input_ids is not None:
                return len(flatten_single_input_ids(input_ids))

    # ------------------------------------------------------------------
    # Fallback: text-only tokenizer
    # ------------------------------------------------------------------
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    input_ids = extract_input_ids(encoded)
    if input_ids is None:
        return None

    input_ids = flatten_single_input_ids(input_ids)

    # If processor path failed, optionally add a fixed estimate per image.
    # Set image_token_fallback e.g. to 256, 576, etc. only if you know your model.
    estimated_image_tokens = len(images) * image_token_fallback if include_image_tokens else 0

    return len(input_ids) + estimated_image_tokens


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    chunks = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        chunks.append(f"{role}:")

        if isinstance(content, str):
            chunks.append(content)

        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type")

                if part_type == "text":
                    chunks.append(part.get("text", ""))

                elif part_type in {"image", "image_url"}:
                    chunks.append("<image>")

    return "\n".join(chunks)


def extract_input_lengths_from_batch(
    batch: dict[str, Any],
    processor,
) -> Optional[list[int]]:
    if "input_ids" not in batch:
        return None

    input_ids = batch["input_ids"]

    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        lengths = attention_mask.sum(dim=-1)
        return [int(x) for x in lengths.detach().cpu().tolist()]

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is not None:
        lengths = (input_ids != pad_token_id).sum(dim=-1)
        return [int(x) for x in lengths.detach().cpu().tolist()]

    return [int(input_ids.shape[-1])] * int(input_ids.shape[0])


def extract_label_lengths_from_batch(
    batch: dict[str, Any],
) -> Optional[list[int]]:
    if "labels" not in batch:
        return None

    labels = batch["labels"]
    lengths = (labels != -100).sum(dim=-1)
    return [int(x) for x in lengths.detach().cpu().tolist()]


def numeric_stats(
    values: list[int | float],
    max_length: Optional[int] = None,
) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    values = sorted(float(v) for v in values)
    count = len(values)
    mean = sum(values) / count

    variance = sum((v - mean) ** 2 for v in values) / count
    std = math.sqrt(variance)

    stats = {
        "count": count,
        "min": values[0],
        "mean": mean,
        "std": std,
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": values[-1],
    }

    if max_length is not None:
        over_max = sum(v > max_length for v in values)
        stats["num_over_max_length"] = over_max
        stats["ratio_over_max_length"] = over_max / count

    return stats


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return sorted_values[int(k)]

    weight = k - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def log_dataset_statistics(stats: dict[str, Any]) -> None:
    split = stats["split"]

    logging.info("Dataset statistics for split '%s'", split)
    logging.info("  Samples: %s", stats["num_samples_total"])
    logging.info("  Inspected samples: %s", stats["num_samples_inspected"])

    raw = stats["raw_token_length"]
    if raw["count"] > 0:
        logging.info(
            "  Raw token length: min=%.0f | mean=%.1f | median=%.1f | p95=%.1f | max=%.0f",
            raw["min"],
            raw["mean"],
            raw["median"],
            raw["p95"],
            raw["max"],
        )

        if "num_over_max_length" in raw:
            logging.info(
                "  Raw samples over max_length: %s / %s %.2f%%",
                raw["num_over_max_length"],
                raw["count"],
                100.0 * raw["ratio_over_max_length"],
            )

    collated = stats["collated_input_length"]
    if collated["count"] > 0:
        logging.info(
            "  Collated input length: min=%.0f | mean=%.1f | median=%.1f | p95=%.1f | max=%.0f",
            collated["min"],
            collated["mean"],
            collated["median"],
            collated["p95"],
            collated["max"],
        )

    labels = stats["collated_label_length"]
    if labels["count"] > 0:
        logging.info(
            "  Label length: min=%.0f | mean=%.1f | median=%.1f | p95=%.1f | max=%.0f",
            labels["min"],
            labels["mean"],
            labels["median"],
            labels["p95"],
            labels["max"],
        )

    instances = stats["target_instances_per_sample"]
    if instances["count"] > 0:
        logging.info(
            "  Target instances/sample: min=%.0f | mean=%.1f | max=%.0f",
            instances["min"],
            instances["mean"],
            instances["max"],
        )

    if stats["num_errors"] > 0:
        logging.warning(
            "  Dataset statistics completed with %s errors. First errors are saved in the JSON file.",
            stats["num_errors"],
        )