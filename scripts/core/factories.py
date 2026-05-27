from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import logging

import albumentations as A
from omegaconf import OmegaConf as OC

try:
    from albumentations.pytorch import ToTensorV2
except Exception:
    ToTensorV2 = None

class TransformBuildError(RuntimeError):
    pass

class PeftConfigBuildError(RuntimeError):
    pass

class DatasetBuildError(RuntimeError):
    pass

def build_transform(
    transform_cfg: Mapping[str, Any],
    *,
    additional_targets: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build a single transform group from config.

    Returns:
      {
        "transform": A.Compose,
        "spatial": SpatialTransform
      }
    """
    if transform_cfg is None:
        transform_cfg = {}

    ops = transform_cfg.get("ops", []) or []
    to_tensor = bool(transform_cfg.get("to_tensor", False))

    cfg_additional = transform_cfg.get("additional_targets", None)
    add_tgts = additional_targets if additional_targets is not None else (cfg_additional or {})

    aug_list = []
    for i, op in enumerate(ops):
        if not isinstance(op, Mapping) or "name" not in op:
            raise TransformBuildError(f"Op #{i} must be a mapping with 'name'. Got: {op}")

        name = op["name"]
        params = op.get("params", {}) or {}

        cls = getattr(A, name, None)
        if cls is None:
            raise TransformBuildError(
                f"Unknown Albumentations op '{name}' at index {i}. "
                f"Check spelling or ensure it exists in albumentations."
            )
        try:
            aug_list.append(cls(**params))
        except Exception as e:
            raise TransformBuildError(
                f"Failed to instantiate '{name}' (index {i}) with params={params}. Error: {e}"
            ) from e

    if to_tensor:
        if ToTensorV2 is None:
            raise TransformBuildError(
                "to_tensor=True but albumentations.pytorch.ToTensorV2 is not available."
            )
        aug_list.append(ToTensorV2())

    return A.Compose(aug_list, additional_targets=add_tgts)

def build_peft_config(peft_cfg):
    from peft import LoraConfig
    from scripts.core.constants import SUPPORTED_PEFT_STRATEGIES

    peft_config = None
    if peft_cfg['strategy'].lower() == "lora":
        params = peft_cfg.get("params", None)
        peft_config = LoraConfig(**params) if params else None
    else:
        raise PeftConfigBuildError(f"Unknown PEFT strategy '{peft_cfg.strategy}'. Supported strategies: {SUPPORTED_PEFT_STRATEGIES}")

    return peft_config


def build_hf_datasets(dataset_cfg, transform_cfg=None, split='training'):
    from scripts.data.hf_dataset import canonical_jsonl_to_hf_dataset, TransformedVLMHFDataset
    from datasets import concatenate_datasets

    logging.info(f"Building HF dataset for data split '{split}'")

    data_sets = []
    for data_cfg in dataset_cfg.get(split, []):
        dataset_schema = data_cfg.get("dataset_schema", "conversational")
        data_jsonl_path = data_cfg.get("jsonl_path", None)
        if data_jsonl_path is None:
            raise DatasetBuildError(f"Dataset '{data_cfg.get('id', None)}' is missing required 'jsonl_path' field.")

        dataset_root = data_cfg.get("root_override", None)

        logging.info(
            f"Building HF dataset for '{data_cfg.get('id', None)}' "
            f"from: {data_jsonl_path}, schema={dataset_schema}"
        )

        dataset_hf = canonical_jsonl_to_hf_dataset(
            data_jsonl_path,
            prompting_schema=dataset_schema,
            dataset_root=dataset_root,
            cache_dir=data_cfg.get("hf_cache_dir", None),
            fingerprint_extra={
                "dataset_id": data_cfg.get("id", None),
                "split": split,
            },
        )
        data_sets.append(dataset_hf)

    data_set = concatenate_datasets(data_sets) if len(data_sets) > 1 else (data_sets[0] if data_sets else None)
    if data_set is None:
        raise DatasetBuildError(f"No datasets specified in config under 'dataset.{split}'.")

    if transform_cfg is not None:
        image_transforms = build_transform(transform_cfg)
    data_set = TransformedVLMHFDataset(
        data_set,
        transform=image_transforms if transform_cfg is not None else None,
    )

    return data_set





