from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from omegaconf import OmegaConf

import albumentations as A


try:
    from albumentations.pytorch import ToTensorV2
except Exception:
    ToTensorV2 = None


class TransformBuildError(RuntimeError):
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