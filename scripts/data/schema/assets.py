from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from data.schema.annotations import InstanceAnnotation

class Asset(ABC, BaseModel):
    asset_type: str
    uri: str
    caption: Optional[str] = None
    annotations: list[InstanceAnnotation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_path(self, dataset_root: str | Path | None = None) -> str:
        p = Path(self.uri)
        if dataset_root is not None and not p.is_absolute():
            p = Path(dataset_root) / p
        return str(p)


class ImageAsset(Asset):
    type: Literal["image"] = "image"
    size: tuple[int, int]  # (width, height)
    camera_id: Optional[str] = None

    @property
    def width(self) -> int:
        return int(self.size[0])

    @property
    def height(self) -> int:
        return int(self.size[1])


class DepthImageAsset(ImageAsset):
    type: Literal["depth"] = "depth"
