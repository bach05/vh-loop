from __future__ import annotations

"""Asset models for dataset samples.

This module defines the abstract :class:`Asset` model and concrete
asset types such as :class:`ImageAsset` and :class:`DepthImageAsset`.
Assets carry a URI, optional caption and instance annotations.
"""

from abc import ABC
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from .annotations import InstanceAnnotation


class Asset(BaseModel, ABC):
    """Base class for assets belonging to a sample.

    ``type`` is a short string identifying the asset kind (typically a
    literal in subclasses). ``uri`` is a path or URL to the underlying
    data. ``annotations`` holds a list of instance annotations attached
    to the asset.
    """

    type: str
    uri: str
    caption: Optional[str] = None
    annotations: list[InstanceAnnotation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_path(self, dataset_root: str | Path | None = None) -> str:
        """Return a filesystem path for this asset.

        If ``dataset_root`` is provided and ``uri`` is a relative path,
        it will be joined with the dataset root. Absolute path URIs are
        returned unchanged.
        """

        if self.uri.startswith(("http://", "https://", "s3://", "gs://")):
            return self.uri

        p = Path(self.uri)
        if dataset_root is not None and not p.is_absolute():
            p = Path(dataset_root) / p
        return str(p)


class ImageAsset(Asset):
    """Image asset with size information.

    ``size`` is a (width, height) tuple. Convenience properties ``width``
    and ``height`` provide integer accessors.
    """

    type: Literal["image"] = "image"
    size: tuple[int, int]  # (width, height)
    camera_id: Optional[str] = None

    @model_validator(mode="after")
    def _valid_size(self) -> "ImageAsset":
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"ImageAsset.size must be positive, got {self.size}")
        return self

    @property
    def width(self) -> int:
        return int(self.size[0])

    @property
    def height(self) -> int:
        return int(self.size[1])


class DepthImageAsset(ImageAsset):
    """Specialized image asset representing depth data."""

    type: Literal["depth"] = "depth"
