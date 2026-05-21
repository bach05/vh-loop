from __future__ import annotations

"""Geometry primitives used by the data schema.

This module defines small Pydantic models for common geometric
primitives: 2-D points, axis-aligned bounding boxes and RLE masks. The
models provide convenience helpers for normalization and simple
serialization to textual compact formats used in prompt/target
construction.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast, Optional

from pydantic import BaseModel, model_validator
import numpy as np


class Geometry(BaseModel, ABC):
    """Abstract geometry base class.

    Subclasses should implement :meth:`to_text` for compact textual
    serialization used by prompt/target builders.
    """

    @abstractmethod
    def to_text(self, **kwargs) -> str:
        """Return a compact textual representation of the geometry.

        The exact keyword arguments depend on the subclass (e.g.
        image width/height and normalization factor) and are documented
        on the concrete classes.
        """
        pass


class Point(Geometry):
    """2-D integer point in image pixel coordinates.

    ``is_positive`` is an optional flag used by some point list
    encodings to indicate positive/negative examples.
    """

    x: int
    y: int
    is_positive: Optional[bool] = None  # used in point lists; optional

    def normalize(self, img_width: int, img_height: int, norm_factor: float = 1.0) -> tuple[float, float]:
        if img_width <= 0 or img_height <= 0:
            raise ValueError("img_width and img_height must be > 0")

        norm_x = self.x / img_width * norm_factor
        norm_y = self.y / img_height * norm_factor
        return norm_x, norm_y

    def to_text(self, img_width: int, img_height: int, norm_factor: float = 1000.0) -> str:
        norm_x, norm_y = self.normalize(img_width, img_height, norm_factor)
        return f"{int(round(norm_x))},{int(round(norm_y))}"


class BoundingBox(Geometry):
    """Axis-aligned bbox with exclusive bottom-right corner: [tl, br).

    ``tl`` and ``br`` are instances of :class:`Point` expressed in image
    pixel coordinates.
    """

    tl: Point
    br: Point
    format: Literal["xyxy"] = "xyxy"

    @model_validator(mode="after")
    def _valid(self) -> "BoundingBox":
        if self.tl.x >= self.br.x or self.tl.y >= self.br.y:
            raise ValueError(f"tl must be strictly above-left of br: tl={self.tl}, br={self.br}")
        return self

    @property
    def width(self) -> float:
        return self.br.x - self.tl.x

    @property
    def height(self) -> float:
        return self.br.y - self.tl.y

    def area(self) -> float:
        return self.width * self.height

    def normalize(self, img_width: int, img_height: int, norm_factor: float = 1.0):
        tl_norm = self.tl.normalize(img_width, img_height, norm_factor)
        br_norm = self.br.normalize(img_width, img_height, norm_factor)
        return tl_norm, br_norm

    def to_text(self, img_width: int, img_height: int, norm_factor: float = 1000.0) -> str:
        tl_norm, br_norm = self.normalize(img_width, img_height, norm_factor)
        return f"{int(round(tl_norm[0]))},{int(round(tl_norm[1]))},{int(round(br_norm[0]))},{int(round(br_norm[1]))}"


class RLEMask(Geometry):
    """Binary mask in COCO RLE format.

    ``counts`` may be either a string (the compact RLE) or a list of
    integers. ``size`` is (height, width) as returned by COCO tools.
    """

    counts: list[int] | str
    size: tuple[int, int]  # (height, width)

    def to_binary_mask(self):  # -> np.ndarray[bool]
        from pycocotools import mask as mask_utils  # type: ignore

        rle = {"counts": self.counts, "size": list(self.size)}
        decoded = mask_utils.decode(cast(Any, rle))
        return decoded.astype(bool)

    @classmethod
    def from_binary_mask(cls, mask) -> "RLEMask":
        from pycocotools import mask as mask_utils  # type: ignore

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        h, w = rle["size"]
        return cls(counts=counts, size=(h, w))

    def to_text(self) -> str:
        if isinstance(self.counts, str):
            return self.counts
        else:
            return ",".join(map(str, self.counts))