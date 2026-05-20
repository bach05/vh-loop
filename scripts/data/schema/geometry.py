from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

from pydantic import BaseModel, model_validator
import numpy as np

#Abstract class Geometry
class Geometry(ABC, BaseModel):
    """Base class for geometric annotations like bounding boxes and masks."""
    primitive_id: str

    @abstractmethod
    def to_text(self, **kwargs) -> str:
        """Convert to a serializable dict for inclusion in VLMSample annotations."""
        pass


class Point(Geometry, BaseModel):
    """2-D point. The owning geometry defines the pixel image space.
        This because the original pixel space allow the maximum precision with the most compact representation (int).
    """

    primitive_id = "point"

    x: int
    y: int
    is_positive: bool = None

    def normalize(self, img_width: int, img_height: int, norm_factor: float = 1.0) -> tuple[float, float]:
        norm_x = self.x / img_width * norm_factor
        norm_y = self.y / img_height * norm_factor
        return norm_x, norm_y

    def to_text(self, img_width: int, img_height: int, norm_factor: float = 1000.0):
        norm_x, norm_y = self.normalize(img_width, img_height, norm_factor)
        return f"{int(round(norm_x))},{int(round(norm_y))}"


class BoundingBox(Geometry, BaseModel):
    """Axis-aligned bbox with exclusive br corner: [tl, br)."""

    tl: Point
    br: Point
    format: Literal["xyxy"] = "xyxy"
    primitive_id = "bbox"

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

    def to_text(self, img_width: int, img_height: int):
        tl_norm, br_norm = self.normalize(img_width, img_height)
        return f"{int(round(tl_norm[0]))},{int(round(tl_norm[1]))},{int(round(br_norm[0]))},{int(round(br_norm[1]))}"

class RLEMask(Geometry, BaseModel):
    """Binary mask in COCO RLE format."""

    counts: list[int] | str
    size: tuple[int, int]  # (height, width)
    primitive_id = "rle_mask"

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
            return f"{self.counts}"
        else:
            return ",".join(self.counts)