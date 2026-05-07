# src/vlmdojo/data/schema.py

from __future__ import annotations
from typing import Literal, Optional, Any, cast
from pydantic import BaseModel, Field, model_validator
import numpy as np

from scripts.core.constants import NORM_SIZE

"""
Image reference frame convention.

We use the standard image coordinate frame used by PIL/OpenCV:

    (0, 0) ----------------------> x
      |                            width direction
      |
      |      tl +-------------+
      |         |             |
      |         |             |
      |         +-------------+ br
      |
      v
      y
    height direction

Coordinate conventions:
- Pixel coordinates are expressed as (x, y).
- x increases from left to right.
- y increases from top to bottom.
- PIL.Image.size is (width, height).
- NumPy/OpenCV image.shape[:2] is (height, width).
- BoundingBox.tl is the top-left corner.
- BoundingBox.br is the bottom-right corner.
- BoundingBox.br should be interpreted as an exclusive corner, i.e. [tl, br).
"""

# Classes to represente the header of the dataset on the first line
class DatasetInfo(BaseModel):
    dataset_id: str
    description: Optional[str] = None
    annotation_source: Optional[str] = None
    has_semantic_masks: bool = False
    has_point: bool = False
    has_bbox: bool = True
    date_collected: Optional[str] = None
    label_info: dict[str, str] = Field(default_factory=dict)


class DatasetInfoRecord(BaseModel):
    record_type: Literal["dataset_info"]
    schema_version: str = "vh_loop.dataset.v1"
    info: DatasetInfo

# Data classes

class ImageAsset(BaseModel):
    path: str
    size: tuple[int, int]  # (width, height)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]


class Point(BaseModel):
    """ A 2-D point stored in normalized NORM_SIZE×NORM_SIZE coordinate space.
        from_pixel(): build from raw pixel coordinates.
        to_pixel(): recover pixel coordinates for rendering. """

    x: int = Field(ge=0, le=NORM_SIZE)
    y: int = Field(ge=0, le=NORM_SIZE)

    @classmethod
    def from_pixel(cls, x: int, y: int, img_width: int, img_height: int) -> "Point":
        """Normalize a pixel coordinate using a NORM_SIZE×NORM_SIZE square space."""
        #scale = NORM_SIZE / max(img_width, img_height)
        scale_x = NORM_SIZE / img_width
        scale_y = NORM_SIZE / img_height
        return cls(x=round(x * scale_x), y=round(y * scale_y))

    def to_pixel(self, img_width: int, img_height: int) -> tuple[int, int]:
        """Denormalize to pixel coordinates of an image with the given dimensions."""
        #scale = max(img_width, img_height) / NORM_SIZE #suppose squared images
        scale_x = img_width / NORM_SIZE
        scale_y = img_height / NORM_SIZE
        return round(self.x * scale_x), round(self.y * scale_y)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalized image coordinates.

    Convention:
    - tl is the top-left corner.
    - br is the bottom-right corner.
    - br is exclusive: the bbox covers [tl.x, br.x) and [tl.y, br.y).
    - Coordinates are normalized independently:
        x_pixel = x_norm / NORM_SIZE * image_width
        y_pixel = y_norm / NORM_SIZE * image_height

    from_pixel(): build from raw pixel coordinates.
    to_pixel(): recover pixel corners for rendering.
    area(): compute bounding box area. """

    tl: Point
    br: Point

    @model_validator(mode="after")
    def _valid(self) -> "BoundingBox":
        if self.tl.x >= self.br.x or self.tl.y >= self.br.y:
            raise ValueError(f"tl must be strictly above-left of br: tl={self.tl}, br={self.br}")
        return self

    @property
    def width(self) -> int:
        return self.br.x - self.tl.x

    @property
    def height(self) -> int:
        return self.br.y - self.tl.y

    @classmethod
    def from_pixel(cls, tl_x: int, tl_y: int, br_x: int, br_y: int, img_width: int, img_height: int) -> "BoundingBox":
        """Normalize pixel corners to NORM_SIZE×NORM_SIZE square space."""
        return cls(
            tl=Point.from_pixel(tl_x, tl_y, img_width, img_height),
            br=Point.from_pixel(br_x, br_y, img_width, img_height),
        ) #may collapse if points are very closed

    def to_pixel(self, img_width: int, img_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Denormalize tl and br in pixel coordinates."""
        return (
            self.tl.to_pixel(img_width, img_height),
            self.br.to_pixel(img_width, img_height),
        )

    def area(self) -> int:
        """Area in normalized coordinate space."""
        return self.width * self.height


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    image_id: str

MessageContent = TextContent | ImageContent


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[MessageContent]

#RLEMask use the numpy/COCO convention
class RLEMask(BaseModel):
    """Binary mask in COCO RLE format."""

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

    def centroid(self) -> Point:
        """Return the mask centroid as a normalized Point."""
        binary = self.to_binary_mask()
        ys, xs = np.nonzero(binary)
        h, w = self.size
        cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) else (w // 2, h // 2)
        return Point.from_pixel(cx, cy, img_width=w, img_height=h)


class Annotation(BaseModel):
    label: int
    bbox: Optional[BoundingBox] = None
    pos_points: Optional[list[Point]] = None
    neg_points: Optional[list[Point]] = None
    mask: Optional[RLEMask] = None
    source: Optional[str] = None

    @classmethod
    def from_mask(cls, label: int, mask, bbox: Optional[BoundingBox] = None) -> "Annotation":
        """Build an Annotation from a binary numpy mask."""
        rle_mask = RLEMask.from_binary_mask(mask)
        centroid = rle_mask.centroid()  # already normalized
        if bbox is None:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                raise ValueError("Cannot build Annotation bbox from an empty mask.")
            h, w = mask.shape[:2]
            bbox = BoundingBox.from_pixel(
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1,
                img_width=w, img_height=h,
            )
        return cls(label=label, bbox=bbox, pos_points=[centroid], mask=rle_mask)


class Target(BaseModel):
    text: Optional[str] = None
    instances: list[Annotation] = Field(default_factory=list)

class VLMSample(BaseModel):
    sample_id: int
    dataset_id: str
    #query_image: ImageAsset
    images: dict[str, ImageAsset] = Field(default_factory=dict)
    messages: list[Message]
    target: Optional[Target] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class SampleRecord(VLMSample):
    record_type: Literal["sample"] = "sample"
    schema_version: str = "vh_loop.dataset.v1"