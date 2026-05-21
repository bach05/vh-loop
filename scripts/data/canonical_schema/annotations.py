from __future__ import annotations

"""Annotation models used by the data canonical_schema package.

This module defines instance-level annotations that describe labeled
objects inside an image sample (bounding boxes, keypoints, masks,
captions and lightweight attributes). Models are Pydantic
BaseModel subclasses so they validate input and serialize cleanly.
"""

from typing import Optional
from pydantic import BaseModel, Field

from .geometry import BoundingBox, Point, RLEMask


class InstanceAnnotation(BaseModel):
    """One active instance annotation in a dataset version.

    Parameters
    ----------
    instance_id:
        Unique id for the instance within the sample (e.g. "obj_001").
    label_id:
        Integer category id referring to the dataset's label_info table.
    label_name:
        Human-readable label name.
    bbox:
        Optional axis-aligned bounding box for the instance.
    points:
        Optional list of keypoints or representative points (instances of
        :class:`Point`).
    mask:
        Optional segmentation mask in COCO RLE format (:class:`RLEMask`).
    caption:
        Optional short description used for prompt construction.
    attributes:
        Arbitrary mapping of string metadata for the instance.
    """

    instance_id: str  # id of the instance in the image, e.g. "obj_001"
    label_id: int  # category id, referring to the label_info
    label_name: str  # label mnemonic name, referring to the label_info

    bbox: Optional[BoundingBox] = None  # bbox of the object
    points: Optional[list[Point]] = None  # list of points of the object
    mask: Optional[RLEMask] = None  # mask of the object

    caption: Optional[str] = None  # description of the instance, used to build prompts
    attributes: dict[str, str] = Field(default_factory=dict)  # schematic attributes (FUTURE)

