from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from scripts.data.schema import BoundingBox, Point, RLEMask

class InstanceAnnotation(BaseModel):
    """One active instance annotation in a dataset version."""

    instance_id: str
    label_id: int | str
    label_name: str

    bbox: Optional[BoundingBox] = None
    points: Optional[list[Point]] = None
    mask: Optional[RLEMask] = None

    caption: Optional[str] = None
    attributes: dict[str, str] = Field(default_factory=dict)

