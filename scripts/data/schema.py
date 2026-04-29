# src/vlmdojo/data/schema.py

from pydantic import BaseModel, Field
from typing import Literal, Optional, Any


class ImageRef(BaseModel):
    id: str
    uri: str
    local_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: Optional[str] = None


class MessageContent(BaseModel):
    type: Literal["text", "image_ref"]
    text: Optional[str] = None
    image_id: Optional[str] = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[MessageContent]


class ObjectAnnotation(BaseModel):
    label: str
    bbox_xyxy: Optional[list[float]] = None
    mask_uri: Optional[str] = None
    polygon: Optional[list[list[float]]] = None
    score: Optional[float] = None
    source: Literal["human", "pseudo", "model", "imported"] = "human"


class Target(BaseModel):
    type: Literal[
        "assistant_text",
        "json_detection",
        "json_segmentation",
        "caption",
        "classification",
        "freeform"
    ]
    text: Optional[str] = None
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    run_id: str
    model_name: str
    adapter_name: Optional[str] = None
    text: Optional[str] = None
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferencePair(BaseModel):
    chosen: Target
    rejected: Target
    source_prediction_run_id: Optional[str] = None


class VLMSample(BaseModel):
    sample_id: int
    dataset_id: str
    split: Optional[Literal["train", "val", "test", "unlabeled"]] = None

    images: list[ImageRef]
    messages: list[Message]

    target: Optional[Target] = None

    metadata: dict[str, Any] = Field(default_factory=dict)