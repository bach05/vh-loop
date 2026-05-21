from __future__ import annotations

"""Dataset-level metadata models.

This module contains dataset and label description models used to
describe dataset-level metadata (labels, provenance, message building
options) and a lightweight record wrapper used when serializing a
dataset info message.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator
import re

# CONSTANTS
SCHEMA_VERSION = "vh_loop.data_schema.v2"
_CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

class LabelInfo(BaseModel):
    """Description of one label/class inside a dataset-specific taxonomy.

    Attributes
    ----------
    label_id:
        Numeric id used by the dataset.
    label_name:
        Short mnemonic name used as dictionary key in DatasetInfo.label_info.
    description:
        Optional human readable description used in prompts.
    aliases:
        Optional alternative names for the same class.
    parent_label:
        Optional parent label name (for hierarchies).
    """

    label_id: int  # integer id, used for detection/segmentation
    label_name: str  # mnemonic name for the class
    description: Optional[str] = None  # used to build prompts
    aliases: list[str] = Field(default_factory=list)  # used to deal with possible label name variations
    parent_label: Optional[str] = None  # building hierarchy (FUTURE)

    @model_validator(mode="after")
    def _validate_label_name(self) -> "LabelInfo":
        if not _CLASS_NAME_RE.match(self.label_name):
            raise ValueError(
                f"Invalid label_name={self.label_name!r}. "
                "Use a compact class token without spaces: 'PET bottle' -> invalid,  'PET_bottle' -> valid."
            )
        return self


class AnnotationInfo(BaseModel):
    """Compact provenance / quality information for labels, captions or geometries."""

    source_type: Literal["human", "ai", "ai_human_reviewed", "synthetic", "imported", "web_scrap"] = "imported"
    quality: Literal["raw", "weak", "auto", "reviewed", "gold"] = "raw"
    notes: Optional[str] = None

class MessageBuildInfo(BaseModel):
    """Reproducibility metadata for sample_to_message generation.

    Contains parameters that affect how prompts and targets are built
    from samples (normalization factor for coordinates, answer format,
    etc.).
    """

    prompt_template_version: str = "single_image_dataset_level_prompt_generation"
    answer_format: Literal["tag_bbox_list", "text"] = "tag_bbox_list" #text can be used for pretraining
    normalization_factor: int = 1000  # used to normalize geometries in a normalization_factor x normalization_factor grid
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInfo(BaseModel):
    """Container for dataset-level metadata.

    ``label_info`` maps short label names to :class:`LabelInfo` objects and
    is used by samples to resolve names/ids when building prompts.
    """

    dataset_id: str  # identifier of the dataset, e.g. "paniz_motors_apr26"
    description: Optional[str] = None  # free-form text describing the dataset

    annotation_info: Optional[AnnotationInfo] = None

    domain: Optional[str] = None  # e.g. waste-sorting
    split: Optional[str] = None  # train, val, test

    date_collected: Optional[str] = None
    date_last_update: Optional[str] = None

    label_info: dict[str, LabelInfo] = Field(default_factory=dict)

    message_build_info: MessageBuildInfo = Field(default_factory=MessageBuildInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_label_info_keys(self) -> "DatasetInfo":
        for key, info in self.label_info.items():
            if key != info.label_name:
                raise ValueError(
                    f"label_info key {key!r} must match LabelInfo.label_name {info.label_name!r}"
                )
        return self

    def label_name_to_id(self, label_name: str) -> int | str:
        label_info = self.label_info.get(label_name, None)
        if label_info is None:
            raise ValueError(f"Label name '{label_name}' not found in dataset label information.")
        return label_info.label_id

    def label_id_to_name(self, label_id: int | str) -> str:
        for label_name, label_info in self.label_info.items():
            if label_info.label_id == label_id:
                return label_name
        raise ValueError(f"Label id '{label_id}' not found in dataset label information.")


