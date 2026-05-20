from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

SCHEMA_VERSION = "vh_loop.data_schema.v2"

class LabelInfo(BaseModel):
    """Description of one task label/class inside a dataset-specific taxonomy."""
    label_id: int | str
    label_name: str
    description: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    parent_label: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationInfo(BaseModel):
    """Compact provenance / quality information for labels, captions or geometries."""

    source_type: Literal["human", "ai", "ai_human_reviewed", "synthetic", "imported", "web_scrap"] = "imported"
    source_name: Optional[str] = None
    quality: Literal["raw", "weak", "auto", "reviewed", "gold"] = "raw"
    notes: Optional[str] = None

class MessageBuildInfo(BaseModel):
    """Reproducibility metadata for sample_to_message generation."""
    prompt_template_version: str = "dataset_level_prompt_generation"
    answer_format: Literal["json_bbox_list", "text"] = "json_bbox_list"
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInfo(BaseModel):
    dataset_id: str
    description: Optional[str] = None

    annotation_info: Optional[AnnotationInfo] = None

    domain: Optional[str] = None # e.g. waste-sorting
    split: Optional[str] = None # train, val, test

    date_collected: Optional[str] = None
    date_last_update: Optional[str] = None

    label_info: dict[str, LabelInfo] = Field(default_factory=dict)

    message_build_info: MessageBuildInfo = Field(default_factory=MessageBuildInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def label_name_to_id(self, label_name: str) -> int | str:
        label_info = self.label_info.get(label_name, None)
        if label_info is None:
            raise ValueError(f"Label name '{label_name}' not found in dataset label information .")
        return label_info.label_id

    def label_id_to_name(self, label_id : int) -> str | None:
        for label_name, label_info in self.label_info.items():
            if label_info.label_id == label_id:
                return label_name
        raise ValueError(f"Label id '{label_id}' not found in dataset label information .")


class DatasetInfoRecord(BaseModel):
    record_type: Literal["dataset_info"] = "dataset_info"
    schema_version: str = SCHEMA_VERSION
    info: DatasetInfo

class DataRecord(BaseModel):
    record_type: Literal["sample"] = "sample"
    sample_type: str
    sample: DataSample

