from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import Field, model_validator

from scripts.data.schema import InstanceAnnotation
from scripts.data.schema import DatasetInfo, MessageBuildInfo
from scripts.data.schema import Asset, ImageAsset
from scripts.data.schema.sample.base import DataSample, PromptingSchema #_format_training_row


def _annotation_label_name(ann: InstanceAnnotation, dataset_info: DatasetInfo) -> str:
    return ann.label_name or dataset_info.label_name(ann.label_id)

def _target_boxes_from_annotations(
    annotations: list[InstanceAnnotation],
    *,
    asset_width: int,
    asset_height: int,
) -> str:

    items = []

    for ann in annotations:
        if ann.bbox is None:
            continue
        class_name = ann.label_name
        box = ann.bbox.to_text(img_width=asset_width, img_height=asset_height)

        items.append(f"<{class_name} {box}>")

    return ";".join(items)


class DatasetPromptDet(DataSample):
    """Image + buonding box annotations. Prompt context comes from DatasetInfo.label_info."""

    sample_type: Literal["dataset_level_detection"] = "dataset_level_detection"
    assets: list[ImageAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_annotations(self) -> "DatasetPromptDet":
        #check that we have a list with one ImageAsset
        if len(self.assets) != 1:
            raise ValueError(f"DatasetPromptDet requires exactly one ImageAsset in assets. Found {len(self.assets)}.")
        asset = self.assets[0]
        if not asset.annotations:
            raise ValueError("DatasetPromptDet requires annotations on the asset.")
        return self
        
    @staticmethod
    def build_dataset_level_description(dataset_info: DatasetInfo) -> str:
        label_info = dataset_info.label_info
        description_lines = []
        for class_name, info in label_info.items():
            desc = info.description or ""
            desc = desc.strip()
            line = f"{class_name}: {desc}"
            description_lines.append(line)
        return "\n".join(description_lines)

    @staticmethod
    def build_prompt(dataset_info: DatasetInfo) -> str:
        label_text = DatasetPromptDet.build_dataset_level_description(dataset_info)
        return (
            "Detect all visible instances belonging to the following classes:\n"
            f"{label_text}\n"
            "Return only a list of formatted elements. Each item must have the format "
            '"<class_name  x1,y1,x2,y2>". Elements are separated by ; '
        )

    def sample_to_message(
        self,
        dataset_info: DatasetInfo,
        *,
        prompting_schema: PromptingSchema = "conversational",
        dataset_root: str | Path | None = None,
        include_target: bool = True,
    ) -> dict[str, Any]:
        
        build_info = dataset_info.message_build_info
        
        asset = self.assets[0]

        image_path = asset.resolve_path(dataset_root)
        prompt = DatasetPromptDet.build_prompt(dataset_info)

        target_text = _target_boxes_from_annotations(
            asset.annotations,
            asset_width=asset.width,
            asset_height=asset.height,
        )

        messages = []
        images = []


        #append prompt
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type":'image', "path": image_path},
                    {"type":'text', "text": prompt},

                ]
            }
        )



        # For inference, return prompt-only format.
        # Adjust this if your adapter expects another key.
        return {
            "messages": messages,
            "images": images,
        }


class AssetLevelDetectionSample(DatasetPromptDetSample):
    """Image + annotations + asset caption. Caption becomes sample-specific prompt context."""

    sample_type: Literal["asset_level_detection"] = "asset_level_detection"

    @model_validator(mode="after")
    def _check_asset_caption(self) -> "AssetLevelDetectionSample":
        if not self.primary_asset().caption:
            raise ValueError("AssetLevelDetectionSample requires primary asset.caption")
        return self

    def build_prompt(self, dataset_info: DatasetInfo, build_info: MessageBuildInfo) -> str:
        base = super().build_prompt(dataset_info, build_info)
        return f"Image description: {self.primary_asset().caption}\n\n{base}"


class InstanceLevelDetectionSample(DatasetPromptDetSample):
    """Image + annotations + per-instance captions/descriptions."""

    sample_type: Literal["instance_level_detection"] = "instance_level_detection"

    @model_validator(mode="after")
    def _check_instance_captions(self) -> "InstanceLevelDetectionSample":
        missing = [ann.instance_id or str(i) for i, ann in enumerate(self.primary_asset().annotations) if not ann.caption]
        if missing:
            raise ValueError(f"InstanceLevelDetectionSample requires annotation.caption for all annotations. Missing: {missing}")
        return self

    def build_prompt(self, dataset_info: DatasetInfo, build_info: MessageBuildInfo) -> str:
        base = super().build_prompt(dataset_info, build_info)
        descriptions = "\n".join(
            f"- {_annotation_label_name(ann, dataset_info)}: {ann.caption}"
            for ann in self.primary_asset().annotations
        )
        return f"Relevant object descriptions:\n{descriptions}\n\n{base}"


class ReasonedDetectionSample(DatasetPromptDetSample):
    """Image + annotations + explicit reasoning summaries/evidence."""

    sample_type: Literal["reasoned_detection"] = "reasoned_detection"

    @model_validator(mode="after")
    def _check_reasoning(self) -> "ReasonedDetectionSample":
        missing = [ann.instance_id or str(i) for i, ann in enumerate(self.primary_asset().annotations) if not ann.reasoning_summary]
        if missing:
            raise ValueError(f"ReasonedDetectionSample requires annotation.reasoning_summary for all annotations. Missing: {missing}")
        return self

    def sample_to_message(
        self,
        dataset_info: DatasetInfo,
        *,
        dataset_schema: DatasetSchema = "conversational",
        dataset_root: str | Path | None = None,
        include_target: bool = True,
        build_info: MessageBuildInfo | None = None,
    ) -> dict[str, Any]:
        build_info = (build_info or dataset_info.message_build_info).model_copy(update={"include_reasoning": True})
        return super().sample_to_message(
            dataset_info,
            dataset_schema=dataset_schema,
            dataset_root=dataset_root,
            include_target=include_target,
            build_info=build_info,
        )
