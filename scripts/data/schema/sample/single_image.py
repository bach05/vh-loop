from __future__ import annotations

"""Single-image sample implementations.

This module contains sample implementation for prompts with a single image (SI), treated as a query image. 
"""

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import Field, model_validator

# relative import to project constants and local schema modules
from ...core.constants import SUPPORTED_PROMPTING_SCHEMAS
from ..assets import ImageAsset
from ..annotations import InstanceAnnotation
from ..dataset_header import DatasetInfo, MessageBuildInfo
from .base import DataSample, PromptingSchema, make_message

def _target_boxes_from_annotations(
    annotation_list: list[InstanceAnnotation],
    *,
    asset_width: int,
    asset_height: int,
    norm_factor: float = 1000.0,
) -> str:

    items = []

    for ann in annotation_list:
        if ann.bbox is None:
            continue
        class_name = ann.label_name
        box = ann.bbox.to_text(img_width=asset_width, img_height=asset_height, norm_factor=norm_factor)

        items.append(f"<{class_name},{box}>")

    if not items:
        return "<none>"

    return ";".join(items)


class SISimpleDataSample(DataSample):
    """
    This sample delivers a prompting strategy with a single image (SI) 
    while the textual part is derived from a DatasetInfo element in a trivial way: 
    all items shares the same textual prompt.
    """

    sample_type:  Literal["si_simple_data"] = "si_simple_data"
    assets: list[ImageAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_annotations(self) -> "SISimpleDataSample":
        #check that we have a list with one ImageAsset
        if len(self.assets) != 1:
            raise ValueError(f"SISimpleDataSample requires exactly one ImageAsset in assets. Found {len(self.assets)}.")
        return self
        
    @staticmethod
    def build_dataset_level_description(dataset_info: DatasetInfo) -> str:
        label_info = dataset_info.label_info
        description_lines = []
        for class_name, info in label_info.items():
            desc = info.description or ""
            desc = desc.strip()
            line = f"{info.label_name}: {desc}"
            description_lines.append(line)
        return "\n".join(description_lines)

    @staticmethod
    def build_prompt(dataset_info: DatasetInfo) -> str:
        label_text = SISimpleDataSample.build_dataset_level_description(dataset_info)
        return (
            "Detect all visible instances belonging to the following classes:\n"
            f"{label_text}\n"
            "Return only a list of formatted elements. Each item must have the format "
            "'<class_name,x1,y1,x2,y2>'. Elements are separated by ; "
            "If no target object is visible, return '<none>'."
        )

    def sample_to_message(
        self,
        dataset_info: DatasetInfo,
        *,
        prompting_schema: PromptingSchema = "conversational",
        include_target: bool = True,
        dataset_root: str | Path | None = None,
        build_info: MessageBuildInfo | None = None,
    ) -> dict[str, Any]:

        # Determine normalization factor: prefer the provided build_info,
        # otherwise fall back to dataset-level message_build_info and a
        # sensible default.
        if build_info is not None:
            norm_factor = build_info.normalization_factor
        elif dataset_info and dataset_info.message_build_info is not None:
            norm_factor = dataset_info.message_build_info.normalization_factor
        else:
            norm_factor = 1000.0
        
        asset = self.assets[0]

        image_path = asset.resolve_path(dataset_root)
        prompt = SISimpleDataSample.build_prompt(dataset_info)

        target_text = _target_boxes_from_annotations(
            asset.annotations,
            asset_width=asset.width,
            asset_height=asset.height,
            norm_factor=norm_factor,
        )

        user_message = make_message(
            "user",
            [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        )

        messages = [user_message]
        images = [image_path]

        if prompting_schema == "conversational":
            if include_target:
                assistant_message = make_message(
                    "assistant", [{"type": "text", "text": target_text}]
                )
                messages.append(assistant_message)

            return {"messages": messages, "images": images}

        elif prompting_schema == "prompt-completion":
            completion = []
            if include_target:
                completion = [
                    make_message("assistant", [{"type": "text", "text": target_text}])
                ]

            return {"prompt": messages, "completion": completion, "images": images}

        else:
            raise ValueError(
                f"Unsupported dataset schema: {prompting_schema}, "
                f"supported schemas: {SUPPORTED_PROMPTING_SCHEMAS}"
            )

########## TO DO ##############################################################
# 
# class AssetLevelDetectionSample(SISimpleDataSampleSample):
#     """Image + annotations + asset caption. Caption becomes sample-specific prompt context."""
# 
# 
# class InstanceLevelDetectionSample(SISimpleDataSampleSample):
#     """Image + annotations + per-instance captions/descriptions."""
# 
# 
# class ReasonedDetectionSample(SISimpleDataSampleSample):
#     """Image + annotations + explicit reasoning summaries/evidence."""

