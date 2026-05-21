from __future__ import annotations

"""Base sample abstractions for the schema package.

This module defines :class:`DataSample`, the abstract contract that
concrete sample types must implement (notably :meth:`sample_to_message`).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel

from scripts.core.constants import SUPPORTED_PROMPTING_SCHEMAS

# Import schema types from the package root (relative import).
if TYPE_CHECKING:
    from ..dataset_header import DatasetInfo, MessageBuildInfo

PromptingSchema = SUPPORTED_PROMPTING_SCHEMAS


class DataSample(BaseModel, ABC):
    """Abstract sample contract. Subclasses define required fields and message logic.

    Concrete subclasses should set ``sample_type`` to a unique literal and
    implement :meth:`sample_to_message` to produce the prompt/target
    structures consumed by downstream tooling.
    """

    sample_type: str  # used to build the subclass specific samples
    sample_id: str  # id of the sample in the dataset, e.g. img_00001

    @abstractmethod
    def sample_to_message(
        self,
        dataset_info: DatasetInfo,
        *,
        prompting_schema: PromptingSchema = "conversational",
        include_target: bool = True,
        dataset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Serialize the sample into a prompt/target payload.

        Implementations should return a dictionary with a stable
        structure (for example keys like ``messages`` and ``images``)
        that calling code can send to a VLM or save to disk.
        """
        ...

################### COMMON UTILS ##############à

def make_message(role: str, content: list[dict]) -> dict:
    return {"role": role, "content": content}

def validate_against_dataset_info(self, dataset_info: DatasetInfo) -> None:
    for asset in self.assets:
        for ann in asset.annotations:
            expected_name = dataset_info.label_id_to_name(ann.label_id)
            if ann.label_name != expected_name:
                raise ValueError(
                    f"Annotation {ann.instance_id}: label_id={ann.label_id} "
                    f"maps to {expected_name!r}, but label_name={ann.label_name!r}"
                )
