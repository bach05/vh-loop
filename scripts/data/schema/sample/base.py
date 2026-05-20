from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from scripts.core.constants import SUPPORTED_PROMPTING_SCHEMAS

from scripts.data.schema import Asset, ImageAsset
from scripts.data.schema import DatasetInfo, MessageBuildInfo

PromptingSchema = SUPPORTED_PROMPTING_SCHEMAS


class DataSample(BaseModel, ABC):
    """Abstract sample contract. Subclasses define required fields and message logic."""

    sample_type: str
    sample_id: str
    dataset_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def sample_to_message(
        self,
        dataset_info: DatasetInfo,
        *,
        prompting_schema: PromptingSchema = "conversational",
        dataset_root: str | Path | None = None,
        build_info: MessageBuildInfo | None = None,
    ) -> dict[str, Any]:
        ...

