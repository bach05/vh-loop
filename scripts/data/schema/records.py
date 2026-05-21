# records.py
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel

from .dataset_header import SCHEMA_VERSION, DatasetInfo
from .sample.registry import SampleUnion


class DatasetInfoRecord(BaseModel):
    """Wrapper record carrying dataset-level information.

    Useful when storing or streaming metadata records alongside
    samples.
    """

    record_type: Literal["dataset_info"] = "dataset_info"
    schema_version: str = SCHEMA_VERSION
    info: DatasetInfo


class DataRecord(BaseModel):
    """Wrapper record carrying a sample union (discriminated).

    The ``sample`` field is a :class:`SampleUnion` allowing different
    sample types to be discriminator-deserialized by Pydantic.
    """

    record_type: Literal["sample"] = "sample"
    sample: SampleUnion
