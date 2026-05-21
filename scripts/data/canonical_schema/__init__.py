"""Public exports for the data canonical_schema package.

This module re-exports commonly used models from the
`scripts.data.canonical_schema` submodules so callers can import them from
``scripts.data.canonical_schema``. Use relative imports to avoid import-time
cycles when modules are imported from inside the package.
"""

from .geometry import Point, BoundingBox, RLEMask
from .assets import Asset, ImageAsset, DepthImageAsset
from .dataset_header import DatasetInfo, SCHEMA_VERSION, MessageBuildInfo
from .records import DatasetInfoRecord, DataRecord
from .annotations import InstanceAnnotation
from .sample.single_image import SISimpleDataSample, PromptingSchema

__all__ = [
    # assets
    "Asset", "ImageAsset", "DepthImageAsset",
    # geometries
    "Point", "BoundingBox", "RLEMask",
    # dataset headers
    "DatasetInfo", "MessageBuildInfo",
    #records
    "DataRecord",
    # annotations
    "InstanceAnnotation",
    # samples
    "SISimpleDataSample", "PromptingSchema",
]


