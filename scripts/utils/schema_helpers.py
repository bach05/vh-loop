from __future__ import annotations

from typing import cast
from pathlib import Path
from typing import Optional
from collections.abc import Iterable
from hydra.utils import to_absolute_path

from scripts.data.canonical_dataset import CanonicalDataset
from scripts.data.canonical_schema import DatasetInfo, InstanceAnnotation
from scripts.data.canonical_schema.assets import ImageAsset
from scripts.data.canonical_schema.sample.base import DataSample


def get_image_assets(sample: Optional[DataSample]) -> list[ImageAsset]:
    """ Return all ImageAsset objects from *sample*, or an empty list. """
    if sample is None:
        return []
    assets = getattr(sample, "assets", None)
    if not assets:
        return []
    return [a for a in cast(Iterable[object], assets) if isinstance(a, ImageAsset)]


def get_primary_image_asset(sample: Optional[DataSample]) -> Optional[ImageAsset]:
    """ Return the first ImageAsset from *sample*, or None. """
    assets = get_image_assets(sample)
    return assets[0] if assets else None


def extract_bbox_annotations(sample: Optional[DataSample]) -> list[InstanceAnnotation]:
    """ Return every InstanceAnnotation that has a bounding-box across all image assets. """
    if sample is None:
        return []
    return [
        ann
        for asset in get_image_assets(sample)
        for ann in asset.annotations
        if ann.bbox is not None
    ]


def resolve_image_path(sample: DataSample, image_root: Optional[str | Path]) -> Optional[Path]:
    """ Resolve the filesystem path to the primary image of *sample*. """
    asset = get_primary_image_asset(sample)
    if asset is None:
        return None
    resolved = asset.resolve_path(
        to_absolute_path(str(image_root)) if image_root is not None else None
    )
    return Path(resolved)


def get_dataset_info(canonical_dataset: CanonicalDataset) -> DatasetInfo:
    """ Return DatasetInfo from CanonicalDataset with compatibility fallbacks. """
    if hasattr(canonical_dataset, "get_dataset_info"):
        return canonical_dataset.get_dataset_info()
    info = cast(Optional[DatasetInfo], getattr(canonical_dataset, "info", None))
    if info is None:
        raise RuntimeError("Canonical dataset has no DatasetInfo.")
    return info


def label_name_to_id_map(dataset_info: DatasetInfo) -> dict[str, int]:
    """ Build {label_name: label_id} from DatasetInfo.label_info. """
    return {
        label_name: int(info.label_id)
        for label_name, info in dataset_info.label_info.items()
    }