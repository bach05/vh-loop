from typing import List, Optional
from .base import BaseSpec

class DatasetItemSpec(BaseSpec):
    id: str
    dataset_schema: str = "conversational"
    jsonl_path: str
    root_override: Optional[str] = None

class DatasetSpec(BaseSpec):
    training: List[DatasetItemSpec] = []
    testing: List[DatasetItemSpec] = []
