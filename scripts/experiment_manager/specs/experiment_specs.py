from typing import Optional
from pydantic import Field
from .base import BaseSpec
from .model_specs import ModelSpec
from .dataset_specs import DatasetSpec
from .trainer_specs import TrainerSpec
from .backend_specs import BackendSpec, LocalBackendSpec

class ExperimentSpec(BaseSpec):
    name: str
    model: ModelSpec
    dataset: DatasetSpec
    trainer: TrainerSpec
    backend: BackendSpec = Field(default_factory=LocalBackendSpec)
