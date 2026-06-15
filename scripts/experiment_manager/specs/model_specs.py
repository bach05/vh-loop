from typing import Optional
from .base import BaseSpec

class ModelParams(BaseSpec):
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = True
    full_finetuning: bool = False
    gpu_memory_utilization: float = 0.8

class ModelConfig(BaseSpec):
    model_name_or_path: str

class ModelParamsConfig(BaseSpec):
    model_name_or_path: str
    model_params: ModelParams = ModelParams()

class ModelSpec(BaseSpec):
    adapter: str
    params: ModelParamsConfig
