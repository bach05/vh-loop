from typing import Literal, Optional, Union
from pydantic import Field
from .base import BaseSpec

class LocalBackendSpec(BaseSpec):
    type: Literal["local"] = "local"
    cuda_visible_devices: Optional[str] = None

class SlurmBackendSpec(BaseSpec):
    type: Literal["slurm"] = "slurm"
    partition: str = "gpu"
    qos: Optional[str] = None
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    gpus_per_task: int = 1
    mem: str = "64G"
    time: str = "24:00:00"
    singularity_image: Optional[str] = None
    bind_mounts: Optional[str] = None

BackendSpec = Union[LocalBackendSpec, SlurmBackendSpec]
