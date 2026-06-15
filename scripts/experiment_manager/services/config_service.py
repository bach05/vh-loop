import os
import yaml
from pathlib import Path
from typing import Type, TypeVar, Any, Dict
from omegaconf import OmegaConf

from ..specs.base import BaseSpec
from ..specs import ModelSpec, DatasetSpec, TrainerSpec, ExperimentSpec

T = TypeVar("T", bound=BaseSpec)

class ConfigService:
    def __init__(self, configs_dir: str):
        self.configs_dir = Path(configs_dir)

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        """Loads a yaml file into a dictionary using OmegaConf."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        # Resolve to standard dict
        conf = OmegaConf.load(path)
        return OmegaConf.to_container(conf, resolve=True)

    def save_yaml(self, data: Dict[str, Any], path: Path):
        """Saves a dictionary to a yaml file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_spec(self, category: str, filename: str, spec_class: Type[T]) -> T:
        """Loads a specific config and parses it into a Pydantic spec."""
        path = self.configs_dir / category / filename
        data = self.load_yaml(path)
        return spec_class.model_validate(data)

    def save_spec(self, category: str, filename: str, spec: BaseSpec):
        """Saves a Pydantic spec to a yaml file."""
        path = self.configs_dir / category / filename
        data = spec.model_dump(exclude_unset=True)
        self.save_yaml(data, path)
        
    def list_configs(self, category: str) -> list[str]:
        """Lists available yaml configurations in a category."""
        category_dir = self.configs_dir / category
        if not category_dir.exists():
            return []
        return [f.name for f in category_dir.glob("*.yaml")]

