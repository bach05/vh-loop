# src/models/adapter.py

from abc import ABC, abstractmethod
from typing import Any
from scripts.data.schema import VLMSample, Target


class VLMAdapter(ABC):
    name: str

    @abstractmethod
    def get_model_and_processor(self, cfg: dict) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def collate_fn(self, examples) -> dict:
        pass

    @abstractmethod
    def get_peft_target_modules(self, cfg: dict = None) -> list[str]:
        pass

    @abstractmethod
    def parse_model_output(self, text: str) -> Target:
        pass

    def supports_qlora(self) -> bool:
        return True