# src/models/adapter.py

from abc import ABC, abstractmethod
from typing import Any
from scripts.data.schema import VLMSample, Target
from transformers import PreTrainedModel, ProcessorMixin
from scripts.core.output_parsers import parse_out_text_json_objects_to_target


class VLMAdapter(ABC):
    model: PreTrainedModel
    processor: ProcessorMixin

    @abstractmethod
    def collate_fn(self, examples) -> dict:
        pass

    @abstractmethod
    def get_peft_target_modules(self, cfg: dict = None) -> list[str]:
        pass

    def parse_model_output(self, text: str, img_size: tuple[int, int], cat_to_id: dict[str, int]) -> tuple[Target, dict]:
        return parse_out_text_json_objects_to_target(text, img_size, category_to_id=cat_to_id)

    def get_model_and_processor(self, cfg: dict) -> tuple[Any, Any]:
        return self.model, self.processor

    #Return model memory footprint in GB
    def get_memory_footprint(self):
        # .get_memory_footprint() returns byte
        bytes_size = self.model.get_memory_footprint()
        # Conversion in Gigabyte (1024^3)
        return bytes_size / (1024 ** 3)

    def supports_qlora(self) -> bool:
        return True