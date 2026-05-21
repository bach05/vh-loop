# src/models/adapter.py

from abc import ABC, abstractmethod
from typing import Any
from data.canonical_schema import DatasetInfo
from transformers import PreTrainedModel, ProcessorMixin
from scripts.core.output_parsers import model_output_parsing, ParseResult


class VLMAdapter(ABC):
    model: PreTrainedModel
    processor: ProcessorMixin
    dataset_info: DatasetInfo

    @abstractmethod
    def collate_fn(self, examples) -> dict:
        pass

    @abstractmethod
    def get_peft_target_modules(self, cfg: dict = None) -> list[str]:
        pass

    def parse_model_output(self, model_output: str, img_size: tuple[int, int], **kwargs) -> ParseResult:
        return model_output_parsing(
            model_output,
            img_size=img_size,
            answer_format=self.dataset_info.message_build_info.answer_format,
            norm_factor=self.dataset_info.message_build_info.normalization_factor,
            **kwargs
        )

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