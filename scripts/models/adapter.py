# src/models/adapter.py

from abc import ABC, abstractmethod
from typing import Any
from scripts.data.canonical_schema import DatasetInfo
from transformers import PreTrainedModel, ProcessorMixin
from scripts.core.output_parsers import model_output_parsing, ParseResult


class VLMAdapter(ABC):
    model: PreTrainedModel
    processor: ProcessorMixin
    dataset_info: DatasetInfo

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

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_tokenizer(self):
        return self.tokenizer

    def get_collate_fn(self):
        return self.collate_fn

    #Return model memory footprint in GB
    def get_memory_footprint(self):
        # .get_memory_footprint() returns byte
        bytes_size = self.model.get_memory_footprint()
        # Conversion in Gigabyte (1024^3)
        return bytes_size / (1024 ** 3)
