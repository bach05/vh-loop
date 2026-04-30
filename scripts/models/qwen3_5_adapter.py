# scripts/models/qwen_3_5_adapter.py

from scripts.core.registry import register_model_adapter

from scripts.models.adapter import VLMAdapter
from scripts.core.output_parsers import parse_out_text_json_objects_to_target

@register_model_adapter("qwen3_5")
class Qwen3_5Adapter(VLMAdapter):

    def __init__(self, cfg):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            cfg["model_name_or_path"],
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg["model_name_or_path"],
            torch_dtype=cfg.get("torch_dtype", "auto"),
            device_map=cfg.get("device_map", "auto"),
            quantization_config=cfg.get("quantization_config"),
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        self.cfg = cfg

    def get_model_and_processor(self):
        return self.model, self.processor

    def build_collator(self):
        from transformers import DataCollatorWithPadding

        data_collator = DataCollatorWithPadding(
            tokenizer=self.processor.tokenizer,
            padding=True,  # pad to longest sequence in batch (default)
            pad_to_multiple_of=8,  # optional: efficient for Tensor Cores
            return_tensors="pt"  # return PyTorch tensors (default)
        )

        return data_collator


    def get_lora_target_modules(self, cfg_lora):
        return cfg_lora.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def parse_model_output(self, text):
        return parse_out_text_json_objects_to_target(text)