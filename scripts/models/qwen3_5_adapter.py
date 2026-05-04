# scripts/models/qwen_3_5_adapter.py

from scripts.core.registry import register_model_adapter

from scripts.models.adapter import VLMAdapter
from scripts.core.output_parsers import parse_out_text_json_objects_to_target

from PIL import Image

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

    def collate_fn(self, examples):
        texts = []
        images = []

        for example in examples:
            text = self.processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )

            texts.append(text.strip())
            images.append(example["images"])

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.processor.tokenizer.boi_token_id] = -100
        labels[labels == self.processor.tokenizer.image_token_id] = -100
        labels[labels == self.processor.tokenizer.eoi_token_id] = -100

        batch["labels"] = labels
        return batch

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