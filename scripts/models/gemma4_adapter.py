# scripts/models/qwen_3_5_adapter.py

from scripts.core.registry import register_model_adapter

from scripts.models.adapter import VLMAdapter
from scripts.core.output_parsers import parse_out_text_json_objects_to_target

@register_model_adapter("gemma4")
class Gemma4Adapter(VLMAdapter):

    def __init__(self, cfg, quantization_config=None):
        from transformers import  AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            cfg["model_name_or_path"],
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_name_or_path"],
            dtype=cfg.get("dtype", "auto"),
            device_map=cfg.get("device_map", "auto"),
            quantization_config=quantization_config,
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

    def get_peft_target_modules(self, cfg_lora=None):

        if cfg_lora is None:
            return "all-linear"
        else:
            return cfg_lora.get("target_modules", "all-linear")

    def parse_model_output(self, text):
        return parse_out_text_json_objects_to_target(text)