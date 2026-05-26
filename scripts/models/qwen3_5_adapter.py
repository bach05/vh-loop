# scripts/models/qwen_3_5_adapter.py
from omegaconf import OmegaConf
from scripts.core.registry import register_model_adapter
from scripts.core.constants import SUPPORTED_TRAIN_LIB

from scripts.models.adapter import VLMAdapter
from scripts.data.canonical_schema import DatasetInfo
from torch import bfloat16 as bf16

@register_model_adapter("qwen3_5")
class Qwen3_5Adapter(VLMAdapter):

    def __init__(self, model_cfg, dataset_info: DatasetInfo, quantization_config=None, train_lib: str = "transformers" ):

        if train_lib == "transformers":
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            processor_params = model_cfg.get("processor_params", {})
            processor_params = OmegaConf.to_container(processor_params)

            self.processor = AutoProcessor.from_pretrained(
                model_cfg["model_name_or_path"],
                trust_remote_code=model_cfg.get("trust_remote_code", True),
                **processor_params,
            )

            self.tokenizer = self.processor.tokenizer

            model_params = model_cfg.get("model_params", {})
            model_params = OmegaConf.to_container(model_params)

            if quantization_config is not None:
                quantization_config = OmegaConf.to_container(quantization_config)
                quantization_config['bnb_4bit_compute_dtype'] = model_params['dtype'] if 'dtype' in model_params else bf16
                quantization_config['bnb_4bit_quant_storage'] = model_params['dtype'] if 'dtype' in model_params else bf16

            model_params['quantization_config'] = quantization_config

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_cfg["model_name_or_path"],
                **model_params,
            )
        elif train_lib == "unsloth":
            from unsloth import FastVisionModel

            #rename 'Qwen/Qwen3.5-2B' to 'unsloth/Qwen3.5-2B' for unsloth
            model_name = model_cfg["model_name_or_path"]
            if model_name.startswith("Qwen/"):
                model_name = model_name.replace("Qwen/", "unsloth/")
            else:
                raise ValueError(f"Cannot replace 'Qwen/' with 'unsloth/'. The given model_name_or_path is: {model_name}")

            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=False,  # MoE QLoRA not recommended, dense 27B is fine
                load_in_16bit=True,  # bf16/16-bit LoRA
                full_finetuning=False,
            )

            FastVisionModel.for_training(self.model)
            self.processor = None

        else:
            raise ValueError(f"Supported libraries for training: {SUPPORTED_TRAIN_LIB}. But got {train_lib}")

        self.dataset_info = dataset_info

        self.cfg = model_cfg

    def get_model_and_processor(self):
        return self.model, self.processor

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_dataset_info(self):
        return self.dataset_info

    def get_tokenizer(self):
        return self.tokenizer

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
        labels[labels == self.processor.tokenizer.image_token_id] = -100

        batch["labels"] = labels
        return batch

    def get_peft_target_modules(self, cfg_lora=None):

        if cfg_lora is None:
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            return cfg_lora.get("target_modules", [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ])

