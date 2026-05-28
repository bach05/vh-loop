# scripts/models/qwen_3_5_adapter.py
import logging

from omegaconf import OmegaConf
from scripts.core.registry import register_model_adapter
from scripts.core.constants import SUPPORTED_TRAIN_LIB, running_env

from scripts.models.adapter import VLMAdapter
from scripts.data.canonical_schema import DatasetInfo
from torch import bfloat16 as bf16

@register_model_adapter("qwen3_5")
class Qwen3_5Adapter(VLMAdapter):

    def __init__(self, model_cfg, dataset_info: DatasetInfo ):

        processor_params = model_cfg.get("processor_params", {})
        processor_params = OmegaConf.to_container(processor_params) if processor_params else None

        model_params = model_cfg.get("model_params", {})
        model_params = OmegaConf.to_container(model_params) if model_params else None

        quantization_params = model_cfg.get("quantization_params", {})
        quantization_params = OmegaConf.to_container(quantization_params) if quantization_params else None

        self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        ######################################################
        # HF TRANSFORMERS TRACK ##############################
        ######################################################
        if running_env.IN_USE_TRAIN_LIB == "HFTransformers":
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig

            if not processor_params:
                raise ValueError("processor_params cannot be None")
            if not model_params:
                raise ValueError("model_params cannot be None")

            self.processor = AutoProcessor.from_pretrained(
                model_cfg["model_name_or_path"],
                trust_remote_code=model_cfg.get("trust_remote_code", True),
                **processor_params,
            )

            self.tokenizer = self.processor.tokenizer

            if quantization_params is not None:
                # Add model-specific params
                quantization_params['bnb_4bit_compute_dtype'] = model_params['dtype'] if 'dtype' in model_params else bf16
                quantization_params['bnb_4bit_quant_storage'] = model_params['dtype'] if 'dtype' in model_params else bf16
                quantization_config = BitsAndBytesConfig(**quantization_params)
            else:
                quantization_config = None

            model_params['quantization_config'] = quantization_config

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_cfg["model_name_or_path"],
                **model_params,
            )

            #collate_fn
            def collate_fn(examples):
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

            self.collate_fn = collate_fn

        ######################################################
        # UNSLOTH TRACK #####################################
        ######################################################
        elif running_env.IN_USE_TRAIN_LIB == "unsloth":
            from unsloth import FastVisionModel, UnslothVisionDataCollator

            if not model_params:
                raise ValueError("model_params cannot be None")

            #rename 'Qwen/Qwen3.5-2B' to 'unsloth/Qwen3.5-2B' for unsloth
            model_name = model_cfg["model_name_or_path"]
            if not(model_name.startswith("unsloth/")):
                logging.warning(f"Model name '{model_name}' does not start with 'unsloth/'. Usually prepending 'unsloth/' for unsloth compatibility.")

            self.model, self.processor = FastVisionModel.from_pretrained(
                model_name=model_name,
                **model_params
            )

            FastVisionModel.for_training(self.model)
            self.collate_fn = UnslothVisionDataCollator(self.model, self.processor)
            self.tokenizer = self.processor.tokenizer

        else:
            raise ValueError(f"Supported libraries for training: {SUPPORTED_TRAIN_LIB}. But got {running_env.IN_USE_TRAIN_LIB}")

        self.dataset_info = dataset_info
        self.cfg = model_cfg

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_dataset_info(self):
        return self.dataset_info

    def get_tokenizer(self):
        return self.tokenizer

    def get_collate_fn(self):
        return self.collate_fn

    def get_peft_target_modules(self):
        return self.target_modules
