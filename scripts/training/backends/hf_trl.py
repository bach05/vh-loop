# src/training/backends/hf_trl.py
from trl import SFTTrainer, SFTConfig

from scripts.core.factories import build_peft_config
from scripts.training.train_backend import TrainingBackend
from scripts.training.backends.debug_sft_trainer import OOMDebugSFTTrainer
from scripts.models.adapter import VLMAdapter
from scripts.core.constants import running_env

from dataclasses import fields
from omegaconf import OmegaConf

from post_training_toolkit import DiagnosticsCallback

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

def build_sft_config(cfg, output_dir: str) -> SFTConfig:
    valid_keys = {f.name for f in fields(SFTConfig)}

    if hasattr(cfg, "__class__") and "omegaconf" in str(type(cfg)).lower():
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    cfg_dict["output_dir"] = output_dir
    cfg_dict["remove_unused_columns"] = False

    sft_kwargs = {
        k: v
        for k, v in cfg_dict.items()
        if k in valid_keys
    }

    unknown_keys = sorted(set(cfg_dict.keys()) - valid_keys)
    if unknown_keys:
        print(f"Ignoring non-SFTConfig keys: {unknown_keys}")

    return SFTConfig(**sft_kwargs)

def get_callbacks(out_dir='./outputs'):
    return [
        DiagnosticsCallback(
            run_dir=Path(out_dir) / "ptt_run",
            stop_on_critical=True,  # failure-conditioned training control
            enable_live_warnings=True,  # emit live heuristic warnings
        )
    ]

class HFSFTBackend(TrainingBackend):

    def __init__(self, adapter: VLMAdapter, cfg, peft_config = None, out_dir='./outputs') -> None:

        self.model, self.processor = adapter.get_model(), adapter.get_processor()
        self.tokenizer = adapter.get_tokenizer()
        self.collate_fn = adapter.get_collate_fn()
        self.peft_config = peft_config

        # Add model target modules
        self.peft_target_modules = adapter.get_peft_target_modules()

        self.cfg = cfg
        self.output_dir = out_dir

        self.trainer = None

    def setup_trainer(self, train_dataset, eval_dataset=None, collator=None, debug=False):

        training_args = build_sft_config(
            cfg=self.cfg,
            output_dir=self.output_dir,
        )

        callbacks = get_callbacks(out_dir=self.output_dir)

        if running_env.IN_USE_TRAIN_LIB == "HFTransformers":
            #build peft config
            peft_params = None
            if self.peft_target_modules is not None and self.peft_config is not None:
                peft_cfg = OmegaConf.to_container(self.peft_config, resolve=True)
                peft_cfg['params']['target_modules'] = self.peft_target_modules
                peft_params = build_peft_config(peft_cfg)

            if debug:
                self.trainer = OOMDebugSFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=self.processor,
                    peft_config=peft_params,
                    data_collator=collator,
                )
            else:
                self.trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=self.processor,
                    peft_config=peft_params,
                    data_collator=collator,
                    callbacks=callbacks,
                )
        elif running_env.IN_USE_TRAIN_LIB == "unsloth":
            from unsloth import FastVisionModel

            if self.peft_config is not None:
                peft_config = OmegaConf.to_container(self.peft_config)
                self.model = FastVisionModel.get_peft_model(
                    self.model,
                    **peft_config
                )

            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.processor,
                data_collator=self.collate_fn,  # Must use!
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args
            )

    def train(self):

        if self.trainer is None:
            raise ValueError("You must call setup_trainer() before train()")

        self.trainer.train()

    def save_results(self):
        self.trainer.save_model(self.output_dir)

    def compute_dataset_statistics(
            self,
            dataset,
            split_name: str,
            collator=None,
            max_length: Optional[int] = None,
            max_samples: Optional[int] = None,
            batch_size: int = 4,
            save_json: bool = True,
            disable_pbar: bool = False,
    ) -> dict[str, Any]:
        """
        Compute useful SFT/VLM dataset statistics.

        The method computes:
        - number of samples
        - raw chat-template token lengths
        - collated input lengths, if a collator is provided
        - label lengths, if labels are produced by the collator
        - number of messages per sample
        - number of text parts per sample
        - number of image assets per sample
        - number of target instances per sample
        - label distribution, if target instances are available
        """
        from scripts.training.dataset_statistics_utils import (
            batched,
            estimate_raw_token_length,
            extract_input_lengths_from_batch,
            extract_label_lengths_from_batch,
            inspect_sample_structure,
            log_dataset_statistics,
            numeric_stats,
            select_dataset_indices,
        )
        from tqdm import tqdm

        if collator is None:
            collator = self.trainer.data_collator

        n_total = len(dataset)
        indices = select_dataset_indices(n_total, max_samples)

        raw_token_lengths = []
        collated_input_lengths = []
        collated_label_lengths = []

        n_messages = []
        n_text_parts = []
        n_image_assets = []
        n_message_image_parts = []
        n_target_instances = []

        label_counter = Counter()
        errors = []

        for idx in tqdm(indices, desc=f"Inspecting {split_name} dataset",
                        disable=disable_pbar,
                        total=len(indices)):
            try:
                sample = dataset[idx]

                sample_info = inspect_sample_structure(sample)

                n_messages.append(sample_info["num_messages"])
                n_text_parts.append(sample_info["num_text_parts"])
                n_image_assets.append(sample_info["num_image_assets"])
                n_message_image_parts.append(sample_info["num_message_image_parts"])
                n_target_instances.append(sample_info["num_target_instances"])

                label_counter.update(sample_info["labels"])

                token_len = estimate_raw_token_length(
                    sample=sample,
                    processor=self.processor,
                    include_image_tokens=True,
                    image_token_fallback=0,
                )

                if token_len is not None:
                    raw_token_lengths.append(token_len)

            except Exception as exc:
                errors.append(
                    {
                        "sample_index": int(idx),
                        "stage": "raw_inspection",
                        "error": repr(exc),
                    }
                )

        if collator is not None:
            for batch_indices in batched(indices, batch_size):
                try:
                    batch_samples = [dataset[i] for i in batch_indices]
                    batch = collator(batch_samples)

                    input_lengths = extract_input_lengths_from_batch(
                        batch=batch,
                        processor=self.processor,
                    )

                    if input_lengths is not None:
                        collated_input_lengths.extend(input_lengths)

                    label_lengths = extract_label_lengths_from_batch(batch)

                    if label_lengths is not None:
                        collated_label_lengths.extend(label_lengths)

                except Exception as exc:
                    errors.append(
                        {
                            "sample_indices": [int(i) for i in batch_indices],
                            "stage": "collator",
                            "error": repr(exc),
                        }
                    )

        stats = {
            "split": split_name,
            "num_samples_total": n_total,
            "num_samples_inspected": len(indices),
            "max_length": max_length,
            "raw_token_length": numeric_stats(
                raw_token_lengths,
                max_length=max_length,
            ),
            "collated_input_length": numeric_stats(
                collated_input_lengths,
                max_length=max_length,
            ),
            "collated_label_length": numeric_stats(
                collated_label_lengths,
                max_length=max_length,
            ),
            "messages_per_sample": numeric_stats(n_messages),
            "text_parts_per_sample": numeric_stats(n_text_parts),
            "image_assets_per_sample": numeric_stats(n_image_assets),
            "message_image_parts_per_sample": numeric_stats(n_message_image_parts),
            "target_instances_per_sample": numeric_stats(n_target_instances),
            "label_distribution": dict(label_counter),
            "num_errors": len(errors),
            "errors": errors[:20],
        }

        log_dataset_statistics(stats)

        if save_json:
            out_path = Path(self.output_dir) / f"dataset_stats_{split_name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logging.info("Saved dataset statistics to: %s", out_path)

        return stats
