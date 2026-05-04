# src/training/backends/hf_trl.py
from pygments.lexers import ada
from trl import SFTTrainer, SFTConfig
from scripts.training.train_backend import TrainingBackend
from scripts.models.adapter import VLMAdapter

from dataclasses import fields
from omegaconf import OmegaConf

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

class HFSFTBackend(TrainingBackend):

    def __init__(self, adapter: VLMAdapter, cfg, peft_config = None) -> None:

        self.model, self.processor = adapter.get_model_and_processor()
        self.peft_config = peft_config

        peft_target_modules = adapter.get_peft_target_modules()
        if peft_target_modules is not None and peft_config is not None:
            self.peft_config.target_modules = peft_target_modules

        self.cfg = cfg
        self.output_dir = './outputs'

    def train(self, train_dataset, eval_dataset=None, collator=None):

        training_args = build_sft_config(
            cfg=self.cfg,
            output_dir=self.output_dir,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            peft_config=self.peft_config,
            data_collator=collator,
        )

        self.trainer.train()

    def save_results(self):
        self.trainer.save_model(self.output_dir)
