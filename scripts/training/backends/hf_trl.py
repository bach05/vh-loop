# src/training/backends/hf_trl.py

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

class HFTRLBackend:
    def train(self, cfg, adapter, train_dataset, eval_dataset=None):

        model, processor = adapter.load_model_and_processor(cfg.model)

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=adapter.get_lora_target_modules(cfg.lora),
            task_type="CAUSAL_LM",
        )

        hf_train = canonical_to_hf_sft(train_dataset.samples, adapter)

        hf_eval = None
        if eval_dataset is not None:
            hf_eval = canonical_to_hf_sft(eval_dataset.samples, adapter)

        collator = adapter.build_collator(processor, cfg.trainer)

        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
            learning_rate=cfg.trainer.learning_rate,
            num_train_epochs=cfg.trainer.num_train_epochs,
            bf16=cfg.trainer.bf16,
            fp16=cfg.trainer.fp16,
            gradient_checkpointing=cfg.trainer.gradient_checkpointing,
            logging_steps=cfg.trainer.logging_steps,
            save_steps=cfg.trainer.save_steps,
            eval_steps=cfg.trainer.eval_steps,
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_train,
            eval_dataset=hf_eval,
            processing_class=processor,
            peft_config=lora_config,
            data_collator=collator,
        )

        trainer.train()
        trainer.save_model(cfg.output_dir)

        return cfg.output_dir