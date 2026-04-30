# src/training/backends/hf_trl.py

from trl import SFTTrainer, SFTConfig
from scripts.training.train_backend import TrainingBackend

class HFSFTBackend(TrainingBackend):

    def __init__(self, model, processor, cfg, peft_config = None) -> None:

        self.model = model
        self.processor = processor
        self.peft_config = peft_config

        self.cfg = cfg
        self.output_dir = './outputs'



    def train(self, train_dataset, eval_dataset=None, collator=None):

        training_args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.num_train_epochs,
            bf16=self.cfg.bf16,
            fp16=self.cfg.fp16,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            eval_strategy=self.cfg.eval_strategy,
            eval_steps=self.cfg.eval_steps,
            remove_unused_columns=False,
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
