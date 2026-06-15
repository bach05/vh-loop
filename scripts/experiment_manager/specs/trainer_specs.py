from typing import Optional, Any, Dict
from .base import BaseSpec

class TrainerSpec(BaseSpec):
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 5
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataset_text_field: str = ""
    dataset_kwargs: Dict[str, Any] = {"skip_prepare_dataset": True}
    remove_unused_columns: bool = False
    packing: bool = False
    loss_type: str = "nll"
    assistant_only_loss: bool = False
    completion_only_loss: bool = False
    use_liger_kernel: bool = True
    logging_strategy: str = "steps"
    logging_steps: int = 50
    logging_first_step: bool = True
    disable_tqdm: bool = False
    report_to: str = "none"
    eval_strategy: str = "steps"
    eval_steps: int = 500
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
