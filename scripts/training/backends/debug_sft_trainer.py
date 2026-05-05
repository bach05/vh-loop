import os
import logging
import traceback
import subprocess
from typing import Any, Dict

import torch
from trl import SFTTrainer


logger = logging.getLogger(__name__)


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    msg = str(exc).lower()
    return "cuda out of memory" in msg or ("out of memory" in msg and "cuda" in msg)


class OOMDebugSFTTrainer(SFTTrainer):
    def __init__(self, *args, max_logged_lengths: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_logged_lengths = max_logged_lengths
        self._last_batch_info: Dict[str, Any] = {}
        self._last_phase = "unknown"

    def _summarize_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {}

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        if torch.is_tensor(input_ids):
            info["input_ids_shape"] = tuple(input_ids.shape)

            if input_ids.ndim >= 2:
                info["batch_size"] = int(input_ids.shape[0])
                info["padded_seq_len"] = int(input_ids.shape[-1])

        if torch.is_tensor(attention_mask):
            info["attention_mask_shape"] = tuple(attention_mask.shape)

            if attention_mask.ndim >= 2:
                seq_lens = attention_mask.sum(dim=-1).detach().cpu()

                info["seq_len_min"] = int(seq_lens.min().item())
                info["seq_len_max"] = int(seq_lens.max().item())
                info["seq_len_mean"] = float(seq_lens.float().mean().item())
                info["seq_len_per_sample"] = [
                    int(x) for x in seq_lens[: self.max_logged_lengths].tolist()
                ]

        if torch.is_tensor(labels):
            info["labels_shape"] = tuple(labels.shape)

            if labels.ndim >= 2:
                supervised = (labels != -100).sum(dim=-1).detach().cpu()

                info["supervised_tokens_min"] = int(supervised.min().item())
                info["supervised_tokens_max"] = int(supervised.max().item())
                info["supervised_tokens_mean"] = float(supervised.float().mean().item())
                info["supervised_tokens_per_sample"] = [
                    int(x) for x in supervised[: self.max_logged_lengths].tolist()
                ]

        # Useful for VLMs.
        for key in [
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "cross_attention_mask",
            "token_type_ids",
            "position_ids",
        ]:
            value = inputs.get(key)
            if torch.is_tensor(value):
                info[f"{key}_shape"] = tuple(value.shape)

        # Catch custom tensor fields too.
        info["all_tensor_shapes"] = {
            key: tuple(value.shape)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

        return info

    def _cuda_memory_info(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        device = torch.cuda.current_device()
        info = {
            "cuda_available": True,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "memory_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
            "memory_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            info["memory_free_gib"] = free_bytes / 1024**3
            info["memory_total_gib"] = total_bytes / 1024**3
        except Exception:
            pass

        return info

    def _run_nvidia_smi(self) -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except Exception as exc:
            return f"Could not run nvidia-smi: {exc}"

    def _dump_oom_debug_info(self, exc: BaseException) -> None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        print("\n" + "=" * 100, flush=True)
        print(f"CUDA OOM DEBUG INFO - rank {rank}", flush=True)
        print("=" * 100, flush=True)

        print(f"\nPhase: {self._last_phase}", flush=True)
        print(f"Exception: {exc}", flush=True)

        print("\nTrainer state:", flush=True)
        print(f"global_step: {getattr(self.state, 'global_step', None)}", flush=True)
        print(f"epoch: {getattr(self.state, 'epoch', None)}", flush=True)
        print(f"max_steps: {getattr(self.state, 'max_steps', None)}", flush=True)

        print("\nTraining args:", flush=True)
        for field in [
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "eval_strategy",
            "evaluation_strategy",
            "eval_steps",
            "logging_steps",
            "remove_unused_columns",
            "dataloader_num_workers",
        ]:
            if hasattr(self.args, field):
                print(f"{field}: {getattr(self.args, field)}", flush=True)

        # TRL/SFTConfig fields vary across versions.
        for field in [
            "max_length",
            "max_seq_length",
            "packing",
            "completion_only_loss",
            "assistant_only_loss",
        ]:
            if hasattr(self.args, field):
                print(f"{field}: {getattr(self.args, field)}", flush=True)

        print("\nLast batch info:", flush=True)
        for key, value in self._last_batch_info.items():
            print(f"{key}: {value}", flush=True)

        print("\nCUDA memory info:", flush=True)
        for key, value in self._cuda_memory_info().items():
            print(f"{key}: {value}", flush=True)

        print("\nnvidia-smi:", flush=True)
        print(self._run_nvidia_smi(), flush=True)

        if torch.cuda.is_available():
            try:
                print("\ntorch.cuda.memory_summary:", flush=True)
                print(torch.cuda.memory_summary(abbreviated=True), flush=True)
            except Exception as mem_exc:
                print(f"Could not print memory_summary: {mem_exc}", flush=True)

        print("\nTraceback:", flush=True)
        print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), flush=True)

        print("=" * 100 + "\n", flush=True)

    def _record_batch(self, inputs: Dict[str, Any], phase: str) -> None:
        self._last_phase = phase
        try:
            self._last_batch_info = self._summarize_batch(inputs)
        except Exception as exc:
            self._last_batch_info = {
                "batch_summary_error": repr(exc),
                "available_keys": list(inputs.keys()) if isinstance(inputs, dict) else None,
            }

    def compute_loss(self, model, inputs, *args, **kwargs):
        """
        This catches both training and evaluation OOMs because both paths call compute_loss().
        """
        phase = "train" if model.training else "eval"
        self._record_batch(inputs, phase=f"{phase}/compute_loss")

        try:
            return super().compute_loss(model, inputs, *args, **kwargs)

        except Exception as exc:
            if is_cuda_oom(exc):
                self._dump_oom_debug_info(exc)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise

    def training_step(self, model, inputs, *args, **kwargs):
        self._record_batch(inputs, phase="train/training_step")

        try:
            return super().training_step(model, inputs, *args, **kwargs)

        except Exception as exc:
            if is_cuda_oom(exc):
                self._dump_oom_debug_info(exc)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        This is the eval/test path.
        """
        self._record_batch(inputs, phase="eval/prediction_step")

        try:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        except Exception as exc:
            if is_cuda_oom(exc):
                self._dump_oom_debug_info(exc)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise