"""
test_sample.py - Inference script for the canonical_schema v2 dataset.

This script:
1. Loads a pretrained model and optional PEFT adapter.
2. Loads a canonical_schema v2 JSONL dataset.
3. Builds prompt-only inference rows through DataSample.sample_to_message().
4. Parses model outputs according to DatasetInfo.message_build_info.answer_format.
5. Writes predictions as canonical_schema v2 JSONL:
   - first line: DatasetInfoRecord
   - following lines: DataRecord(sample=<same sample structure, predicted annotations>)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.core.factories import DatasetBuildError, build_transform
from scripts.core.registry import get_model_adapter
from scripts.data.canonical_dataset import CanonicalDataset, InferenceCanonicalDataset
from scripts.data.canonical_schema.annotations import InstanceAnnotation
from scripts.data.canonical_schema.dataset_header import AnnotationInfo, DatasetInfo
from scripts.data.canonical_schema.records import DataRecord, DatasetInfoRecord

import scripts.models #register the adapters
from scripts.core import registry
print("registered adapters:", list(registry._MODEL_ADAPTERS.keys()))

# Recommended parser for the new target grammar, e.g.:
#   <class_name,x1,y1,x2,y2>;<class_name,x1,y1,x2,y2>
# If your parser lives in another module, adjust this import.
from scripts.core.output_parsers import model_output_parsing



# Register custom Hydra resolvers only once.
if not OmegaConf.has_resolver("strip_null"):
    OmegaConf.register_new_resolver(
        "strip_null",
        lambda val: f"_{val}" if val is not None else "",
    )


def make_inference_collate_fn(adapter_collate_fn):
    """Wrap the adapter collate function while preserving canonical sample indices."""

    def collate_fn(examples):
        sample_indices = [ex["_sample_idx"] for ex in examples]
        model_examples = [
            {k: v for k, v in ex.items() if not k.startswith("_")}
            for ex in examples
        ]
        batch = adapter_collate_fn(model_examples)
        return batch, sample_indices

    return collate_fn


def build_testing_transform(transform_cfg):
    """Build the transform used for inference."""

    if transform_cfg is None:
        return None

    if "testing" in transform_cfg:
        transform_cfg = transform_cfg["testing"]

    return build_transform(transform_cfg)


def reconstruct_checkpoint_path(cfg: DictConfig) -> str:
    """Find the latest training checkpoint corresponding to the Hydra testing run."""

    hydra_cfg = HydraConfig.get()
    testing_dir = hydra_cfg.run.dir
    training_base_dir = testing_dir.replace("/testing/", "/training/")
    checkpoint_dir = Path(training_base_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {checkpoint_dir}\n"
            f"Expected checkpoint at: {training_base_dir}/checkpoint-*"
        )

    checkpoint_paths = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in: {checkpoint_dir}\n"
            f"Available contents: {list(checkpoint_dir.iterdir())}"
        )

    latest_checkpoint = checkpoint_paths[-1]
    logging.info("Using checkpoint: %s", latest_checkpoint)
    return str(latest_checkpoint)


def get_dataset_info(canonical_dataset: CanonicalDataset) -> DatasetInfo:
    """Return DatasetInfo from CanonicalDataset with compatibility fallbacks."""

    if hasattr(canonical_dataset, "get_dataset_info"):
        return canonical_dataset.get_dataset_info()

    if getattr(canonical_dataset, "info", None) is not None:
        return canonical_dataset.info

    raise RuntimeError("Canonical dataset has no DatasetInfo.")


def label_name_to_id_map(dataset_info: DatasetInfo) -> dict[str, int]:
    """Build {label_name: label_id} from DatasetInfo.label_info."""

    return {
        label_name: int(info.label_id)
        for label_name, info in dataset_info.label_info.items()
    }


def first_image_asset(sample):
    """Return the first image asset from a v2 DataSample."""

    assets = getattr(sample, "assets", None)
    if not assets:
        raise ValueError(f"Sample {getattr(sample, 'sample_id', '<unknown>')} has no assets.")

    return assets[0]


def parse_generated_output(
    generated_text: str,
    *,
    img_size: tuple[int, int],
    dataset_info: DatasetInfo,
) -> tuple[list[InstanceAnnotation], dict[str, Any]]:
    """Parse the model text into canonical_schema v2 InstanceAnnotation objects."""

    build_info = dataset_info.message_build_info
    label_to_id = label_name_to_id_map(dataset_info)

    result = model_output_parsing(
        generated_text,
        img_size=img_size,
        answer_format=build_info.answer_format,
        label_name_to_id=label_to_id,
        norm_factor=build_info.normalization_factor,
    )

    # New parser API.
    if hasattr(result, "annotations"):
        annotations = result.annotations
        debug = result.to_debug_dict() if hasattr(result, "to_debug_dict") else {}
        return annotations, debug

    # Compatibility with tuple-returning parsers.
    if isinstance(result, tuple) and len(result) == 2:
        maybe_target, debug = result
        if hasattr(maybe_target, "instances"):
            raise TypeError(
                "model_output_parsing returned the old Target/Annotation structure. "
                "Please migrate output_parsers.py to return InstanceAnnotation objects."
            )
        return maybe_target, debug

    raise TypeError(f"Unsupported model_output_parsing result type: {type(result)}")


def build_prediction_record(
    *,
    original_sample,
    predicted_annotations: list[InstanceAnnotation],
    generated_text: str,
    parse_debug: dict[str, Any],
) -> DataRecord:
    """Copy the original DataSample and replace active annotations with predictions."""

    prediction_sample = original_sample.model_copy(deep=True)
    asset = prediction_sample.assets[0]

    asset.annotations = predicted_annotations
    asset.metadata = dict(asset.metadata or {})
    asset.metadata["prediction"] = {
        "source": "model_inference",
        "generated_text": generated_text,
        "parse_debug": parse_debug,
        "num_predicted_instances": len(predicted_annotations),
    }

    return DataRecord(sample=prediction_sample)


def build_prediction_dataset_info(
    *,
    source_info: DatasetInfo,
    model_name: str | None,
    checkpoint_path: str | None,
) -> DatasetInfo:
    """Create a DatasetInfo header for the prediction JSONL."""

    info = source_info.model_copy(deep=True)
    info.description = f"Predictions for dataset {source_info.dataset_id}"
    info.annotation_info = AnnotationInfo(
        source_type="ai",
        quality="auto",
        notes="Model predictions generated by test_sample.py.",
    )

    info.metadata = dict(info.metadata or {})
    info.metadata["prediction_run"] = {
        "source_dataset_id": source_info.dataset_id,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
    }

    return info


@hydra.main(version_base=None, config_path="../configs", config_name="test_entrypoint")
def main(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if cfg.debug:
        logging.warning("\n\n************************\n*** Debug mode is ON ***\n************************\n\n")

    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.run.dir
    if not cfg.use_adapter:
        out_dir = out_dir + "_ORI_MODEL"
    os.makedirs(out_dir, exist_ok=True)
    logging.info("Output directory: %s", out_dir)

    # =========================================================================
    # 2. Load test dataset
    # =========================================================================
    logging.info("Loading test dataset...")

    test_dataset_cfg = cfg.dataset.get("testing", [])
    if not test_dataset_cfg:
        raise DatasetBuildError("No datasets specified in config under 'dataset.testing'.")

    if len(test_dataset_cfg) > 1:
        raise DatasetBuildError(
            "This inference script currently supports one testing manifest at a time. "
            f"Found {len(test_dataset_cfg)} entries under 'dataset.testing'."
        )

    test_data_cfg = test_dataset_cfg[0]

    jsonl_path = test_data_cfg.get("jsonl_path", None)
    if jsonl_path is None:
        raise DatasetBuildError(
            f"Dataset '{test_data_cfg.get('id', None)}' is missing required 'jsonl_path'."
        )

    dataset_root = test_data_cfg.get("root_override", None)
    prompting_schema = test_data_cfg.get(
        "prompting_schema",
        test_data_cfg.get("dataset_schema", "conversational"),
    )

    try:
        canonical_dataset = CanonicalDataset(jsonl_path)
        dataset_info = get_dataset_info(canonical_dataset)

        image_transform = build_testing_transform(cfg.get("transform", None))

        test_dataset = InferenceCanonicalDataset(
            canonical_dataset,
            prompting_schema=prompting_schema,
            dataset_root=dataset_root,
            transform=image_transform,
        )

    except Exception as e:
        logging.error("Failed to load test dataset: %s", e)
        raise

    logging.info("Test dataset size: %d", len(test_dataset))

    # =========================================================================
    # 1. Load pretrained model and adapter
    # =========================================================================
    logging.info("Loading model and adapter...")

    checkpoint_path = reconstruct_checkpoint_path(cfg)

    adapter = get_model_adapter(
        cfg.model.adapter,
        model_cfg=cfg.model.params,
        quantization_config=cfg.get("quantization", None),
        dataset_info=dataset_info
    )

    base_model, processor = adapter.get_model_and_processor()

    if cfg.use_adapter and cfg.peft is not None and cfg.peft.strategy.lower() == "lora":
        logging.info("Loading LoRA adapter from: %s", checkpoint_path)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logging.info("LoRA adapter loaded.")
    else:
        model = base_model
        logging.info("No PEFT adapter configured, using base model.")

    model.eval()
    torch.set_grad_enabled(False)

    device = next(model.parameters()).device
    logging.info("Model device: %s", device)
    logging.info("Model memory footprint: %.2f GB VRAM", adapter.get_memory_footprint())

    # =========================================================================
    # 3. Create dataloader
    # =========================================================================
    batch_size = cfg.trainer.per_device_eval_batch_size

    if cfg.get("debug", False):
        max_samples = cfg.get("debug_max_samples", 32)
        debug_dataset = torch.utils.data.Subset(
            test_dataset,
            range(min(max_samples, len(test_dataset))),
        )
        dataloader = torch.utils.data.DataLoader(
            debug_dataset,
            batch_size=batch_size,
            collate_fn=make_inference_collate_fn(adapter.collate_fn),
            shuffle=False,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=make_inference_collate_fn(adapter.collate_fn),
            shuffle=False,
        )

    logging.info("Batch size: %s", batch_size)
    logging.info("Number of batches: %s", len(dataloader))

    # =========================================================================
    # 4. Inference and prediction collection
    # =========================================================================
    logging.info("Starting inference...")
    time.sleep(0.5)

    predictions: list[DataRecord] = []
    parse_errors: list[dict[str, Any]] = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (batch, sample_indices) in enumerate(tqdm(dataloader, desc="Inference")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            outputs = model.generate(
                **batch,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

            if not getattr(model.config, "is_encoder_decoder", False) and "input_ids" in batch:
                input_len = batch["input_ids"].shape[1]
                generated_ids = outputs[:, input_len:]
            else:
                generated_ids = outputs

            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            batch_samples = [canonical_dataset[i] for i in sample_indices]

            for original_sample, generated_text in zip(batch_samples, generated_texts):
                total_samples += 1

                try:
                    image_asset = first_image_asset(original_sample)
                    img_size = image_asset.size  # (width, height)

                    predicted_annotations, parse_debug = parse_generated_output(
                        generated_text,
                        img_size=img_size,
                        dataset_info=dataset_info,
                    )

                    prediction_record = build_prediction_record(
                        original_sample=original_sample,
                        predicted_annotations=predicted_annotations,
                        generated_text=generated_text,
                        parse_debug=parse_debug,
                    )
                    predictions.append(prediction_record)

                except Exception as e:
                    sample_id = getattr(original_sample, "sample_id", None)
                    logging.exception("Failed to parse output for sample %s", sample_id)
                    parse_errors.append(
                        {
                            "sample_id": sample_id,
                            "error": str(e),
                            "generated_text": generated_text,
                        }
                    )

    # =========================================================================
    # 5. Write predictions to JSONL
    # =========================================================================
    output_file = Path(out_dir) / "predictions.jsonl"
    logging.info("Writing predictions to: %s", output_file)

    prediction_info = build_prediction_dataset_info(
        source_info=dataset_info,
        model_name=str(cfg.model.adapter),
        checkpoint_path=checkpoint_path,
    )

    with output_file.open("w", encoding="utf-8") as f:
        dataset_info_record = DatasetInfoRecord(info=prediction_info)
        f.write(dataset_info_record.model_dump_json(exclude_none=True))
        f.write("\n")

        for record in predictions:
            f.write(record.model_dump_json(exclude_none=True))
            f.write("\n")

    # =========================================================================
    # 6. Log summary
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("Inference completed.")
    logging.info("=" * 60)
    logging.info("Total samples processed: %d", total_samples)
    logging.info("Predictions written: %d", len(predictions))
    logging.info("Predictions saved to: %s", output_file)

    if parse_errors:
        logging.warning("Parse errors encountered: %d", len(parse_errors))
        error_file = Path(out_dir) / "parse_errors.json"
        with error_file.open("w", encoding="utf-8") as f:
            json.dump(parse_errors, f, indent=2, ensure_ascii=False)
        logging.info("Parse errors logged to: %s", error_file)

    logging.info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
