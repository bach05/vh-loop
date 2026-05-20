"""
test_sample.py - Inference script for model predictions

This script:
1. Loads a pretrained model and adapter from checkpoint
2. Processes the test dataset in batches
3. Extracts model predictions (text output, bounding boxes, classes)
4. Generates a JSONL predictions file with the canonical dataset format
"""

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os, time
import torch
import logging
from pathlib import Path

from tqdm import tqdm
import json

from scripts.core.factories import build_transform
from scripts.core.factories import DatasetBuildError
from scripts.data.canonical_dataset import CanonicalVLMDataset, InferenceVLMDataset
from scripts.core.registry import get_model_adapter
from data.schema.schema import DatasetInfo, DatasetInfoRecord, SampleRecord
from scripts.data.manifest_utils import read_manifest_info

from scripts.core import registry
print("registered adapters:", list(registry._MODEL_ADAPTERS.keys()))

# Register custom Hydra resolvers
OmegaConf.register_new_resolver(
    "strip_null",
    lambda val: f"_{val}" if val is not None else ""
)

def make_inference_collate_fn(adapter_collate_fn):
    """
    Wrap the adapter collate function so we can keep track of the canonical
    sample indices without passing them to model.generate().
    """

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
    """
    Build the transform used for inference.

    Supports both:
      cfg.transform directly containing ops/to_tensor
    and:
      cfg.transform.testing containing ops/to_tensor
    """

    if transform_cfg is None:
        return None

    if "testing" in transform_cfg:
        transform_cfg = transform_cfg["testing"]

    return build_transform(transform_cfg)


def reconstruct_checkpoint_path(cfg: DictConfig) -> str:
    """
    Reconstruct the checkpoint path from exp_name by replacing 'testing' with 'training'
    in the output directory path.
    
    E.g., ${MODEL_PATH}/vhloop/testing/qwen_panizzolo_lora -> 
           ${MODEL_PATH}/vhloop/training/qwen_panizzolo_lora/checkpoint-*
    """
    hydra_cfg = HydraConfig.get()
    testing_dir = hydra_cfg.run.dir
    
    # Replace 'testing' with 'training'
    training_base_dir = testing_dir.replace("/testing/", "/training/")
    
    # Look for the latest checkpoint in the training directory
    checkpoint_dir = Path(training_base_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {checkpoint_dir}\n"
            f"Expected checkpoint at: {training_base_dir}/checkpoint-*/adapter_model.bin"
        )
    
    # Find checkpoint subdirectories (format: checkpoint-*)
    checkpoint_paths = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
    )
    
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in: {checkpoint_dir}\n"
            f"Available contents: {list(checkpoint_dir.iterdir())}"
        )
    
    # Use the latest checkpoint
    latest_checkpoint = checkpoint_paths[-1]
    logging.info(f"Using checkpoint: {latest_checkpoint}")
    
    return str(latest_checkpoint)


@hydra.main(version_base=None, config_path="../configs", config_name="test_entrypoint")
def main(cfg: DictConfig) -> None:
    
    # Setup logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    if cfg.debug:
        logging.warning("\n\n************************\n*** Debug mode is ON ***\n************************\n\n")
    
    # Get output directory
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.run.dir
    if not cfg.use_adapter:
        out_dir = out_dir + "_ORI_MODEL"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.info(f"Output directory: {out_dir}")
    
    # =========================================================================
    # 1. Load pretrained model and adapter
    # =========================================================================
    logging.info("Loading model and adapter...")
    
    checkpoint_path = reconstruct_checkpoint_path(cfg)
    
    # Load base model via adapter
    adapter = get_model_adapter(
        cfg.model.adapter,
        model_cfg=cfg.model.params,
        quantization_config=cfg.get('quantization', None),
    )
    
    base_model, processor = adapter.get_model_and_processor()
    
    # Load LoRA adapter weights if peft is configured
    if cfg.use_adapter and cfg.peft is not None and cfg.peft.strategy.lower() == "lora":
        logging.info(f"Loading LoRA adapter from: {checkpoint_path}")
        from peft import PeftModel
        
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logging.info("LoRA adapter loaded and merged")
    else:
        model = base_model
        logging.info("No PEFT adapter configured, using base model")
    
    # Set model to eval mode
    model.eval()
    torch.set_grad_enabled(False)
    
    # Get device
    device = next(model.parameters()).device
    logging.info(f"Model device: {device}")
    logging.info(f"Model memory footprint: {adapter.get_memory_footprint():.2f} GB VRAM")

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

    manifest_path = test_data_cfg.get("jsonl_path", None)
    if manifest_path is None:
        raise DatasetBuildError(
            f"Dataset '{test_data_cfg.get('id', None)}' is missing required 'jsonl_path' field."
        )

    dataset_root = test_data_cfg.get("root_override", None)
    dataset_schema = test_data_cfg.get("dataset_schema", "conversational")

    try:
        canonical_dataset = CanonicalVLMDataset(manifest_path)

        image_transform = build_testing_transform(cfg.get("transform", None))

        test_dataset = InferenceVLMDataset(
            canonical_dataset,
            dataset_schema=dataset_schema,
            dataset_root=dataset_root,
            transform=image_transform,
        )

    except Exception as e:
        logging.error(f"Failed to load test dataset: {e}")
        raise

    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Dataset info for the output JSONL header
    category_to_id = None
    try:
        dataset_info = read_manifest_info(manifest_path)
        label_info = dataset_info.label_info
        category_to_id = { cat:int(id) for id, cat in label_info.items()}
    except Exception as e:
        logging.warning(f"Could not read dataset info: {e}. Using defaults.")
        dataset_info = DatasetInfo(
            dataset_id=test_data_cfg.get("id", "unknown"),
            description="Test dataset predictions",
        )
    
    # =========================================================================
    # 3. Create dataloader with batch size from trainer config
    # =========================================================================
    batch_size = cfg.trainer.per_device_eval_batch_size

    if cfg.get('debug', False):
        # reduce the number of samples for debugging
        max_samples = cfg.get('debug_max_samples', 32)
        # Create a subset of the dataset to actually limit the samples
        debug_dataset = torch.utils.data.Subset(test_dataset, range(min(max_samples, len(test_dataset))))

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
            shuffle=False,  # Do not shuffle: sample_indices map back to canonical_dataset
        )

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Number of batches: {len(dataloader)}")
    
    # =========================================================================
    # 4. Inference and prediction collection
    # =========================================================================
    logging.info("Starting inference...")
    #Quick pause to avoid overlapping pbar
    time.sleep(0.5)
    
    predictions = []
    parse_errors = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (batch, sample_indices) in enumerate(tqdm(dataloader, desc="Inference")):
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass - generate predictions
            outputs = model.generate(
                **batch,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

            # Decode only newly generated tokens.
            # For decoder-only VLMs, generate() returns prompt + completion.
            if not getattr(model.config, "is_encoder_decoder", False) and "input_ids" in batch:
                input_len = batch["input_ids"].shape[1]
                generated_ids = outputs[:, input_len:]
            else:
                generated_ids = outputs

            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # Retrieve the original canonical VLMSample objects.
            batch_samples = [canonical_dataset[i] for i in sample_indices]

            # Process each sample in batch
            for original_sample, generated_text in zip(batch_samples, generated_texts):
                total_samples += 1

                # Get image size from the first canonical ImageAsset.
                # ImageAsset.size is stored as (width, height).
                first_image = next(iter(original_sample.images.values()), None)

                if first_image is not None:
                    img_size = first_image.size
                else:
                    img_size = (1024, 768)  # Default fallback
                
                # Parse model output to get predictions
                try:
                    predicted_target, meta_data = adapter.parse_model_output(generated_text, img_size=img_size, cat_to_id=category_to_id)
                    
                    # Convert instances to JSON format
                    if predicted_target.instances:
                        target_json = []
                        for instance in predicted_target.instances:
                            item = {
                                "label": instance.label,
                            }
                            if instance.bbox:
                                item["bbox"] = [
                                    instance.bbox.tl.x,
                                    instance.bbox.tl.y,
                                    instance.bbox.br.x,
                                    instance.bbox.br.y,
                                ]
                            target_json.append(item)
                        predicted_target.text = json.dumps(target_json, ensure_ascii=False, separators=(",", ":"))
                    else:
                        predicted_target.text = "[]"

                except Exception as e:
                    # logging.error(f"Failed to parse output for sample {original_sample.sample_id}: {e}")
                    # parse_errors.append({
                    #     "sample_idx": original_sample.sample_id,
                    #     "error": str(e),
                    #     "generated_text": generated_text,
                    # })
                    # # Create empty target on parse error
                    # predicted_target = Target(text="[]", instances=[])
                    raise e
                
                # Reconstruct sample with predictions
                prediction_sample = SampleRecord(
                    sample_id=original_sample.sample_id,
                    dataset_id=original_sample.dataset_id,
                    images=original_sample.images,
                    messages=original_sample.messages,
                    target=predicted_target,
                    metadata=original_sample.metadata,
                )
                
                predictions.append(prediction_sample)
    
    # =========================================================================
    # 5. Write predictions to JSONL
    # =========================================================================
    output_file = Path(out_dir) / "predictions.jsonl"
    logging.info(f"Writing predictions to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write dataset info as first line
        dataset_info_record = DatasetInfoRecord(
            record_type="dataset_info",
            info=dataset_info,
        )
        f.write(dataset_info_record.model_dump_json(ensure_ascii=False))
        f.write("\n")
        
        # Write each prediction sample
        for sample in predictions:
            f.write(sample.model_dump_json(ensure_ascii=False))
            f.write("\n")
    
    # =========================================================================
    # 6. Log summary
    # =========================================================================
    logging.info("\n" + "="*60)
    logging.info("Inference completed successfully!")
    logging.info("="*60)
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Predictions saved to: {output_file}")
    
    if parse_errors:
        logging.warning(f"Parse errors encountered: {len(parse_errors)}")
        error_file = Path(out_dir) / "parse_errors.json"
        with open(error_file, 'w') as f:
            json.dump(parse_errors, f, indent=2)
        logging.info(f"Parse errors logged to: {error_file}")
    
    logging.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()