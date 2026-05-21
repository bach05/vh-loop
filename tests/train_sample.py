import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os
from scripts.core.factories import build_peft_config, build_hf_datasets
from scripts.core.factories import DatasetBuildError
from scripts.data.utils import train_val_split
from scripts.data.canonical_schema.io_utils import read_dataset_info

import scripts.models  # Ensure model adapters are registered
from scripts.core.registry import get_model_adapter
from scripts.training.backends.hf_trl import HFSFTBackend

import logging

OmegaConf.register_new_resolver(
    "strip_null",
    lambda val: f"_{val}" if val is not None else ""
)

@hydra.main(version_base=None, config_path="../configs", config_name="train_entrypoint")
def main(cfg: DictConfig) -> None:

    #log level
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if cfg.debug:
        logging.warning("\n***********************\n*** Debug mode is ON ***\n***********************")

    # output dir
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.run.dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.info(f"Output directory: {out_dir}")

    #Load data

    dataset_info = read_dataset_info(cfg.dataset.training[0].jsonl_path)
    logging.info(f"Dataset info loaded: {dataset_info.dataset_id}, {dataset_info.message_build_info}")
    # NB: HERE WE ARE ASSUMING ALL THE DATASET TO FOLLOW THE SAME SCHEMA, BUT NOBODY IS CHECKING.
    # In general some fields may be different, while some fields must agree.

    train_dataset = build_hf_datasets(cfg.dataset, transform_cfg=cfg.transform, split='training')
    try:
        valid_dataset = build_hf_datasets(cfg.dataset, transform_cfg=cfg.transform, split='validation')
    except DatasetBuildError as e:
        logging.warning(f"Validation dataset not found or failed to build: {e} Splitting train dataset..")
        train_dataset, valid_dataset = train_val_split(train_dataset)

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(valid_dataset)}")

    #Model config
    adapter = get_model_adapter(cfg.model.adapter,
                                model_cfg=cfg.model.params,
                                dataset_info=dataset_info,
                                quantization_config=cfg.get('quantization', None))

    #memory footprint
    logging.info(f"Model memory footprint: {adapter.get_memory_footprint():.2f} GB VRAM")

    #Trainer config
    peft_config = build_peft_config(cfg.peft)

    trainer = HFSFTBackend(adapter, cfg.trainer, peft_config=peft_config, out_dir=out_dir)
    trainer.setup_trainer(train_dataset=train_dataset, eval_dataset=valid_dataset, collator=None, debug=cfg.debug)

    # if cfg.debug:
    #     trainer.compute_dataset_statistics(train_dataset, split_name='train', max_samples=500, batch_size=cfg.trainer.per_device_train_batch_size)

    logging.info("\n***********************\n*** Starting Training ***\n***********************")
    trainer.train()

if __name__ == "__main__":
    main()