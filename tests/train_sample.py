import hydra
from omegaconf import DictConfig, OmegaConf
import os
from scripts.core.factories import build_peft_config, build_hf_datasets
from scripts.core.factories import DatasetBuildError
from scripts.data.utils import train_val_split

from scripts.core.registry import get_model_adapter
from scripts.training.backends.hf_trl import HFSFTBackend

import logging

@hydra.main(version_base=None, config_path="../configs", config_name="train_entrypoint")
def main(cfg: DictConfig) -> None:

    #log level
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if cfg.debug:
        logging.warning("*******************\n*** Debug mode is ON ***\n*******************")

    #Load data

    train_dataset = build_hf_datasets(cfg.dataset, transform_cfg=cfg.transform, split='training')
    try:
        valid_dataset = build_hf_datasets(cfg.dataset, transform_cfg=cfg.transform, split='validation')
    except DatasetBuildError as e:
        logging.error(f"Validation dataset not found or failed to build: {e}, splitting train dataset..")
        train_dataset, valid_dataset = train_val_split(train_dataset)

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(valid_dataset)}")

    #Model config
    adapter = get_model_adapter(cfg.model.adapter, model_cfg=cfg.model.params, quantization_config=cfg.get('quantization', None))

    #memory footprint
    logging.info(f"Model memory footprint: {adapter.get_memory_footprint():.2f} GB VRAM")

    #Trainer config
    peft_config = build_peft_config(cfg.peft)

    trainer = HFSFTBackend(adapter, cfg.trainer, peft_config=peft_config)
    trainer.train(train_dataset=train_dataset, eval_dataset=valid_dataset, collator=None, debug=cfg.debug)

if __name__ == "__main__":
    main()