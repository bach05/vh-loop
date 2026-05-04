import hydra
from omegaconf import DictConfig, OmegaConf
from scripts.core.factories import build_transform, build_peft_config

from scripts.core.registry import get_model_adapter
from scripts.training.backends.hf_trl import HFSFTBackend

from scripts.data.hf_dataset import canonical_manifest_to_hf_sft, TransformedVLMHFDataset

@hydra.main(version_base=None, config_path="../configs", config_name="train_entrypoint")
def main(cfg: DictConfig) -> None:

    #Load data
    json_file = "/media/iaslab/data_bacchin/panizzolo/paniz_train_04_02_SINGLE.vh_loop.jsonl"
    dataset_root = "/media/iaslab/data_bacchin/panizzolo"
    dataset_hf = canonical_manifest_to_hf_sft(json_file, dataset_root=dataset_root)

    print(dataset_hf)
    print(f'Cache file: {dataset_hf.cache_files}')
    print(f'Fingerprint: {dataset_hf._fingerprint}')

    #transform
    image_transforms = build_transform(cfg.transform)
    train_dataset = TransformedVLMHFDataset(
        dataset_hf,
        transform=image_transforms,
    )

    #Model config
    adapter = get_model_adapter(cfg.model.adapter, cfg=cfg.model.params)

    #Trainer config

    peft_config = build_peft_config(cfg.peft)

    trainer = HFSFTBackend(adapter, cfg.trainer, peft_config=peft_config)
    trainer.train(train_dataset=train_dataset, collator=adapter.collate_fn)

if __name__ == "__main__":
    main()