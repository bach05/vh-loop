import hydra
from omegaconf import DictConfig, OmegaConf
from scripts.core.transform_utils import build_transform

from scripts.models.qwen3_5_adapter import Qwen3_5Adapter
from scripts.training.backends.hf_trl import HFSFTBackend
from peft import LoraConfig

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
    adapter = Qwen3_5Adapter(cfg.model)
    model, processor = adapter.get_model_and_processor()

    #Trainer config

    lora_config = LoraConfig(
        r=cfg.lora.rank_dimension,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=adapter.get_lora_target_modules(cfg.lora),
        task_type=cfg.lora.task_type,
    )

    trainer = HFSFTBackend(model, processor, cfg.trainer, peft_config=lora_config)
    trainer.train(train_dataset=train_dataset)

if __name__ == "__main__":
    main()