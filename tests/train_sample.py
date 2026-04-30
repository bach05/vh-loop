import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.models.qwen3_5_adapter import Qwen3_5Adapter
from scripts.training.backends.hf_trl import HFSFTBackend
from peft import LoraConfig

from scripts.data.hf_dataset import canonical_manifest_to_hf_sft

@hydra.main(version_base=None, config_path="../configs", config_name="train_entrypoint")
def main(cfg: DictConfig) -> None:

    #Load data
    json_file = "/media/iaslab/data_bacchin/panizzolo/paniz_train_04_02_SINGLE.vh_loop.jsonl"
    dataset_hf = canonical_manifest_to_hf_sft(json_file)

    print(dataset_hf)
    print(f'Cahce file: {dataset_hf.cache_files}')
    print(f'Fingerprint: {dataset_hf._fingerprint}')

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

    trainer = HFSFTBackend(model, processor, cfg.trainer)
    trainer.train(train_dataset=dataset_hf)

if __name__ == "__main__":
    main()