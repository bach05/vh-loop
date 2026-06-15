import os
import sys
import unittest
from pathlib import Path

# Ensure import works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_manager.services.config_service import ConfigService
from scripts.experiment_manager.specs import ModelSpec, DatasetSpec, TrainerSpec

class TestConfigService(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = Path(__file__).resolve().parents[1]
        self.configs_dir = self.workspace_dir / "configs"
        self.config_service = ConfigService(str(self.configs_dir))

    def test_list_configs(self):
        models = self.config_service.list_configs("model")
        self.assertTrue(len(models) > 0)
        self.assertIn("qwen3_5_unsloth.yaml", models)

    def test_load_model_spec(self):
        model_spec = self.config_service.load_spec("model", "qwen3_5_unsloth.yaml", ModelSpec)
        self.assertEqual(model_spec.adapter, "qwen3_5")
        self.assertEqual(model_spec.params.model_name_or_path, "unsloth/Qwen3.5-2B")

    def test_load_dataset_spec(self):
        dataset_spec = self.config_service.load_spec("dataset", "generic_waste.yaml", DatasetSpec)
        self.assertTrue(len(dataset_spec.training) > 0)
        self.assertEqual(dataset_spec.training[0].id, "cosmari_train_3000")

    def test_load_trainer_spec(self):
        trainer_spec = self.config_service.load_spec("trainer", "qwen_sft_trainer.yaml", TrainerSpec)
        self.assertEqual(trainer_spec.per_device_train_batch_size, 4)
        self.assertEqual(trainer_spec.bf16, True)

if __name__ == "__main__":
    unittest.main()
