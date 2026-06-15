import os
import sys
import unittest
import shutil
import tempfile
import json
from pathlib import Path

# Ensure import works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_manager.services.execution_service import ExecutionService
from scripts.experiment_manager.specs import ModelSpec, DatasetSpec, TrainerSpec, LocalBackendSpec, SlurmBackendSpec, ExperimentSpec
from scripts.experiment_manager.specs.dataset_specs import DatasetItemSpec
from scripts.experiment_manager.specs.model_specs import ModelParamsConfig, ModelParams

class TestExecutionService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir)
        self.execution_service = ExecutionService(str(self.workspace_dir))
        
        # Create a mock entrypoint script
        (self.workspace_dir / "scripts" / "training").mkdir(parents=True, exist_ok=True)
        with open(self.workspace_dir / self.execution_service.entrypoint_script, "w") as f:
            f.write("print('mock training run')\n")
            
        # Build a sample ExperimentSpec
        self.spec = ExperimentSpec(
            name="test_exp",
            model=ModelSpec(
                adapter="qwen3_5",
                params=ModelParamsConfig(
                    model_name_or_path="mock/path",
                    model_params=ModelParams(max_seq_length=2048)
                )
            ),
            dataset=DatasetSpec(
                training=[DatasetItemSpec(id="mock_train", jsonl_path="mock.jsonl")]
            ),
            trainer=TrainerSpec(num_train_epochs=1),
            backend=LocalBackendSpec(type="local", cuda_visible_devices="0")
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_experiment_local_creates_files(self):
        # We mock Popen to avoid actually starting a subprocess
        from unittest.mock import patch
        with patch('subprocess.Popen') as mock_popen:
            _, exp_dir = self.execution_service.run_experiment(self.spec, ["model=mock", "dataset=mock"])
            
            # Verify folder exists
            self.assertTrue(exp_dir.exists())
            self.assertIn("qwen3_5_mock_train_test_exp", exp_dir.name)
            
            # Verify files are created
            self.assertTrue((exp_dir / "experiment_spec.yaml").exists())
            self.assertTrue((exp_dir / "config.yaml").exists())
            self.assertTrue((exp_dir / "command.sh").exists())
            self.assertTrue((exp_dir / "metadata.json").exists())
            self.assertTrue((exp_dir / "notes.md").exists())
            
            # Verify metadata.json content
            with open(exp_dir / "metadata.json") as f:
                metadata = json.load(f)
                self.assertEqual(metadata["status"], "launched")
                self.assertEqual(metadata["model"], "qwen3_5")
                self.assertEqual(metadata["dataset"], "mock_train")

    def test_generate_slurm_script(self):
        slurm_backend = SlurmBackendSpec(
            type="slurm",
            partition="gpu",
            nodes=1,
            ntasks=2,
            cpus_per_task=4,
            gpus_per_task=1,
            mem="16G",
            time="01:00:00",
            singularity_image="/path/to/image.sif",
            bind_mounts="/data:/data"
        )
        
        script = self.execution_service._generate_slurm_script(slurm_backend, ["model=qwen"])
        self.assertIn("#SBATCH --partition=gpu", script)
        self.assertIn("#SBATCH --ntasks=2", script)
        self.assertIn("#SBATCH --cpus-per-task=4", script)
        self.assertIn("#SBATCH --mem=16G", script)
        self.assertIn("singularity exec --nv --bind /data:/data /path/to/image.sif", script)

if __name__ == "__main__":
    unittest.main()
