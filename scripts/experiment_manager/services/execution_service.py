import subprocess
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from ..specs.backend_specs import BackendSpec, LocalBackendSpec, SlurmBackendSpec
from ..specs.experiment_specs import ExperimentSpec

class ExecutionService:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.entrypoint_script = "scripts/training/sft.py"

    def prepare_experiment(self, spec: ExperimentSpec, hydra_overrides: list[str]) -> Path:
        """Creates the directory structure and saves config, sh, and slurm files without running."""
        # 1. Create directory structure
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_name = spec.model.adapter
        dataset_name = getattr(spec.dataset, "training", [None])[0]
        dataset_id = dataset_name.id if dataset_name else "unknown"
        
        folder_name = f"{date_str}_{model_name}_{dataset_id}_{spec.name}"
        exp_dir = self.workspace_dir / "experiments" / folder_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 2. Save experiment_spec.yaml
        spec_path = exp_dir / "experiment_spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec.model_dump(exclude_unset=True), f, default_flow_style=False)

        # 3. Create config.yaml (Composed Hydra options)
        composed_config = {
            "model": spec.model.model_dump(exclude_unset=True),
            "dataset": spec.dataset.model_dump(exclude_unset=True),
            "trainer": spec.trainer.model_dump(exclude_unset=True)
        }
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(composed_config, f, default_flow_style=False)

        # 4. Generate command.sh
        cmd_args = ["python", str(self.workspace_dir / self.entrypoint_script)] + hydra_overrides
        cmd_sh_path = exp_dir / "command.sh"
        with open(cmd_sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            if isinstance(spec.backend, LocalBackendSpec) and spec.backend.cuda_visible_devices:
                f.write(f"export CUDA_VISIBLE_DEVICES={spec.backend.cuda_visible_devices}\n")
            f.write(" ".join(cmd_args) + "\n")
        cmd_sh_path.chmod(0o755)

        # 5. Generate job.slurm if needed
        if isinstance(spec.backend, SlurmBackendSpec):
            slurm_content = self._generate_slurm_script(spec.backend, hydra_overrides)
            with open(exp_dir / "job.slurm", "w") as f:
                f.write(slurm_content)

        # 6. Write metadata.json
        metadata = {
            "status": "prepared",
            "prepared_time": datetime.now().isoformat(),
            "model": model_name,
            "dataset": dataset_id,
            "name": spec.name
        }
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # 7. Create empty notes.md
        notes_path = exp_dir / "notes.md"
        if not notes_path.exists():
            with open(notes_path, "w") as f:
                f.write(f"# Experiment Notes: {spec.name}\n\nCreated on {date_str}\n")
                
        return exp_dir

    def run_experiment(self, spec: ExperimentSpec, hydra_overrides: list[str]):
        """Prepares files and executes the experiment."""
        exp_dir = self.prepare_experiment(spec, hydra_overrides)
        
        # Update metadata to show launched status
        metadata_path = exp_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["status"] = "launched"
            metadata["start_time"] = datetime.now().isoformat()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

        if isinstance(spec.backend, LocalBackendSpec):
            env = os.environ.copy()
            if spec.backend.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = spec.backend.cuda_visible_devices
            cmd_sh_path = exp_dir / "command.sh"
            print(f"Executing locally: {cmd_sh_path}")
            process = subprocess.Popen([str(cmd_sh_path)], env=env, cwd=self.workspace_dir)
            return process, exp_dir
        elif isinstance(spec.backend, SlurmBackendSpec):
            slurm_file = exp_dir / "job.slurm"
            print(f"Submitting slurm job: {slurm_file}")
            result = subprocess.run(["sbatch", str(slurm_file)], capture_output=True, text=True, cwd=self.workspace_dir)
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr}")
            return result.stdout, exp_dir
        else:
            raise ValueError(f"Unknown backend type: {type(spec.backend)}")

    def _generate_slurm_script(self, backend: SlurmBackendSpec, hydra_overrides: list[str]) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={backend.partition}",
            f"#SBATCH --nodes={backend.nodes}",
            f"#SBATCH --ntasks={backend.ntasks}",
            f"#SBATCH --cpus-per-task={backend.cpus_per_task}",
            f"#SBATCH --gpus-per-task={backend.gpus_per_task}",
            f"#SBATCH --mem={backend.mem}",
            f"#SBATCH --time={backend.time}"
        ]
        if backend.qos:
            lines.append(f"#SBATCH --qos={backend.qos}")
            
        lines.append("")
        
        cmd_args = " ".join(hydra_overrides)
        
        if backend.singularity_image:
            binds = f"--bind {backend.bind_mounts} " if backend.bind_mounts else ""
            lines.append(f"singularity exec --nv {binds}{backend.singularity_image} python {self.entrypoint_script} {cmd_args}")
        else:
            lines.append(f"python {self.entrypoint_script} {cmd_args}")
            
        return "\n".join(lines)
