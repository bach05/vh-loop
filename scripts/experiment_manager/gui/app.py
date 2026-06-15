import os
import sys
from pathlib import Path
from nicegui import ui

# Ensure we can import scripts modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiment_manager.services.config_service import ConfigService
from scripts.experiment_manager.services.execution_service import ExecutionService
from scripts.experiment_manager.specs.model_specs import ModelSpec
from scripts.experiment_manager.specs.dataset_specs import DatasetSpec
from scripts.experiment_manager.specs.trainer_specs import TrainerSpec
from scripts.experiment_manager.specs.backend_specs import LocalBackendSpec, SlurmBackendSpec
from scripts.experiment_manager.specs.experiment_specs import ExperimentSpec
from scripts.experiment_manager.gui.components import render_pydantic_form

# Initialize Services
RUNNING_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = Path(f"{os.getenv('MODEL_PATH', '/models')}/vhloop")
configs_dir = RUNNING_DIR / "configs"
config_service = ConfigService(str(configs_dir))
execution_service = ExecutionService(str(RUNNING_DIR), str(OUT_DIR))

# App State
app_state = {
    "exp_name": "",
    "selected_model_file": "",
    "selected_dataset_file": "generic_waste3.yaml",
    "selected_trainer_file": "qwen_sft_trainer.yaml",
    "model_data": {},
    "dataset_data": {},
    "trainer_data": {},
    "backend_type": "local",
    "backend_data": {},
}

def load_selected_configs():
    if app_state["selected_model_file"]:
        try:
            model_spec = config_service.load_spec("model", app_state["selected_model_file"], ModelSpec)
            app_state["model_data"] = model_spec.model_dump()
        except Exception as e:
            ui.notify(f"Error loading model: {e}", type="negative")
            
    if app_state["selected_dataset_file"]:
        try:
            dataset_spec = config_service.load_spec("dataset", app_state["selected_dataset_file"], DatasetSpec)
            app_state["dataset_data"] = dataset_spec.model_dump()
        except Exception as e:
            ui.notify(f"Error loading dataset: {e}", type="negative")
            
    if app_state["selected_trainer_file"]:
        try:
            trainer_spec = config_service.load_spec("trainer", app_state["selected_trainer_file"], TrainerSpec)
            app_state["trainer_data"] = trainer_spec.model_dump()
        except Exception as e:
            ui.notify(f"Error loading trainer: {e}", type="negative")

    # Refresh the form container
    form_container.refresh()

@ui.refreshable
def form_container():
    if not app_state["selected_model_file"]:
        ui.label("Please select configurations first.").classes("text-gray-500 italic text-center w-full py-8")
        return

    with ui.column().classes("w-full gap-4"):
        with ui.tabs().classes('w-full') as tabs:
            model_tab = ui.tab('Model')
            dataset_tab = ui.tab('Dataset')
            trainer_tab = ui.tab('Trainer')
            backend_tab = ui.tab('Backend')
            
        with ui.tab_panels(tabs, value=model_tab).classes('w-full bg-transparent'):
            with ui.tab_panel(model_tab):
                render_pydantic_form(ModelSpec, app_state["model_data"], on_change=form_container.refresh)
            with ui.tab_panel(dataset_tab):
                render_pydantic_form(DatasetSpec, app_state["dataset_data"], on_change=form_container.refresh)
            with ui.tab_panel(trainer_tab):
                render_pydantic_form(TrainerSpec, app_state["trainer_data"], on_change=form_container.refresh)
            with ui.tab_panel(backend_tab):
                # Backend selector
                with ui.row().classes('w-full items-center justify-between mb-4'):
                    ui.label('Backend Type').classes('text-sm font-medium')
                    ui.select(["local", "slurm"], value=app_state["backend_type"]).bind_value(app_state, "backend_type").on_value_change(form_container.refresh)
                
                if app_state["backend_type"] == "local":
                    if "type" not in app_state["backend_data"] or app_state["backend_data"]["type"] != "local":
                        app_state["backend_data"] = {"type": "local", "cuda_visible_devices": "0"}
                    render_pydantic_form(LocalBackendSpec, app_state["backend_data"], on_change=form_container.refresh)
                else:
                    if "type" not in app_state["backend_data"] or app_state["backend_data"]["type"] != "slurm":
                        app_state["backend_data"] = {
                            "type": "slurm",
                            "partition": "gpu",
                            "nodes": 1,
                            "ntasks": 1,
                            "cpus_per_task": 8,
                            "gpus_per_task": 1,
                            "mem": "64G",
                            "time": "24:00:00"
                        }
                    render_pydantic_form(SlurmBackendSpec, app_state["backend_data"], on_change=form_container.refresh)

def _build_experiment_spec() -> tuple[ExperimentSpec, list[str]]:
    model_spec = ModelSpec.model_validate(app_state["model_data"])
    dataset_spec = DatasetSpec.model_validate(app_state["dataset_data"])
    trainer_spec = TrainerSpec.model_validate(app_state["trainer_data"])
    
    if app_state["backend_type"] == "local":
        backend_spec = LocalBackendSpec.model_validate(app_state["backend_data"])
    else:
        backend_spec = SlurmBackendSpec.model_validate(app_state["backend_data"])
        
    experiment_spec = ExperimentSpec(
        name=app_state["exp_name"],
        model=model_spec,
        dataset=dataset_spec,
        trainer=trainer_spec,
        backend=backend_spec
    )
    
    hydra_overrides = [
        f"model={app_state['selected_model_file'].replace('.yaml', '')}",
        f"dataset={app_state['selected_dataset_file'].replace('.yaml', '')}",
        f"trainer={app_state['selected_trainer_file'].replace('.yaml', '')}",
        f"exp={app_state['exp_name']}"
    ]
    return experiment_spec, hydra_overrides

def save_experiment_only():
    try:
        spec, overrides = _build_experiment_spec()
        exp_dir = execution_service.prepare_experiment(spec, overrides)
        ui.notify(f"Experiment saved/prepared at: {exp_dir}", type="positive")
    except Exception as e:
        ui.notify(f"Validation or Saving Error: {e}", type="negative")

def launch_experiment():
    try:
        spec, overrides = _build_experiment_spec()
        result, exp_dir = execution_service.run_experiment(spec, overrides)
        ui.notify(f"Experiment launched! Output at: {exp_dir}", type="positive")
    except Exception as e:
        ui.notify(f"Validation or Execution Error: {e}", type="negative")

# Build GUI layout
ui.query('.q-page').classes('bg-slate-50')

with ui.header().classes('bg-indigo-700 text-white p-4 flex items-center justify-between shadow-md'):
    ui.label('Experiment Configuration Manager').classes('text-2xl font-bold tracking-tight')
    ui.icon('settings').classes('text-3xl')

with ui.column().classes('w-full max-w-5xl mx-auto p-6 gap-6'):
    # Selection card
    with ui.card().classes('w-full shadow-sm p-6'):
        ui.label('Select Base Configurations').classes('text-xl font-semibold text-slate-800 mb-4')
        with ui.row().classes('w-full gap-4'):
            model_files = config_service.list_configs("model")
            dataset_files = config_service.list_configs("dataset")
            trainer_files = config_service.list_configs("trainer")

            ui.select(model_files, label="Model Template").bind_value(app_state, "selected_model_file").classes('flex-1')
            ui.select(dataset_files, label="Dataset Template").bind_value(app_state, "selected_dataset_file").classes('flex-1')
            ui.select(trainer_files, label="Trainer Template").bind_value(app_state, "selected_trainer_file").classes('flex-1')
            
        with ui.row().classes('w-full items-center justify-between mt-4 border-t pt-4'):
            ui.input("Experiment Suffix").bind_value(app_state, "exp_name").classes('w-64')
            ui.button("Load Configurations", on_click=load_selected_configs).classes('bg-indigo-600 text-white font-medium px-6 py-2 rounded shadow hover:bg-indigo-700 transition')

    # Config details card
    with ui.card().classes('w-full shadow-sm p-6'):
        ui.label('Configure and Customize').classes('text-xl font-semibold text-slate-800 mb-2')
        form_container()

    # Run Card
    with ui.row().classes('w-full justify-end gap-4 mt-2'):
        ui.button("Save Config Only", on_click=save_experiment_only).classes('bg-blue-600 text-white font-bold px-8 py-3 rounded-lg shadow-md hover:bg-blue-700 transition')
        ui.button("Launch Experiment", on_click=launch_experiment).classes('bg-emerald-600 text-white font-bold px-8 py-3 rounded-lg shadow-md hover:bg-emerald-700 transition')

ui.run(port=8080, title="Experiment Manager", show=False)
