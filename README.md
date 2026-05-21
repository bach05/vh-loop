# VLM-Human Loop

This repository contains the code for the VLM-Human Loop project, which is a framework for integrating human feedback into vision-language models (VLMs) to improve their performance on various tasks. 

# Installation and Setup 

To set up the VLM-Human Loop environment, follow these steps:

1. Install pixi package manager if you haven't already. You can find the installation instructions on the [pixi repository](https://github.com/prefix-dev/pixi#installation).
2. Clone the VLM-Human Loop repository and install the dependencies using pixi:
```bash
    git clone https://github.com/bach05/vh-loop.git
    cd vh-loop
    pixi install
```

# Library Design

The folder structure of the VLM-Human Loop project is organized as follows:
- `scripts/`: main container for Python source code, including modules for data processing, model training, and evaluation.
  - `core/`: contains core common utilities and functions used across the project.
  - `data/`: contains code for data representation, loading, and preprocessing.
  - `models/`: contains code for wrapping and integrating the vision-language model.
  - `training/`: contains code for training the VLMs.
- `tests/`: contains code for testing the different components of the project.

## Data Representation

We define a **canonical multimodal sample** that reflects the structure of a JSONL file. From the canonical format you can export the datasets in different formats (HF Datasets, COCO format, LabelStudio Format, etc… )

### Schema Visualization

<details>
  <summary>Click to visualize data schema diagram</summary>

```mermaid
%% data_schema_current_code_v2.mmd
%% Mermaid schema generated from the current uploaded codebase.
%% Focus: current implementation, not the broader target architecture.

classDiagram
direction TB

%% =========================
%% Dataset header / records
%% =========================

class DatasetInfoRecord {
  +Literal record_type = "dataset_info"
  +str schema_version
  +DatasetInfo info
}

class DataRecord {
  +Literal record_type = "sample"
  +SampleUnion sample
}

class DatasetInfo {
  +str dataset_id
  +Optional~str~ description
  +Optional~AnnotationInfo~ annotation_info
  +Optional~str~ domain
  +Optional~str~ split
  +Optional~str~ date_collected
  +Optional~str~ date_last_update
  +dict~str,LabelInfo~ label_info
  +MessageBuildInfo message_build_info
  +dict~str,Any~ metadata
  +label_name_to_id(label_name) int|str
  +label_id_to_name(label_id) str
}

class LabelInfo {
  +int|str label_id
  +str label_name
  +Optional~str~ description
  +list~str~ aliases
  +Optional~str~ parent_label
  +_validate_label_name() LabelInfo
}

class AnnotationInfo {
  +source_type source_type
  +quality quality
  +Optional~str~ notes
}

class MessageBuildInfo {
  +str prompt_template_version
  +answer_format answer_format
  +int normalization_factor
  +dict~str,Any~ metadata
}

DatasetInfoRecord *-- DatasetInfo : info
DatasetInfo *-- LabelInfo : label_info
DatasetInfo o-- AnnotationInfo : annotation_info
DatasetInfo *-- MessageBuildInfo : message_build_info

%% =========================
%% Polymorphic sample parsing
%% =========================

class SampleUnion {
  <<discriminated union>>
  +discriminator sample_type
  +SISimpleDataSample
}

DataRecord --> SampleUnion : sample
SampleUnion --> SISimpleDataSample : currently supports

%% =========================
%% Data samples
%% =========================

class DataSample {
  <<abstract>>
  +str sample_type
  +str sample_id
  +sample_to_message(dataset_info, prompting_schema, include_target, dataset_root, build_info) dict
}

class SISimpleDataSample {
  +Literal sample_type = "si_simple_data"
  +list~ImageAsset~ assets
  +_check_annotations() SISimpleDataSample
  +build_dataset_level_description(dataset_info) str
  +build_prompt(dataset_info) str
  +sample_to_message(dataset_info, prompting_schema, include_target, dataset_root, build_info) dict
}

DataSample <|-- SISimpleDataSample
SISimpleDataSample *-- ImageAsset : assets exactly 1
SISimpleDataSample ..> DatasetInfo : builds prompt from label_info
SISimpleDataSample ..> MessageBuildInfo : uses normalization_factor
SISimpleDataSample ..> InstanceAnnotation : serializes asset.annotations
SISimpleDataSample ..> PromptingSchema : conversational / prompt-completion

class PromptingSchema {
  <<alias>>
  SUPPORTED_PROMPTING_SCHEMAS
}

%% =========================
%% Assets
%% =========================

class Asset {
  <<abstract>>
  +str type
  +str uri
  +Optional~str~ caption
  +list~InstanceAnnotation~ annotations
  +dict~str,Any~ metadata
  +resolve_path(dataset_root) str
}

class ImageAsset {
  +Literal type = "image"
  +tuple~int,int~ size
  +Optional~str~ camera_id
  +width int
  +height int
}

class DepthImageAsset {
  +Literal type = "depth"
}

Asset <|-- ImageAsset
ImageAsset <|-- DepthImageAsset
Asset *-- InstanceAnnotation : annotations

%% =========================
%% Annotations
%% =========================

class InstanceAnnotation {
  +str instance_id
  +int label_id
  +str label_name
  +Optional~BoundingBox~ bbox
  +Optional~list~Point~~ points
  +Optional~RLEMask~ mask
  +Optional~str~ caption
  +dict~str,str~ attributes
}

InstanceAnnotation o-- BoundingBox : bbox
InstanceAnnotation o-- Point : points
InstanceAnnotation o-- RLEMask : mask
InstanceAnnotation ..> LabelInfo : label_id / label_name refer to label_info

%% =========================
%% Geometry
%% =========================

class Geometry {
  <<abstract>>
  +to_text(**kwargs) str
}

class Point {
  +int x
  +int y
  +Optional~bool~ is_positive
  +normalize(img_width, img_height, norm_factor) tuple
  +to_text(img_width, img_height, norm_factor) str
}

class BoundingBox {
  +Point tl
  +Point br
  +Literal format = "xyxy"
  +width float
  +height float
  +area() float
  +normalize(img_width, img_height, norm_factor) tuple
  +to_text(img_width, img_height, norm_factor) str
  +_valid() BoundingBox
}

class RLEMask {
  +list~int~|str counts
  +tuple~int,int~ size
  +to_binary_mask()
  +from_binary_mask(mask) RLEMask
  +to_text() str
}

Geometry <|-- Point
Geometry <|-- BoundingBox
Geometry <|-- RLEMask
BoundingBox *-- Point : tl
BoundingBox *-- Point : br

%% =========================
%% Current message flow
%% =========================

SISimpleDataSample --> TrainingRow : sample_to_message()

class TrainingRow {
  <<dict>>
  conversational: messages + images
  prompt-completion: prompt + completion + images
}

note for SISimpleDataSample "Single-image dataset-level prompt strategy.\nPrompt is built from DatasetInfo.label_info.\nTarget is compact tag bbox list:\n<class_name,x1,y1,x2,y2>;...\nCoordinates are normalized using MessageBuildInfo.normalization_factor."

note for DataRecord "DataRecord stores a discriminated SampleUnion.\nThe active discriminator is sample.sample_type."

note for LabelInfo "label_name is validated as compact token without spaces:\n^[A-Za-z0-9_.:-]+$"

```
</details>

A dataset is stored in a JSONL file, where each line is a JSON object record:

```json
TO DO
```

# Train, Test, and Compare (Hydra entrypoints)

This project uses Hydra entry scripts under `tests/`:

- `tests/train_sample.py` -> trains and saves checkpoints
- `tests/test_sample.py` -> runs inference and writes `predictions.jsonl`
- `tests/compare_sample.py` -> evaluates predictions vs GT and creates CSV/plots/visualizations

## 1) Environment variables and paths

Several configs use environment-variable-based paths.

- `DATA_PATH` default: `/data`
- `MODEL_PATH` default: `/models`

Set them before running from repository root:

```bash
export DATA_PATH=/absolute/path/to/your/data
export MODEL_PATH=/absolute/path/to/your/model_outputs
```

## 2) How config composition works

The entrypoint files compose config groups from `configs/`.

- Training entrypoint: `configs/train_entrypoint.yaml`
- Testing entrypoint: `configs/test_entrypoint.yaml`
- Comparison entrypoint: `configs/compare_entrypoint.yaml`

For train/test, the active defaults are selected in the `defaults:` list, for example:

- `model`: `configs/model/{gemma4,qwen3_5}.yaml`
- `dataset`: `configs/dataset/panizzolo.yaml`
- `transform`: `configs/transform/paniz_s1000.yaml`
- `peft`: `configs/peft/lora.yaml`
- `trainer`: `configs/trainer/{gemma4_sft_trainer,qwen_sft_trainer}.yaml`
- `quantization`: `configs/quantization/{4bit,8bit}` or `null`

Use Hydra overrides directly from CLI to change groups/fields at runtime.

## 3) Train

Basic run (uses `configs/train_entrypoint.yaml` defaults):

```bash
pixi run python tests/train_sample.py
```

Common overrides:

```bash
pixi run python tests/train_sample.py model=qwen3_5 trainer=qwen_sft_trainer
pixi run python tests/train_sample.py quantization=4bit
pixi run python tests/train_sample.py debug=false
pixi run python tests/train_sample.py trainer.num_train_epochs=5 trainer.learning_rate=1e-4
```

Training output directory is controlled by `hydra.run.dir` in `configs/train_entrypoint.yaml`:

`$MODEL_PATH/vhloop/training/${exp_name}`

Checkpoints are saved there as `checkpoint-*` directories.

## 4) Test (inference)

Basic run:

```bash
pixi run python tests/test_sample.py
```

Important behavior:

- `test_sample.py` reconstructs the checkpoint path automatically from the testing run dir by replacing `/testing/` with `/training/` and selecting the latest `checkpoint-*`.
- Predictions are written to `predictions.jsonl` in the test output directory.
- If `use_adapter=false`, output dir gets `_ORI_MODEL` suffix and inference runs base model only.

Common overrides:

```bash
pixi run python tests/test_sample.py use_adapter=true
pixi run python tests/test_sample.py use_adapter=false
pixi run python tests/test_sample.py model=qwen3_5 trainer=qwen_sft_trainer
pixi run python tests/test_sample.py debug=true +debug_max_samples=32
```

Testing output directory is controlled by `hydra.run.dir` in `configs/test_entrypoint.yaml`:

`$MODEL_PATH/vhloop/testing/${exp_name}`

## 5) Compare predictions against GT

Default run (uses explicit files from `configs/compare_entrypoint.yaml`):

```bash
pixi run python tests/compare_sample.py
```

Key config fields in `configs/compare_entrypoint.yaml`:

- `gt_jsonl`: ground-truth canonical JSONL
- `predictions`: explicit list of prediction files (`name` + `path`)
- `thresholds`: IoU thresholds used for precision/recall/F1
- `class_aware`: if `true`, only same-class boxes can match
- `visualization.enabled`: create per-sample visualization grids

Useful override examples:

```bash
pixi run python tests/compare_sample.py class_aware=false thresholds=[0.5,0.75,0.9]
pixi run python tests/compare_sample.py visualization.enabled=false
pixi run python tests/compare_sample.py visualization.sample_ids=[1,2,3] visualization.max_samples=3
pixi run python tests/compare_sample.py \
  predictions=[{"name":"run_a","path":"/abs/path/run_a/predictions.jsonl"},{"name":"run_b","path":"/abs/path/run_b/predictions.jsonl"}]
```

Comparison outputs (default):

- `metrics_by_threshold.csv`
- `summary.csv`
- `precision_by_threshold.png`, `recall_by_threshold.png`, `f1_by_threshold.png`, `mean_iou_tp_by_threshold.png`
- `miou_gt_penalized.png`
- `visualizations/sample_*.png` (if visualization enabled)

## 6) Minimal end-to-end workflow

```bash
# 1) Train
pixi run python tests/train_sample.py debug=false

# 2) Test with adapter
pixi run python tests/test_sample.py use_adapter=true

# 3) Test base model (optional baseline)
pixi run python tests/test_sample.py use_adapter=false

# 4) Compare
pixi run python tests/compare_sample.py
```

If your data manifests or output locations differ from defaults, update the relevant files in `configs/` or apply Hydra CLI overrides as shown above.
