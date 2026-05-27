#Collection of constant values
from typing import Literal

#NORM_SIZE = 1000  # canonical square side length for coordinates normalization
#SCHEMA_VERSION = "vhloop.dataset.v1"
SUPPORTED_PROMPT_STRATEGIES = ["simple_prompt"]
SUPPORTED_TARGET_STRATEGIES = ["bbox_json_norm1000"]
SUPPORTED_PEFT_STRATEGIES = ["LORA"]
SUPPORTED_PROMPTING_SCHEMAS = Literal["conversational", "prompt-completion"]
SUPPORTED_TRAIN_LIB = ["HFTransformers", "unsloth"]
SFT_CONVERTER_VERSION = "sft-v2.1"

class RuntimeEnvironmentSettings:
    def __init__(self):
        self.IN_USE_TRAIN_LIB = "unsloth"

    def set_train_framework(self, train_framework: str):
        if train_framework not in SUPPORTED_TRAIN_LIB:
            raise ValueError(f"Unsupported training library '{train_framework}'. Supported libraries: {SUPPORTED_TRAIN_LIB}")
        self.IN_USE_TRAIN_LIB = train_framework


running_env = RuntimeEnvironmentSettings()