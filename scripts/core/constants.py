#Collection of constant values
from typing import Literal

#NORM_SIZE = 1000  # canonical square side length for coordinates normalization
#SCHEMA_VERSION = "vhloop.dataset.v1"
SUPPORTED_PROMPT_STRATEGIES = ["simple_prompt"]
SUPPORTED_TARGET_STRATEGIES = ["bbox_json_norm1000"]
SUPPORTED_PEFT_STRATEGIES = ["LORA"]
SUPPORTED_PROMPTING_SCHEMAS = Literal["conversational", "prompt-completion"]
SUPPORTED_TRAIN_LIB = ["transformers", "unsloth"]
SFT_CONVERTER_VERSION = "sft-v2.1"