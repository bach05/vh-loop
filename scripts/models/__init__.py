# scripts/models/__init__.py

from scripts.core.registry import register_model_adapter
from scripts.models.gemma4_adapter import Gemma4Adapter
from scripts.models.qwen3_5_adapter import Qwen3_5Adapter


register_model_adapter("qwen3_5")(Qwen3_5Adapter)
register_model_adapter("gemma4")(Gemma4Adapter)