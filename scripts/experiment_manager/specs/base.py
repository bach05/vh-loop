from pydantic import BaseModel, ConfigDict

class BaseSpec(BaseModel):
    """Base configuration spec."""
    model_config = ConfigDict(extra='allow')
