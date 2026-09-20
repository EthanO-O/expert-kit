"""Shared Pydantic configuration behavior for generator schemas."""

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Strict, immutable base model for deployment configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )
