"""Experiment, model, dataset, and serving configuration models."""

from __future__ import annotations
import yaml
from typing import Self
from typing import Any, Literal
from pathlib import Path
from pydantic import (
    AnyHttpUrl,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    model_validator,
)

from enum import StrEnum
from expertkit_transport.ascend_graph import (
    AscendGraphOptions,
    AscendGraphQuantization,
)

from .config import ConfigModel


class DatasetType(StrEnum):
    """Dataset source supported by the benchmark generator."""

    RANDOM = "random"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class Dtype(StrEnum):
    """Activation or weight dtype label emitted in generated configs."""

    FP16 = "fp16"
    BF16 = "bf16"
    # FP4 and FP8 are accepted as configuration labels for generated benchmarks.
    FP8 = "fp8"
    FP4 = "fp4"


class TracingConfig(ConfigModel):
    """Optional tracing settings shared by the generated Frontend and Workers."""

    enabled: bool = False
    endpoint: AnyHttpUrl | None = None
    sample_ratio: float = Field(default=0.01, gt=0, le=1)

    @model_validator(mode="after")
    def validate_endpoint(self) -> TracingConfig:
        """Require a plaintext collector endpoint when tracing is enabled."""
        if self.enabled and self.endpoint is None:
            raise ValueError("tracing.endpoint is required when tracing is enabled")
        if (
            self.enabled
            and self.endpoint is not None
            and self.endpoint.scheme != "http"
        ):
            raise ValueError("tracing requires a plaintext HTTP endpoint")
        return self


class ExperimentConfig(ConfigModel):
    """Complete model, dataset, serving, and run configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    serve: ServeConfig
    run: RunConfig
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    ascend_decode_graph: AscendGraphOptions | None = None

    @classmethod
    def from_yaml(cls, file: Path) -> Self:
        data = yaml.safe_load(file.read_text())
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_dataset_run_options(self) -> Self:
        if self.dataset.type is DatasetType.RANDOM:
            if self.run.input_len is None:
                raise ValueError("random dataset requires run.input_len")
        elif self.run.input_len is not None:
            raise ValueError(
                f"{self.dataset.type} dataset must not define run.input_len"
            )

        return self


class ModelConfig(ConfigModel):
    """Model artifact and frontend execution settings."""

    name: str
    path_ref: str
    weight_version: str
    num_layers: int | None = Field(default=None, gt=0)
    experts_per_layer: int | None = Field(default=None, gt=0)
    hidden_dim: int | None = Field(default=None, gt=0)
    intermediate_dim: int | None = Field(default=None, gt=0)
    topk: int | None = Field(default=None, gt=0)
    weight_dtype: Dtype
    activation_dtype: Dtype
    expert_compute: Literal["swiglu", "deepseek_v4"] = "swiglu"
    swiglu_limit: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    quantization: AscendGraphQuantization | None = None

    @model_validator(mode="after")
    def validate_computation(self) -> Self:
        """Keep clipping specific to the declared V4 expert expression."""
        if (
            "expert_compute" in self.model_fields_set
            and self.expert_compute == "swiglu"
            and self.swiglu_limit != 0
        ):
            raise ValueError("swiglu_limit requires deepseek_v4 expert computation")
        return self

    @computed_field
    @property
    def worker_model(self) -> dict[str, Any]:
        """Render required execution settings and only explicitly supplied model facts."""
        values = {
            "name": self.name,
            "weight_version": self.weight_version,
            "activation_dtype": str(self.activation_dtype),
            "weight_dtype": str(self.weight_dtype),
        }
        aliases = {"intermediate_dim": "expert_intermediate_dim", "topk": "top_k"}
        for field in (
            "num_layers",
            "experts_per_layer",
            "hidden_dim",
            "intermediate_dim",
            "topk",
            "expert_compute",
            "swiglu_limit",
            "quantization",
        ):
            value = getattr(self, field)
            if field in self.model_fields_set and value is not None:
                values[aliases.get(field, field)] = (
                    value.model_dump(mode="json") if field == "quantization" else value
                )
        return values


class DatasetConfig(ConfigModel):
    """Dataset artifact and sampling settings."""

    type: DatasetType
    name: str
    path_ref: str | None = None
    mounted_path: Path | None = None
    file: str | None = None

    @model_validator(mode="after")
    def validate_path_fields(self) -> Self:
        path_fields = (
            self.path_ref,
            self.mounted_path,
            self.file,
        )

        if self.type is DatasetType.RANDOM:
            if any(value is not None for value in path_fields):
                raise ValueError(
                    "random dataset must not define path_ref, mounted_path, or file"
                )
        elif any(value is None for value in path_fields):
            raise ValueError(
                f"{self.type} dataset requires path_ref, mounted_path, and file"
            )

        return self


class ServeConfig(ConfigModel):
    """Serving frontend options for generated deployment files."""

    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool = False
    max_num_seqs: StrictInt | None = None
    max_num_batched_tokens: StrictInt | None = None
    max_num_scheduled_tokens: StrictInt | None = None
    block_size: StrictInt | None = None
    num_gpu_blocks_override: StrictInt | None = None
    kv_cache_memory_bytes: StrictInt | None = None
    enable_chunked_prefill: StrictBool | None = None
    quantization: StrictStr | None = None
    load_format: StrictStr | None = None
    safetensors_load_strategy: StrictStr | None = None
    distributed_executor_backend: StrictStr | None = None
    trust_remote_code: StrictBool | None = None
    compilation_mode: StrictStr | StrictInt | None = None
    seed: StrictInt | None = None

    def with_legacy_model_defaults(
        self, model: ModelConfig, *, graph_enabled: bool
    ) -> Self:
        """Preserve legacy loading options unless a native setting is explicit.

        Explicit null suppresses a compatibility default and leaves the native
        option absent. Model declarations remain independent Worker assertions.
        """
        defaults: dict[str, Any] = {}
        if model.quantization is not None:
            defaults["quantization"] = "ascend"
        if model.expert_compute == "deepseek_v4":
            defaults.update(
                load_format="safetensors",
                safetensors_load_strategy="lazy",
                distributed_executor_backend="mp",
            )
            if graph_enabled:
                defaults["compilation_mode"] = "VLLM_COMPILE"
        if graph_enabled:
            defaults["trust_remote_code"] = model.expert_compute != "deepseek_v4"
        return self.model_copy(
            update={
                key: value
                for key, value in defaults.items()
                if key not in self.model_fields_set
            }
        )


class RunConfig(ConfigModel):
    """Benchmark repetition, concurrency, and output settings."""

    num_prompts: int
    max_concurrency: int
    input_len: int | None = None
    output_len: int
    num_warmups: int
    ignore_eos: bool
    temperature: float
    save_result: bool
    save_detailed: bool | None = None
    percentile_metrics: str | None = None
    metric_percentiles: str | None = None
    ready_check_timeout_sec: int | None = Field(default=None, ge=0)
    seed: StrictInt | None = None
    request_rate: float | None = Field(default=None, gt=0)
    disable_shuffle: StrictBool | None = None
    no_oversample: StrictBool | None = None
    extra_body: dict[str, JsonValue] | None = None
