"""Dataset protocol and tensor batch containers for benchmarks."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One prompt and reference completion selected for measurement."""

    prompt: str
    completion: str


@dataclass(frozen=True, slots=True)
class ModelInputBatch:
    """Tokenized, left-padded inputs placed on the benchmark device."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def input_tokens(self) -> int:
        return int(self.attention_mask.sum().item())


class BenchmarkDataset(Protocol):
    """Interface implemented by benchmark prompt sources."""

    def iter_batches(
        self,
        tokenizer: Any,
        *,
        batch_size: int,
        num_prompts: int,
        output_length: int,
        device: torch.device,
    ) -> Iterator[ModelInputBatch]: ...
