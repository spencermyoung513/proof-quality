from __future__ import annotations

from pathlib import Path
from typing import Literal
from re import sub
from pydantic import BaseModel, field_validator


def to_kebab_case(string: str) -> str:
    return "-".join(
        sub(
            "([A-Z][a-z]+)",
            r" \1",
            sub("([A-Z]+)", r" \1", string.replace("-", " ").replace("_", " ")),
        ).split()
    ).lower()


class RunConfig(BaseModel):
    name: str
    ckpt_dir: Path = Path("ckpt")
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    precision: Literal["bf16-mixed", "32-true"] = "32-true"
    seed: int | None = None
    smoke_test: bool = False

    @field_validator("name")
    def cast_to_kebab_case(cls, v: str) -> str:
        return to_kebab_case(v)


class OptimizerConfig(BaseModel):
    batch_size: int = 4
    accumulate_grad_batches: int = 1
    num_epochs: int = 10
    weight_decay: float = 0.01
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.05


class ModelConfig(BaseModel):
    base_lm: str = "google/gemma-3-270m-it"
    lora_rank: int = 8
    lora_scaling_factor: int = 8
    lora_dropout: float = 0.1
    max_length: int = 1024


class DataConfig(BaseModel):
    root: Path = Path("data/proofjudge")


class TrainingConfig(BaseModel):
    run: RunConfig
    optimizer: OptimizerConfig
    model: ModelConfig
    data: DataConfig = DataConfig()
