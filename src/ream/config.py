from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ModelConfig:
    text_model: str = "bert-base-uncased"
    max_length: int = 128
    representation_dim: int = 128
    hidden_dim: int = 64
    dropout: float = 0.1
    max_neighbors: int = 15
    max_neighbor_name_chars: int = 64
    char_embedding_dim: int = 32
    char_cnn_channels: int = 32
    char_kernel_sizes: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    category_embedding_dim: int = 32
    rbf_bins: int = 16
    rbf_sigma: float = 1.0 / 15.0
    beta: float = 0.2
    temperature: float = 1.0
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 25
    warmup_epochs: int = 5
    patience: int = 5
    text_lr: float = 2e-5
    other_lr: float = 1e-4
    weight_decay: float = 0.01
    lambda_branch: float = 0.3
    lambda_reliability: float = 0.2
    use_bf16: bool = True
    num_workers: int = 4
    threshold_grid_points: int = 181


@dataclass
class DegradationConfig:
    clean_fraction: float = 0.50
    single_fraction_of_degraded: float = 0.70
    text_mask_min: float = 0.10
    text_mask_max: float = 0.40
    coordinate_shift_min_m: float = 20.0
    coordinate_shift_max_m: float = 200.0
    neighborhood_delete_min: float = 0.20
    neighborhood_delete_max: float = 0.60


@dataclass
class MetricsConfig:
    ece_bins: int = 15
    reliability_bins: int = 15


@dataclass
class Config:
    seed: int = 13
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        seed=raw.get("seed", 13),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        degradation=_merge_dataclass(DegradationConfig, raw.get("degradation")),
        metrics=_merge_dataclass(MetricsConfig, raw.get("metrics")),
    )
