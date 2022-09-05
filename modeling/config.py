from typing import List
from dataclasses import (dataclass, field)
from omegaconf import OmegaConf

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from common_config import (GCPConfig, NeptuneConfig)


@dataclass
class DataConfig:
    symbol: str = "usdjpy"
    first_year: int = 2020
    first_month: int = 11
    last_year: int = 2020
    last_month: int = 12


@dataclass
class FeatureConfig:
    timings: List[str] = field(default_factory=lambda: ["high", "low", "close"])
    freqs: List[str] = field(default_factory=lambda: ["1min", "5min", "15min", "1h", "4h"])
    lag_max: int = 5
    sma_timing: str = "close"
    sma_window_sizes: List[int] = field(default_factory=lambda: [10])
    sma_window_size_center: int = 10


# @dataclass
# class LGBMFeatureConfig(FeatureConfig):
#     lag_max: int = 5


@dataclass
class LabelConfig:
    # この値以上に上昇するならエントリーする
    thresh_entry: float = 0.05
    # この値を以下の下落であれば持ち続ける
    thresh_hold: float = 0.025


@dataclass
class LGBMModelConfig:
    objective: str = "binary"
    num_iterations: int = 10
    num_leaves: int = 31
    learning_rate: float = 0.1
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    pos_bagging_fraction: float = 1.0
    neg_bagging_fraction: float = 1.0
    bagging_freq: int = 0
    is_unbalance: bool = False
    scale_pos_weight: float = 1.0
    verbosity: int = 1


@dataclass
class CNNModelConfig:
    num_epochs: int = 1
    learning_rate: float = 1.0e-3
    pos_weight: float = 1.0
    batch_size: int = 256
    window_size: int = 32
    out_channels_list: List[int] = field(default_factory=lambda: [5, 10, 5])
    kernel_size_list: List[int] = field(default_factory=lambda: [5, 5, 5])
    base_out_dim: int = 32
    hidden_dim_list: List[int] = field(default_factory=lambda: [128])


@dataclass
class TrainConfig:
    random_seed: int = 123
    valid_ratio: float = 0.1
    save_model: bool = True

    gcp: GCPConfig = GCPConfig()
    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()
    label: LabelConfig = LabelConfig()


@dataclass
class LGBMTrainConfig(TrainConfig):
    feature: FeatureConfig = FeatureConfig(lag_max=5)
    model: LGBMModelConfig = LGBMModelConfig()


@dataclass
class CNNTrainConfig(TrainConfig):
    feature: FeatureConfig = FeatureConfig(lag_max=32)
    model: CNNModelConfig = CNNModelConfig()


@dataclass
class EvalConfig:
    start_hour: int = 2
    end_hour: int = 22
    thresh_loss_cut: float = 0.05
    simulate_timing: str = "open"
    spread: float = 0.02
    percentile_entry_list: List[int] = field(default_factory=lambda: [75, 90, 95])
    percentile_exit_list: List[int] = field(default_factory=lambda: [75, 90, 95])

    gcp: GCPConfig = GCPConfig()
    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()


def validate_train_config(config: OmegaConf):
    assert config.feature.sma_window_size_center in config.feature.sma_window_sizes
    window_size = config.model.get("window_size", None)
    if window_size is not None:
        assert window_size == config.feature.lag_max
