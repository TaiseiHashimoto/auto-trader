from typing import List
from dataclasses import (dataclass, field)

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
    sma_window_size: int = 10


@dataclass
class LabelConfig:
    # この値以上に上昇するならエントリーする
    thresh_entry: float = 0.05
    # この値を以下の下落であれば持ち続ける
    thresh_hold: float = 0.025
    # この値以上に上昇しないなら決済する
    thresh_exit: float = 0.0


@dataclass
class ModelConfig:
    objective: str = "binary"
    num_leaves: int = 31
    verbosity: int = 1


@dataclass
class TrainConfig:
    valid_ratio: float = 0.1
    num_iterations: int = 10
    save_model: bool = True


@dataclass
class LGBMConfig:
    on_colab: bool = False
    random_seed: int = 123

    gcp: GCPConfig = GCPConfig()
    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()
    feature: FeatureConfig = FeatureConfig()
    label: LabelConfig = LabelConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
