from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class NeptuneConfig:
    project: str = "thashimoto/auto-trader"
    mode: str = "async"

    def __post_init__(self) -> None:
        if self.mode not in ["async", "debug"]:
            raise ValueError(f"Unknown mode `{self.mode}`")


YYYYMM_BEGIN_MAP = {
    "usdjpy": 201204,  # 201203 にスプレッドマイナス
    "eurusd": 201012,  # 201011 まではスプレッドが同じ値を取ることが多い
    "gbpusd": 201011,  # 201010 までは桁数が少ない
    "usdcad": 201011,  # 201010 までは桁数が少ない
    "usdchf": 201011,  # 201010 までは桁数が少ない
    "audusd": 201109,  # 201108 にスプレッドマイナス
    "nzdusd": 201011,  # 201010 までは桁数が少ない
}


@dataclass
class FeatureConfig:
    base_timing: str = "close"
    window_sizes: list[int] = field(default_factory=lambda: [5, 10, 15])
    use_sma_frac: bool = True
    sma_frac_unit: int = 100
    use_hour: bool = True
    use_dow: bool = True
    hist_len: int = 64
    hour_begin: int = 2
    hour_end: int = 22


@dataclass
class LabelConfig:
    future_begin: int = 10
    future_end: int = 20
    bin_boundary: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.future_begin < self.future_end:
            raise ValueError(
                f'The condition "0 < future_begin `{self.future_begin}` < '
                f'future_end `{self.future_end}`" is not met.'
            )


@dataclass
class NetConfig:
    numerical_emb_dim: int = 16
    periodic_activation_num_coefs: int = 8
    periodic_activation_sigma: float = 1.0
    categorical_emb_dim: int = 16
    out_channels: list[int] = field(default_factory=lambda: [64, 64])
    kernel_sizes: list[int] = field(default_factory=lambda: [8, 8])
    strides: list[int] = field(default_factory=lambda: [8, 8])
    batchnorm: bool = True
    dropout: float = 0.2
    head_hidden_dims: list[int] = field(default_factory=lambda: [64])
    head_batchnorm: bool = False
    head_dropout: float = 0.0


@dataclass
class LossConfig:
    boundary: float = 2.0
    temperature: float = 0.1


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cosine_decay_steps: int = 0
    cosine_decay_min: float = 0.01


@dataclass
class TrainConfig:
    cleansed_data_dir: str = "./cleansed"
    symbol: str = "usdjpy"
    yyyymm_begin: Optional[int] = None
    yyyymm_end: int = MISSING
    output_dir: str = "./output"
    max_epochs: int = 20
    early_stopping_patience: int = 3
    batch_size: int = 1000
    valid_block_size: int = 60 * 4
    valid_ratio: float = 0.1
    random_seed: int = 123

    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    net: NetConfig = field(default_factory=NetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def __post_init__(self) -> None:
        if self.symbol not in YYYYMM_BEGIN_MAP:
            raise ValueError(f"Unknown symbol {self.symbol}")


@dataclass
class StrategyConfig:
    percentile_entry: float = 95.0
    entry_time_max: int = 60


@dataclass
class SimulationConfig:
    timing: str = "open"
    spread: float = 2.0
    start_hour: int = 2
    end_hour: int = 22
    thresh_losscut: float = 5.0


@dataclass
class EvalConfig:
    cleansed_data_dir: str = "./cleansed"
    symbol: str = "usdjpy"
    yyyymm_begin: int = MISSING
    yyyymm_end: int = MISSING
    output_dir: str = "./output"
    train_run_id: str = ""
    params_file: str = ""
    batch_size: int = 1000

    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self) -> None:
        if self.symbol not in YYYYMM_BEGIN_MAP:
            raise ValueError(f"Unknown symbol `{self.symbol}`")

        if self.train_run_id == "" and self.params_file == "":
            raise ValueError("Either train_run_id or params_file must be specified")
