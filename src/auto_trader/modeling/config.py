from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class NeptuneConfig:
    project: str = "thashimoto/auto-trader"
    mode: str = "async"

    def __post_init__(self) -> None:
        if self.mode not in ["async", "debug"]:
            raise ValueError(f"Unknown mode {self.mode}")


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
    timeframes: list[str] = field(default_factory=lambda: ["1min"])
    base_timing: str = "close"
    moving_window_sizes: list[int] = field(default_factory=lambda: [5, 8, 13])
    moving_window_size_center: int = 5
    use_sma_frac: bool = True
    sma_frac_unit: int = 100
    use_hour: bool = True
    use_dow: bool = True
    hist_len: int = 10

    def __post_init__(self) -> None:
        if "1min" not in self.timeframes:
            raise ValueError(f"timeframes {self.timeframes} must include '1min'")

        if self.moving_window_size_center not in self.moving_window_sizes:
            raise ValueError(
                f"sma_window_sizes {self.moving_window_sizes} must include "
                f"moving_window_size_center {self.moving_window_size_center}"
            )


@dataclass
class LiftConfig:
    alpha: float = 0.1


@dataclass
class NetConfig:
    numerical_emb_dim: int = 16
    periodic_activation_num_coefs: int = 8
    periodic_activation_sigma: float = 1.0
    categorical_emb_dim: int = 16
    emb_kernel_size: int = 5

    num_blocks: int = 3
    block_num_heads: int = 4
    block_qkv_kernel_size: int = 5
    block_ff_kernel_size: int = 5
    block_channels: int = 20
    block_ff_channels: int = 40
    block_dropout: float = 0.0

    head_hidden_dims: list[int] = field(default_factory=lambda: [64])
    head_batchnorm: bool = False
    head_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.numerical_emb_dim % 2 != 0:
            raise ValueError(
                f"numerical_emb_dim must be a even number: {self.numerical_emb_dim}"
            )

        if self.emb_kernel_size % 2 != 1:
            raise ValueError(f"kernel must be odd numbers: {self.emb_kernel_size}")

        if self.block_channels % self.block_num_heads != 0:
            raise ValueError(
                "block_channels must be divisible by block_num_heads: "
                f"{self.block_channels} % {self.block_num_heads} != 0"
            )

        if self.block_qkv_kernel_size % 2 != 1:
            raise ValueError(
                f"kernel must be odd numbers: {self.block_qkv_kernel_size}"
            )

        if self.block_ff_kernel_size % 2 != 1:
            raise ValueError(f"kernel must be odd numbers: {self.block_ff_kernel_size}")


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
    symbols: list[str] = field(default_factory=lambda: ["usdjpy"])
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
    lift: LiftConfig = field(default_factory=LiftConfig)
    net: NetConfig = field(default_factory=NetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def __post_init__(self) -> None:
        for symbol in self.symbols:
            if symbol not in YYYYMM_BEGIN_MAP:
                raise ValueError(f"Unknown symbol {symbol}")


@dataclass
class StrategyConfig:
    percentile_entry: float = 95.0
    max_entry_time: int = 10


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
            raise ValueError(f"Unknown symbol {self.symbol}")

        if self.train_run_id == "" and self.params_file == "":
            raise ValueError("Either train_run_id or params_file must be specified")
