from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class NeptuneConfig:
    project: str = "thashimoto/auto-trader"
    mode: str = "async"

    def __post_init__(self) -> None:
        if self.mode not in ["async", "debug"]:
            raise ValueError(f"Unknown mode {self.mode}")


@dataclass
class DataConfig:
    symbol: str = MISSING
    cleansed_data_dir: str = "./cleansed"
    yyyymm_begin: int = MISSING
    yyyymm_end: int = MISSING

    def __post_init__(self) -> None:
        if self.symbol != MISSING and self.symbol not in ["usdjpy", "eurusd"]:
            raise ValueError(f"Unknown symbol {self.symbol}")


@dataclass
class FeatureConfig:
    timeframes: list[str] = field(default_factory=lambda: ["1min", "5min", "1h"])
    base_timing: str = "close"
    sma_window_sizes: list[int] = field(default_factory=lambda: [5, 8, 13])
    sma_window_size_center: int = 5
    sigma_window_sizes: list[int] = field(default_factory=lambda: [9])
    sma_frac_unit: int = 100
    hist_len: int = 10
    start_hour: int = 2
    end_hour: int = 22

    def __post_init__(self) -> None:
        if "1min" not in self.timeframes:
            raise ValueError(f"timeframes {self.timeframes} must include '1min'")

        if self.sma_window_size_center not in self.sma_window_sizes:
            raise ValueError(
                f"sma_window_sizes {self.sma_window_sizes} "
                f"must include sma_window_size_center {self.sma_window_size_center}"
            )


@dataclass
class GainConfig:
    alpha: float = 0.1
    thresh_losscut: float = 5.0


@dataclass
class NetConfig:
    numerical_emb_dim: int = 16
    periodic_activation_num_coefs: int = 8
    periodic_activation_sigma: float = 1.0
    categorical_emb_dim: int = 16
    emb_output_dim: int = 64

    base_net_type: str = "attention"

    base_attention_num_layers: int = 3
    base_attention_num_heads: int = 1
    base_attention_feedforward_dim: int = 128
    base_attention_dropout: float = 0.1

    base_conv_out_channels: list[int] = field(default_factory=lambda: [20, 40, 20])
    base_conv_kernel_sizes: list[int] = field(default_factory=lambda: [5, 5, 5])
    base_conv_batchnorm: bool = True
    base_conv_dropout: float = 0.0

    base_fc_hidden_dims: list[int] = field(default_factory=lambda: [128])
    base_fc_batchnorm: bool = False
    base_fc_dropout: float = 0.0
    base_fc_output_dim: int = 128

    head_hidden_dims: list[int] = field(default_factory=lambda: [64])
    head_batchnorm: bool = False
    head_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.numerical_emb_dim % 2 != 0:
            raise ValueError(
                f"numerical_emb_dim must be a even number: {self.numerical_emb_dim}"
            )

        if self.base_net_type not in ["attention", "conv"]:
            raise ValueError(f"Unknown base_type {self.base_net_type}")

        if self.emb_output_dim % self.base_attention_num_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_output_dim}) must be divisible by "
                f"num_heads ({self.base_attention_num_heads})"
            )

        if len(self.base_conv_out_channels) != len(self.base_conv_kernel_sizes):
            raise ValueError(
                f"Number of conv channels and kernel sizes does not match: "
                f"{self.base_conv_out_channels} != {self.base_conv_kernel_sizes}"
            )


@dataclass
class LossConfig:
    entropy_coef: float = 1.0
    spread: float = 2.0
    entry_pos_coef: float = 1.0
    exit_pos_coef: float = 1.0


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    output_dir: str = "./output"
    max_epochs: int = 20
    early_stopping_patience: int = 3
    batch_size: int = 1000
    valid_block_size: int = 60 * 4
    valid_ratio: float = 0.1
    random_seed: int = 123

    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    net: NetConfig = field(default_factory=NetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class SimulationConfig:
    timing: str = "open"
    spread: float = 2.0
    start_hour: int = 2
    end_hour: int = 22
    thresh_losscut: float = 5.0


@dataclass
class EvalConfig:
    output_dir: str = "./output"
    train_run_id: str = ""
    params_file: str = ""
    percentile_entry_list: list[float] = field(default_factory=lambda: [90, 95])
    percentile_exit_list: list[float] = field(default_factory=lambda: [90, 95])

    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self) -> None:
        if self.train_run_id == "" and self.params_file == "":
            raise ValueError("Either train_run_id or params_file must be specified")
