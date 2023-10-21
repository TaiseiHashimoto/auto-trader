from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class NeptuneConfig:
    project: str = "thashimoto/auto-trader"
    project_key: str = "AUT"
    mode: str = "async"


@dataclass
class DataConfig:
    symbol: str = MISSING
    cleansed_data_dir: str = "./cleansed"
    yyyymm_begin: int = MISSING
    yyyymm_end: int = MISSING


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
        assert "1min" in self.timeframes
        assert self.sma_window_size_center in self.sma_window_sizes


@dataclass
class LiftConfig:
    target_alpha: float = 0.1


@dataclass
class NetConfig:
    numerical_emb_dim: int = 16
    categorical_emb_dim: int = 16
    base_cnn_out_channels: list[int] = field(default_factory=lambda: [20, 40, 20])
    base_cnn_kernel_sizes: list[int] = field(default_factory=lambda: [5, 5, 5])
    base_cnn_batchnorm: bool = True
    base_cnn_dropout: float = 0.0
    base_fc_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    base_fc_batchnorm: bool = False
    base_fc_dropout: float = 0.0
    base_fc_output_dim: int = 128
    head_hidden_dims: list[int] = field(default_factory=lambda: [64])
    head_batchnorm: bool = False
    head_dropout: float = 0.0


@dataclass
class LossConfig:
    entropy_coef: float = 0.01
    spread: float = 2.0


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
    valid_ratio: float = 0.1
    random_seed: int = 123

    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()
    feature: FeatureConfig = FeatureConfig()
    lift: LiftConfig = LiftConfig()
    net: NetConfig = NetConfig()
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()


@dataclass
class SimulationConfig:
    timing: str = "open"
    spread: float = 2.0
    start_hour: int = 2
    end_hour: int = 22
    thresh_loss_cut: float = 0.05


@dataclass
class EvalConfig:
    output_dir: str = "./output"
    train_run_id: str = ""
    params_file: str = ""
    percentile_entry_list: list[float] = field(default_factory=lambda: [75, 90, 95])
    percentile_exit_list: list[float] = field(default_factory=lambda: [75, 90, 95])

    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()
    simulation: SimulationConfig = SimulationConfig()
