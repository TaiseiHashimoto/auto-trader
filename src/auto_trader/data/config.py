from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class CollectConfig:
    symbol: str = "usdjpy"
    raw_data_dir: str = "./raw"
    yyyymm_begin: int = MISSING
    yyyymm_end: int = MISSING
    recreate_latest: bool = True


@dataclass
class CleanseConfig:
    symbol: str = "usdjpy"
    raw_data_dir: str = "./raw"
    cleansed_data_dir: str = "./cleansed"
    yyyymm_begin: int = MISSING
    yyyymm_end: int = MISSING
    recreate_latest: bool = True
    validate: bool = True
