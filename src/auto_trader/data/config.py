import datetime
from dataclasses import dataclass, field
from typing import List

from auto_trader.common.common_config import GCPConfig


@dataclass
class RawConfig:
    symbols: List[str] = field(default_factory=lambda: ["usdjpy", "eurusd"])
    first_datetime: str = "2010-01-01T00:00:00+00:00"
    last_datetime: str = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


@dataclass
class PreprocessConfig:
    symbols: List[str] = field(default_factory=lambda: ["usdjpy", "eurusd"])
    first_year: int = 2020
    first_month: int = 11
    last_year: int = datetime.datetime.now().year
    last_month: int = datetime.datetime.now().month
    gcp: GCPConfig = GCPConfig()
