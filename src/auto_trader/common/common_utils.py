import random
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"


class Order:
    def __init__(
        self,
        position_type: PositionType,
        entry_timestamp: pd.Timestamp,
        entry_rate: float,
    ):
        self.position_type = position_type
        self.entry_timestamp = entry_timestamp
        self.entry_rate = entry_rate
        self.exit_timestamp = None
        self.exit_rate = None

    def exit(
        self,
        exit_timestamp: pd.Timestamp,
        exit_rate: float,
    ):
        self.exit_timestamp = exit_timestamp
        self.exit_rate = exit_rate

    @property
    def gain(self):
        rate_diff = self.exit_rate - self.entry_rate
        if self.position_type == PositionType.LONG:
            return rate_diff
        elif self.position_type == PositionType.SHORT:
            return -rate_diff

    def __repr__(self) -> str:
        gain = None
        if self.exit_rate is not None:
            gain = self.gain
        return (
            f"{self.position_type} ({self.entry_timestamp} ~ {self.exit_timestamp}) "
            f"{self.entry_rate} -> {self.exit_rate} ({gain})"
        )


class OrderSimulator:
    def __init__(
        self,
        start_hour: int,
        end_hour: int,
        thresh_loss_cut: float,
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.thresh_loss_cut = thresh_loss_cut
        self.order_history = []
        self.open_position = None

    def step(
        self,
        timestamp: pd.Timestamp,
        rate: float,
        long_entry: bool,
        short_entry: bool,
        long_exit: bool,
        short_exit: bool,
    ):
        is_open = self.start_hour <= timestamp.hour < self.end_hour and (
            timestamp.month,
            timestamp.day,
        ) != (12, 25)

        # 決済する条件: ポジションをもっている and (取引時間外 or モデルが決済を選択 or 損切り)
        if (
            self._has_position(PositionType.LONG)
            and (
                not is_open
                or long_exit
                or self.open_position.entry_rate - rate >= self.thresh_loss_cut
            )
        ) or (
            self._has_position(PositionType.SHORT)
            and (
                not is_open
                or short_exit
                or rate - self.open_position.entry_rate >= self.thresh_loss_cut
            )
        ):
            self.open_position.exit(timestamp, rate)
            self.order_history.append(self.open_position)
            self.open_position = None

        if is_open:
            if not self._has_position() and long_entry:
                self.open_position = Order(PositionType.LONG, timestamp, rate)
            if not self._has_position() and short_entry:
                self.open_position = Order(PositionType.SHORT, timestamp, rate)

    def _has_position(self, position_type: Optional[PositionType] = None) -> bool:
        if position_type is None:
            return self.open_position is not None
        else:
            return (
                self.open_position is not None
                and self.open_position.position_type == position_type
            )

    def export_results(self) -> pd.DataFrame:
        results = {
            "position_type": [],
            "entry_timestamp": [],
            "exit_timestamp": [],
            "entry_rate": [],
            "exit_rate": [],
            "gain": [],
        }
        for order in self.order_history:
            results["position_type"].append(order.position_type.value)
            results["entry_timestamp"].append(order.entry_timestamp)
            results["exit_timestamp"].append(order.exit_timestamp)
            results["entry_rate"].append(order.entry_rate)
            results["exit_rate"].append(order.exit_rate)
            results["gain"].append(order.gain)

        return pd.DataFrame(results)


def parse_yyyymm(yyyymm: int) -> date:
    """
    yyyymm 形式をパース
    """
    return date(yyyymm // 100, yyyymm % 100, 1)


def calc_yyyymm(yyyymm_base: int, month_delta: int) -> int:
    """
    ある年月からnヶ月後/前の年月を求める
    """
    parsed = parse_yyyymm(yyyymm_base)
    year = parsed.year + (parsed.month + month_delta - 1) // 12
    month = (parsed.month + month_delta - 1) % 12 + 1
    return year * 100 + month


def drop_keys(d: Dict, keys_to_drop: List[str]) -> Dict:
    return {k: v for k, v in d.items() if k not in keys_to_drop}


def get_pip_scale(symbol: str) -> float:
    return 0.01 if symbol == "usdjpy" else 0.0001


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_neptune_model_id(project_key: str, model_type: str) -> str:
    return f"{project_key}-{model_type.upper()}"
