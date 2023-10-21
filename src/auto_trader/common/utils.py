import os
import random
from datetime import date

import numpy as np
import torch


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


def get_pip_scale(symbol: str) -> float:
    return 0.01 if symbol == "usdjpy" else 0.0001


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def validate_neptune_settings(mode: str) -> None:
    if mode not in ("debug") and "NEPTUNE_API_TOKEN" not in os.environ:
        raise RuntimeError("NEPTUNE_API_TOKEN has to be set.")
