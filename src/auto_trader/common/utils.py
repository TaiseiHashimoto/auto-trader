import os
import random
import sys
from datetime import date
from typing import Optional, TypeVar, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, SCMode

T = TypeVar("T")


def get_config(base_class: type[T], list_config: Optional[list[str]] = None) -> T:
    if list_config is None:
        list_config = sys.argv[1:]

    dict_config = cast(DictConfig, OmegaConf.structured(base_class))
    dict_config.merge_with_dotlist(list_config)
    return cast(
        T,
        OmegaConf.to_container(dict_config, structured_config_mode=SCMode.INSTANTIATE),
    )


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
    if symbol in ("usdjpy"):
        return 0.01
    elif symbol in (
        "eurusd",
        "gbpusd",
        "usdcad",
        "usdchf",
        "audusd",
        "nzdusd",
    ):
        return 0.0001
    else:
        raise ValueError(f"Unknown symbol {symbol}")


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def validate_neptune_settings(mode: str) -> None:
    if mode not in ("debug") and "NEPTUNE_API_TOKEN" not in os.environ:
        raise RuntimeError("NEPTUNE_API_TOKEN has to be set.")
