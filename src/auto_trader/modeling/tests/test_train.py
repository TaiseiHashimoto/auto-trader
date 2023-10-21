from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from auto_trader.modeling import train
from auto_trader.modeling.config import TrainConfig


@pytest.mark.skip(reason="called by test_evaluate.test_main")
def test_main(tmp_path: Path) -> None:
    config = cast(
        TrainConfig,
        OmegaConf.merge(
            OmegaConf.structured(TrainConfig),
            OmegaConf.create(
                {
                    "output_dir": str(tmp_path / "output"),
                    "max_epochs": 1,
                    "neptune": {"mode": "debug"},
                    "data": {
                        "symbol": "usdjpy",
                        "cleansed_data_dir": str(tmp_path / "cleansed"),
                        "yyyymm_begin": 202301,
                        "yyyymm_end": 202301,
                    },
                    "net": {
                        "numerical_emb_dim": 1,
                        "categorical_emb_dim": 1,
                        "base_cnn_out_channels": [10],
                        "base_cnn_kernel_sizes": [5],
                        "base_fc_hidden_dims": [4],
                        "head_hidden_dims": [2],
                    },
                }
            ),
        ),
    )

    index = pd.date_range("2023-1-1 00:00", "2023-1-10 23:59", freq="1min")
    df_cleansed = pd.DataFrame(
        {
            "bid_open": np.random.randn(len(index)),
            "ask_open": np.random.randn(len(index)),
            "bid_high": np.random.randn(len(index)),
            "ask_high": np.random.randn(len(index)),
            "bid_low": np.random.randn(len(index)),
            "ask_low": np.random.randn(len(index)),
            "bid_close": np.random.randn(len(index)),
            "ask_close": np.random.randn(len(index)),
        },
        index=index,
    )
    (tmp_path / "cleansed").mkdir()
    df_cleansed.to_parquet(tmp_path / "cleansed" / "usdjpy-202301.parquet")

    train.main(config)
    assert (tmp_path / "output" / "params.pt").exists()
