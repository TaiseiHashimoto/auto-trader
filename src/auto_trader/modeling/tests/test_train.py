from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_trader.common import utils
from auto_trader.modeling import train
from auto_trader.modeling.config import TrainConfig


@pytest.mark.skip(reason="called by test_evaluate.test_main")
def test_main(tmp_path: Path) -> None:
    config = utils.get_config(
        TrainConfig,
        [
            "output_dir=" + str(tmp_path / "output"),
            "max_epochs=1",
            "neptune.mode=debug",
            "symbols=[usdjpy]",
            "cleansed_data_dir=" + str(tmp_path / "cleansed"),
            "yyyymm_begin=202301",
            "yyyymm_end=202301",
            "net.numerical_emb_dim=2",
            "net.categorical_emb_dim=1",
            "net.emb_output_dim=4",
            "net.base_net_type=attention",
            "net.base_attention_num_layers=1",
            "net.base_attention_feedforward_dim=8",
            "net.base_fc_hidden_dims=[4]",
            "net.head_hidden_dims=[2]",
            "feature.timeframes=[1min]",
        ],
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
    (tmp_path / "cleansed" / "usdjpy").mkdir(parents=True)
    df_cleansed.to_parquet(tmp_path / "cleansed" / "usdjpy" / "202301.parquet")

    train.main(config)
    assert (tmp_path / "output" / "params.pt").exists()
