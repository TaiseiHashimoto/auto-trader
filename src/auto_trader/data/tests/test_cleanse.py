from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from auto_trader.common import utils
from auto_trader.data import cleanse
from auto_trader.data.config import CleanseConfig


def test_read_raw_data(tmp_path: Path) -> None:
    timestamp = (
        pd.date_range("2023-1-1 00:00", "2023-1-31 23:59", freq="1min").astype(int)
        # n sec -> m sec
        // 10**6
    )
    df_bid = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": np.full(len(timestamp), 0.0),
            "high": np.full(len(timestamp), 2.0),
            "low": np.full(len(timestamp), -1.0),
            "close": np.full(len(timestamp), 1.0),
        }
    )
    df_ask = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": np.full(len(timestamp), 0.1),
            "high": np.full(len(timestamp), 2.1),
            "low": np.full(len(timestamp), -1.1),
            "close": np.full(len(timestamp), 1.1),
        }
    )
    df_bid.to_csv(tmp_path / "usdjpy-bid-20230101-20230131.csv", index=False)
    df_ask.to_csv(tmp_path / "usdjpy-ask-20230101-20230131.csv", index=False)

    df_actual = cleanse.read_raw_data(
        symbol="usdjpy",
        raw_data_dir=str(tmp_path),
        yyyymm=202301,
        convert_timezone=True,
    )

    df_expected = pd.DataFrame(
        {
            "bid_open": df_bid["open"].values.astype(np.float32),
            "bid_high": df_bid["high"].values.astype(np.float32),
            "bid_low": df_bid["low"].values.astype(np.float32),
            "bid_close": df_bid["close"].values.astype(np.float32),
            "ask_open": df_ask["open"].values.astype(np.float32),
            "ask_high": df_ask["high"].values.astype(np.float32),
            "ask_low": df_ask["low"].values.astype(np.float32),
            "ask_close": df_ask["close"].values.astype(np.float32),
        },
        # UCT+2 (1月なのでサマータイムなし)
        index=pd.date_range("2023-1-1 02:00", "2023-2-1 01:59", freq="1min"),
    )
    pd.testing.assert_frame_equal(
        df_actual, df_expected, check_names=False, check_freq=False
    )


def test_remove_flat_data() -> None:
    # この年の年末年始は 12/25 金, 1/1 金
    index_org = pd.date_range("2020-12-20 00:00", "2021-1-10 23:59", freq="1min")
    df_org = pd.DataFrame(index=index_org)
    df_actual = cleanse.remove_flat_data(df_org)
    index_actual = cast(pd.DatetimeIndex, df_actual.index)

    # 月~金のみを含む
    assert set(index_actual.dayofweek) == {0, 1, 2, 3, 4}
    # 月~金は1分間隔ですべて含む (1/1, 12/25 は特別扱い)
    count_org = (index_org.dayofweek < 5).sum()
    count_returned = (index_actual.dayofweek < 5).sum()
    assert count_org == count_returned + 60 * (24 + 14)
    # 1/1 は含まない
    assert not ((index_actual.month == 1) & (index_actual.day == 1)).any()
    # 12/25 は 9:59 まで含む
    mask_christmas = (index_actual.month == 12) & (index_actual.day == 25)
    assert not (mask_christmas & index_actual.hour >= 10).any()
    assert mask_christmas.sum() == 60 * 10


def test_main(tmp_path: Path) -> None:
    config = utils.get_config(
        CleanseConfig,
        [
            "symbol=usdjpy",
            "raw_data_dir=" + str(tmp_path / "raw"),
            "cleansed_data_dir=" + str(tmp_path / "cleansed"),
            "yyyymm_begin=202302",
            "yyyymm_end=202302",
            "recreate_latest=True",
            "validate=False",
        ],
    )

    timestamp_202301 = (
        pd.date_range("2023-1-1 00:00", "2023-1-31 23:59", freq="1min").astype(int)
        // 10**6
    )
    timestamp_202302 = (
        pd.date_range("2023-2-1 00:00", "2023-2-28 23:59", freq="1min").astype(int)
        // 10**6
    )
    df_bid_202301 = pd.DataFrame(
        {
            "timestamp": timestamp_202301,
            "open": np.full(len(timestamp_202301), 0.0),
            "high": np.full(len(timestamp_202301), 2.0),
            "low": np.full(len(timestamp_202301), -1.0),
            "close": np.full(len(timestamp_202301), 1.0),
        }
    )
    df_ask_202301 = pd.DataFrame(
        {
            "timestamp": timestamp_202301,
            "open": np.full(len(timestamp_202301), 0.1),
            "high": np.full(len(timestamp_202301), 2.1),
            "low": np.full(len(timestamp_202301), -1.1),
            "close": np.full(len(timestamp_202301), 1.1),
        }
    )
    df_bid_202302 = pd.DataFrame(
        {
            "timestamp": timestamp_202302,
            "open": np.full(len(timestamp_202302), 10.0),
            "high": np.full(len(timestamp_202302), 12.0),
            "low": np.full(len(timestamp_202302), 9.0),
            "close": np.full(len(timestamp_202302), 11.0),
        }
    )
    df_ask_202302 = pd.DataFrame(
        {
            "timestamp": timestamp_202302,
            "open": np.full(len(timestamp_202302), 10.1),
            "high": np.full(len(timestamp_202302), 12.1),
            "low": np.full(len(timestamp_202302), 9.1),
            "close": np.full(len(timestamp_202302), 11.1),
        }
    )
    (tmp_path / "raw").mkdir()
    df_bid_202301.to_csv(
        tmp_path / "raw" / "usdjpy-bid-20230101-20230131.csv", index=False
    )
    df_ask_202301.to_csv(
        tmp_path / "raw" / "usdjpy-ask-20230101-20230131.csv", index=False
    )
    df_bid_202302.to_csv(
        tmp_path / "raw" / "usdjpy-bid-20230201-20230228.csv", index=False
    )
    df_ask_202302.to_csv(
        tmp_path / "raw" / "usdjpy-ask-20230201-20230228.csv", index=False
    )

    cleanse.main(config)

    assert len(list((tmp_path / "cleansed").glob("*.parquet"))) == 1
    df_actual = pd.read_parquet(tmp_path / "cleansed" / "usdjpy-202302.parquet")

    df_expected = cleanse.remove_flat_data(
        cast(
            pd.DataFrame,
            pd.concat(
                [
                    cleanse.read_raw_data(
                        symbol="usdjpy",
                        raw_data_dir=str(tmp_path / "raw"),
                        yyyymm=202301,
                    ).loc["2023-02"],
                    cleanse.read_raw_data(
                        symbol="usdjpy",
                        raw_data_dir=str(tmp_path / "raw"),
                        yyyymm=202302,
                    ).loc["2023-02"],
                ]
            )
            / utils.get_pip_scale("usdjpy"),
        )
    )
    pd.testing.assert_frame_equal(
        df_actual, df_expected, check_names=False, check_freq=False
    )
