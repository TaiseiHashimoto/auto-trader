import glob
import os
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from auto_trader.common import utils
from auto_trader.data.config import CleanseConfig


def read_raw_data(
    symbol: str,
    raw_data_dir: str,
    yyyymm: int,
    convert_timezone: bool = True,
) -> pd.DataFrame:
    """
    Dukascopy から取得した生データを読み込む
    """

    bid_paths = glob.glob(os.path.join(raw_data_dir, f"{symbol}-bid-{yyyymm}*.csv"))
    ask_paths = glob.glob(os.path.join(raw_data_dir, f"{symbol}-ask-{yyyymm}*.csv"))
    if len(bid_paths) != 1 or len(ask_paths) != 1:
        raise RuntimeError(f"Raw data for {symbol} {yyyymm} is not properly prepared.")

    df_bid = pd.read_csv(bid_paths[0])
    df_ask = pd.read_csv(ask_paths[0])
    assert (df_bid["timestamp"] == df_ask["timestamp"]).all()

    datetime = pd.DatetimeIndex(pd.to_datetime(df_bid["timestamp"], unit="ms"))
    if convert_timezone:
        # UTC -> 東部時間 -> Eastern European Time
        datetime = (
            datetime.tz_localize("UTC").tz_convert("US/Eastern")
            + pd.Timedelta("7 hours")
        ).tz_localize(None)

    df = pd.DataFrame(
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
        index=datetime,
    )

    return df


def remove_flat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    フラット期間を除去する
    フラット期間: 土日全日・1/1 全日・12/25 10:00~
    """
    index = cast(pd.DatetimeIndex, df.index)
    mask = (
        (index.dayofweek < 5)
        & ~((index.month == 1) & (index.day == 1))
        & ~((index.month == 12) & (index.day == 25) & (index.hour >= 10))
    )
    return df.loc[mask]


def validate_data(df: pd.DataFrame, symbol: str) -> None:
    """
    データの妥当性を検証する
    """

    FLAT_RATIO_TOLERANCE = 0.1
    NO_MOVE_RATIO_TOLERANCE = 0.1
    BID_HIGHER_RATIO_TOLERANCE = 0.0

    # フラット期間が一定割合以下
    flat_idxs = np.nonzero(np.all(df.iloc[1:].values == df.iloc[:-1].values, axis=1))[0]
    flat_ratio = len(flat_idxs) / len(df)
    if flat_ratio > FLAT_RATIO_TOLERANCE:
        raise ValueError(
            f"flat_ratio is too high: {flat_ratio} > {FLAT_RATIO_TOLERANCE}"
        )

    # 4値同一が一定割合以下
    no_move_mask = (df["bid_high"] == df["bid_low"]) | (df["ask_high"] == df["ask_low"])
    no_move_ratio = no_move_mask.mean()
    if no_move_ratio > NO_MOVE_RATIO_TOLERANCE:
        raise ValueError(
            f"no_move_ratio is too high: {no_move_ratio} > {NO_MOVE_RATIO_TOLERANCE}"
        )

    # bid > ask が一定割合以下
    bid_higher_mask = (
        (df["bid_open"] > df["ask_open"])
        | (df["bid_high"] > df["ask_high"])
        | (df["bid_low"] > df["ask_low"])
        | (df["bid_close"] > df["ask_close"])
    )
    bid_higher_ratio = bid_higher_mask.mean()
    if bid_higher_ratio > BID_HIGHER_RATIO_TOLERANCE:
        raise ValueError(
            f"bid_higer_ratio is too high: "
            f"{bid_higher_ratio} > {BID_HIGHER_RATIO_TOLERANCE}"
        )

    # low < open, close < high の順になっている
    invalid_order_mask = (
        (df["bid_open"] < df["bid_low"])
        | (df["bid_close"] < df["bid_low"])
        | (df["bid_open"] > df["bid_high"])
        | (df["bid_close"] > df["bid_high"])
        | (df["ask_open"] < df["ask_low"])
        | (df["ask_close"] < df["ask_low"])
        | (df["ask_open"] > df["ask_high"])
        | (df["ask_close"] > df["ask_high"])
    )
    if invalid_order_mask.any():
        raise ValueError(
            "Order of prices (low <= open, close <= high) is not satisfied"
        )

    if symbol == "usdjpy":
        extreme_value_mask = (df < 5000) | (df > 20000)
    elif symbol == "eurusd":
        extreme_value_mask = (df < 8000) | (df > 16000)

    if extreme_value_mask.any(axis=None):
        raise ValueError(
            f"Found extreme values: min={df.min(axis=None)}, max={df.max(axis=None)}"
        )


def main(config: CleanseConfig) -> None:
    os.makedirs(config.cleansed_data_dir, exist_ok=True)

    if config.recreate_latest:
        # 最新ファイルを削除して作り直す
        cleansed_data_files = sorted(
            glob.glob(f"{config.cleansed_data_dir}/{config.symbol}-*.parquet")
        )
        if len(cleansed_data_files) > 0:
            latest_file_path = cleansed_data_files[-1]
            print(f"Delete {latest_file_path}")
            os.remove(latest_file_path)

    # データを整形して保存
    yyyymm = config.yyyymm_begin
    raw_data_buffer = {}
    while yyyymm <= config.yyyymm_end:
        print(yyyymm)

        cleansed_data_file = (
            f"{config.cleansed_data_dir}/{config.symbol}-{yyyymm}.parquet"
        )
        if os.path.exists(cleansed_data_file):
            print("Skip")
        else:
            raw_data_buffer[yyyymm] = read_raw_data(
                config.symbol, config.raw_data_dir, yyyymm, convert_timezone=True
            )
            # 元データファイルは UTC+0 基準で保存されているので, UTC+2/+3 に合わせるために前月のデータが2/3時間分だけ必要
            yyyymm_prev = utils.calc_yyyymm(yyyymm, month_delta=-1)
            if yyyymm_prev not in raw_data_buffer:
                raw_data_buffer[yyyymm_prev] = read_raw_data(
                    config.symbol,
                    config.raw_data_dir,
                    yyyymm_prev,
                    convert_timezone=True,
                )

            df_source = pd.concat(
                [raw_data_buffer[yyyymm_prev], raw_data_buffer[yyyymm]]
            ) / utils.get_pip_scale(config.symbol)
            del raw_data_buffer[yyyymm_prev]

            # 当月データを抽出
            df = cast(
                pd.DataFrame,
                df_source.loc[utils.parse_yyyymm(yyyymm).strftime("%Y-%m")],
            )
            df = remove_flat_data(df)

            if config.validate:
                validate_data(df, config.symbol)

            df.to_parquet(cleansed_data_file)

        yyyymm = utils.calc_yyyymm(yyyymm, month_delta=1)


if __name__ == "__main__":
    config = cast(CleanseConfig, utils.get_config(CleanseConfig))
    print(OmegaConf.to_yaml(config))

    main(config)
