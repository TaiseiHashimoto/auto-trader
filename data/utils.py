import numpy as np
import pandas as pd
import datetime
import glob
from typing import Optional


def read_raw_data(
    symbol: str,
    year: int,
    month: int,
    convert_timezone: Optional[bool] = True,
    data_directory: Optional[str] = './raw'
):
    """
    Dukascopy から取得した生データを読み込む
    """

    bid_paths = glob.glob(f"{data_directory}/{symbol}-m1-bid-{year}-{month:02d}-*.csv")
    ask_paths = glob.glob(f"{data_directory}/{symbol}-m1-ask-{year}-{month:02d}-*.csv")
    assert len(bid_paths) == 1 and len(ask_paths) == 1

    df_bid = pd.read_csv(bid_paths[0])
    df_ask = pd.read_csv(ask_paths[0])
    assert (df_bid["timestamp"] == df_ask["timestamp"]).all()

    datetime = pd.DatetimeIndex(pd.to_datetime(df_bid["timestamp"], unit='ms'))
    if convert_timezone:
        # UTC -> 東部時間 -> Eastern Europe Time
        datetime = (datetime.tz_localize('UTC').tz_convert('US/Eastern') + pd.Timedelta('7 hours')).tz_localize(None)

    df = pd.DataFrame({
        "bid_open": df_bid["open"].values,
        "bid_high": df_bid["high"].values,
        "bid_low": df_bid["low"].values,
        "bid_close": df_bid["close"].values,
        "ask_open": df_ask["open"].values,
        "ask_high": df_ask["high"].values,
        "ask_low": df_ask["low"].values,
        "ask_close": df_ask["close"].values,
    }, index=datetime)

    return df


def remove_flat_data(df: pd.DataFrame):
    """
    フラット期間を除去する
    フラット期間: 土日全日・1/1 全日・12/25 10:00~
    """
    mask = (
        (df.index.dayofweek < 5) &
        ~((df.index.month == 1) & (df.index.day == 1)) &
        ~((df.index.month == 12) & (df.index.day == 25) & (df.index.hour >= 10))
    )
    return df.loc[mask]


# def extract_valid_datetime(df):
#     # データとして適切な時間帯を抽出する
#     # 月-金 02:00 ~ 21:59 (EET)
#     # ただし 1/1, 12/25 は除く

#     # print("Warning: Timezone of dataframe's index must be EET.")

#     mask = (
#         ((df.index.dayofweek >= 0) & (df.index.dayofweek <= 4)) &
#         ((df.index.hour >= 2) & (df.index.hour < 22)) &
#         ~((df.index.month == 1) & (df.index.day == 1)) &
#         ~((df.index.month == 12) & (df.index.day == 25))
#     )

#     return df.loc[mask]
