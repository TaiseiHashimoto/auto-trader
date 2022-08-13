import numpy as np
import pandas as pd
from typing import Optional, List

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import common_utils


def download_preprocessed_data(
    gcs: common_utils.GCSWrapper,
    symbol: str,
    year: int,
    month: int,
    data_directory: str,
    force: Optional[bool] = False
):
    """
    前処理したデータを GCS からダウンロード
    """
    file_name = f"{symbol}-{year}-{month:02d}.pkl"
    local_path = f"{data_directory}/{file_name}"
    if force or not pathlib.Path(local_path).exists():
        gcs.download_file(local_path=local_path, gcs_path=file_name)


def download_preprocessed_data_range(
    gcs: common_utils.GCSWrapper,
    symbol: str,
    year_from: int,
    month_from: int,
    year_to: int,
    month_to: int,
    data_directory: str
):
    """
    前処理したデータを範囲指定して GCS からダウンロード
    """
    year = year_from
    month = month_from
    while (year, month) <= (year_to, month_to):
        download_preprocessed_data(gcs, symbol, year, month, data_directory)
        year, month = common_utils.calc_year_month_offset(year, month, month_offset=1)


def read_preprocessed_data(
    symbol: str,
    year: int,
    month: int,
    data_directory: str
) -> pd.DataFrame:
    """
    前処理したデータを読み込む
    """
    df = pd.read_pickle(f"{data_directory}/{symbol}-{year}-{month:02d}.pkl")
    return df


def read_preprocessed_data_range(
    symbol: str,
    year_from: int,
    month_from: int,
    year_to: int,
    month_to: int,
    data_directory: str
) -> pd.DataFrame:
    """
    前処理したデータを範囲指定して読み込む
    """
    year = year_from
    month = month_from
    df = pd.DataFrame()
    while (year, month) <= (year_to, month_to):
        df = pd.concat([df, read_preprocessed_data(symbol, year, month, data_directory)])
        year, month = common_utils.calc_year_month_offset(year, month, month_offset=1)

    return df


def aggregate_time(s: pd.Series, freq: str, how: str) -> pd.DataFrame:
    s_agg = s.resample(freq)
    if how == "first":
        s_agg = s_agg.first()
    elif how == "last":
        s_agg = s_agg.last()
    elif how == "max":
        s_agg = s_agg.max()
    elif how == "min":
        s_agg = s_agg.min()
    else:
        raise ValueError(f"how must be \"first\", \"last\", \"max\", or \"min\"")

    # # データが足りている時間帯を特定
    # valid_mask = (
    #     (s_agg.index >= s.index[0]) &
    #     (s_agg.index <= s.index[-1] - pd.Timedelta(freq) + pd.Timedelta("1min"))
    # )
    # valid_index = s_agg.index[valid_mask]
    # valid_start = valid_index[0]
    # valid_end = valid_index[-1] + pd.Timedelta(freq) - pd.Timedelta("1min")

    # s_agg.loc[(s_agg.index < valid_start) | (s_agg.index > valid_end)] = np.nan
    # s_agg = s_agg.reindex(s.index, method="ffill")
    # s_agg.loc[s_agg.index > valid_end] = np.nan

    return s_agg.dropna()


def calc_slope(arr: np.ndarray, lookahead: int) -> np.ndarray:
    i_sum = lookahead * (lookahead + 1) / 2
    i2_sum = lookahead * (lookahead + 1) * (lookahead * 2 + 1) / 6
    arange = np.arange(1, lookahead + 1)
    slopes = []
    for i in range(len(arr) - lookahead):
        bias = arr[i]
        prod_sum = (arange * arr[i+1:i+1+lookahead]).sum()
        (prod_sum - bias * i_sum) / (i2_sum)
        slope = (prod_sum - bias * i_sum) / (i2_sum)
        slopes.append(slope)

    for i in range(lookahead):
        slopes.append(np.nan)

    return np.array(slopes)


def polyline_approx(arr: np.ndarray, tol: float, lookahead_max: int):
    approx_values = []
    approx_slopes = []
    # start_values = []
    # end_values = []
    # joint_idxs = []

    start_idx = 0
    # joint_idxs.append(start_idx)
    # start_values.append(arr[start_idx])

    while start_idx < len(arr) - 1:
        end_idx = min(start_idx + lookahead_max - 1, len(arr)-1)
        while True:
            reference = arr[start_idx:end_idx+1]
            approx = arr[start_idx] + np.linspace(0., 1., num=end_idx-start_idx+1) * (arr[end_idx] - arr[start_idx])
            if np.max(np.abs(reference - approx)) <= tol:
                break

            end_idx -= 1

        approx_values.extend(approx[:-1])
        slope = (arr[end_idx] - arr[start_idx]) / (end_idx - start_idx)
        approx_slopes.extend([slope] * (end_idx - start_idx))
        # joint_idxs.append(end_idx)
        # start_values.extend([arr[start_idx]] * (end_idx-start_idx))
        # end_values.extend([arr[end_idx]] * (end_idx - start_idx))
        start_idx = end_idx

    approx_values.append(arr[end_idx])
    approx_slopes.append(np.nan)
    # joint_flags.append(True)
    # end_values.append(np.nan)
    # return np.array(approx_values), np.array(start_values), np.array(end_values)
    # return np.array(approx_values), np.array(joint_idxs)
    return np.array(approx_values), np.array(approx_slopes)


def calc_critical_idxs(values: np.ndarray, joint_idxs: np.ndarray):
    state = None
    critical_idxs = []

    for i in range(len(joint_idxs) - 1):
        joint_idx_start = joint_idxs[i]
        joint_idx_end = joint_idxs[i+1]
        value_diff = values[joint_idx_end] - values[joint_idx_start]
        if value_diff > 0:
            new_state = "up"
        else:
            new_state = "down"

        if state != new_state:
            critical_idxs.append(joint_idx_start)
            state = new_state

    critical_idxs.append(joint_idx_end)
    return np.array(critical_idxs)


def calc_next_critical_values(values: np.ndarray, critical_idxs: np.ndarray):
    done_count = 0
    next_critical_values = []
    for critical_idx in critical_idxs[1:]:
        next_critical_values.extend([values[critical_idx]] * (critical_idx - done_count))
        done_count = critical_idx

    next_critical_values.append(np.nan)
    return np.array(next_critical_values)


# def align_time(index: pd.DatetimeIndex, s: pd.Series):




# def label_with_count(s: pd.Series, horizon: int, direction: str, tolerance: int):


# def add_candle(df: pd.DataFrame, freq: str) -> pd.DataFrame:
#     df.asfreq('1h')

