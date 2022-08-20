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


def merge_bid_ask(df):
    # bid と ask の平均値を計算
    return pd.DataFrame({
        "open": (df["bid_open"] + df["ask_open"]) / 2,
        "high": (df["bid_high"] + df["ask_high"]) / 2,
        "low": (df["bid_low"] + df["ask_low"]) / 2,
        "close": (df["bid_close"] + df["ask_close"]) / 2,
    }, index=df.index)


def create_featurs(
    df: pd.DataFrame,
    symbol: str,
    timings: List[str],
    freqs: List[str],
    sma_timing: str,
    sma_window_size: int,
    lag_max: int,
) -> pd.DataFrame:
    """
    特徴量を作成する
    """

    df = merge_bid_ask(df)
    df = df.add_suffix("_1min")

    PIP_SCALE = 0.01 if symbol == "usdjpy" else 0.0001
    TIMING2OP = {"open": "first", "high": "max", "low": "min", "close": "last"}

    # 複数タイムスケールのデータ作成
    # import pdb; pdb.set_trace()
    df_dict = {
        "1min": df[[f"{timing}_1min" for timing in timings]]
    }
    for freq in freqs[1:]:
        df_dict[freq] = pd.concat({
            f"{timing}_{freq}": aggregate_time(df["open_1min"], freq, how=TIMING2OP[timing])
            for timing in timings
        }, axis=1)

    # 各タイムスケールで特徴量作成
    for freq, df_agg in df_dict.items():
        # 単純移動平均
        sma = (
            df_agg[f"{sma_timing}_{freq}"]
                .shift(1)
                .rolling(sma_window_size)
                .mean()
                .astype(np.float32)
        )
        # SettingWithCopyWarning が出るが、問題なさそうなので無視する。
        df_agg[f"{sma_timing}_{freq}_sma{sma_window_size}_frac"] = (sma / PIP_SCALE) % 100

        for lag_i in range(1, lag_max + 1):
            for timing in timings:
                df_agg[f"{timing}_{freq}_lag{lag_i}_cent"] = df_agg[f"{timing}_{freq}"].shift(lag_i) - sma

    # 全タイムスケールのデータをまとめる
    df_merged = df_dict["1min"].copy()
    for freq in freqs[1:]:
        df_agg = df_dict[freq]
        df_merged = pd.concat([df_merged, df_agg.reindex(df_merged.index, method="ffill")], axis=1)

    df_merged = df_merged.astype(np.float32)

    # 時間関連の特徴量
    df_merged["hour"] = df_merged.index.hour
    df_merged["day_of_week"] = df_merged.index.day_of_week
    df_merged["month"] = df_merged.index.month

    # 未来のデータを入力に使わないよう除外 (close_1min など)
    non_feature_names = []
    for freq in freqs:
        for timing in timings:
            non_feature_names.append(f"{timing}_{freq}")

    feature_names = list(set(df_merged.columns) - set(non_feature_names))

    return df_merged[feature_names]


def compute_critical_idxs(values: np.ndarray, thresh_hold: float) -> np.ndarray:
    """
    極大・極小をとるインデックスの配列を取得する。
    thresh_hold 以下の変動は無視する。
    """
    # TODO: uptrend から始めて大丈夫か？
    is_uptrend = True
    max_idx = 0
    max_value = values[0]
    min_idx = None
    min_value = np.inf
    critical_idxs = []
    for i in range(1, len(values)):
        v = values[i]
        # print(i, v, is_uptrend)
        if is_uptrend:
            if v > max_value:
                # 上昇中にさらに高値が更新された場合
                max_idx = i
                max_value = v
                # これまでの安値は無効化
                min_idx = None
                min_value = np.inf
            elif v < min_value:
                # 上昇中に安値が更新された場合
                min_idx = i
                min_value = v

                if max_value - min_value > thresh_hold:
                    # print("Add max_idx", max_idx)
                    # print("To downtrend")
                    # 下降に転換
                    critical_idxs.append(max_idx)
                    is_uptrend = False
                    max_idx = None
                    max_value = -np.inf
        else:
            if v < min_value:
                # 下降中にさらに安値が更新された場合
                min_idx = i
                min_value = v
                # これまでの高値は無効化
                max_idx = None
                max_value = -np.inf
            elif v > max_value:
                # 下降中に高値が更新された場合
                max_idx = i
                max_value = v

                if max_value - min_value > thresh_hold:
                    # print("Add min_idx", min_idx)
                    # print("To uptrend")
                    # 上昇に転換
                    critical_idxs.append(min_idx)
                    is_uptrend = True
                    min_idx = None
                    min_value = np.inf
        # print(max_value, min_value)

    if is_uptrend:
        critical_idxs.append(max_idx)
    else:
        critical_idxs.append(min_idx)

    return np.array(critical_idxs)


def create_labels_sub(values: np.ndarray, thresh_entry: float, thresh_hold: float):
    critical_idxs = compute_critical_idxs(values, thresh_hold)

    values_next_critical = np.empty(len(values))
    prev_cidx = 0
    for cidx in critical_idxs:
        values_next_critical[prev_cidx:cidx] = values[cidx]
        prev_cidx = cidx

    values_diff = values_next_critical - values

    # entry: 利益が出る and それ以降にさらに利益が出ることはない
    long_entry_labels_ = values_diff > thresh_entry
    short_entry_labels_ = values_diff < -thresh_entry
    # exit: 損失が出る
    long_exit_labels = values_diff < -thresh_hold
    short_exit_labels = values_diff > thresh_hold

    # "それ以降にさらに利益が出ることはない" の部分を考慮
    long_entry_labels = np.zeros(len(values), dtype=bool)
    short_entry_labels = np.zeros(len(values), dtype=bool)
    done_count = 0
    is_uptrend = True
    for cidx in critical_idxs:
        min_value = np.inf
        max_value = -np.inf
        # critical_idx までの index を逆順に走査して、エントリーのタイミングを探す
        for i in reversed(range(done_count, cidx)):
            if is_uptrend and values[i] < min_value:
                min_value = values[i]
                long_entry_labels[i] = long_entry_labels_[i]
            elif not is_uptrend and values[i] > max_value:
                max_value = values[i]
                short_entry_labels[i] = short_entry_labels_[i]

        done_count = cidx
        is_uptrend = not is_uptrend

    return long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels


def create_labels(
    df: pd.DataFrame,
    thresh_entry: float,
    thresh_hold: float,
) -> pd.DataFrame:
    """
    ラベルを作成する
    """

    df = merge_bid_ask(df)

    mean_high_low = (df["high"].values + df["low"].values) / 2
    long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels = create_labels_sub(mean_high_low, thresh_entry, thresh_hold)

    df_labels = pd.DataFrame({
        "long_entry": long_entry_labels,
        "short_entry": short_entry_labels,
        "long_exit": long_exit_labels,
        "short_exit": short_exit_labels,
    }, index=df.index)

    assert not (df_labels["long_entry"] & df_labels["long_exit"]).any()
    assert not (df_labels["short_entry"] & df_labels["short_exit"]).any()

    return df_labels
