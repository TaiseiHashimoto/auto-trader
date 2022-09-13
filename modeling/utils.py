import numpy as np
import pandas as pd
from typing import Optional, List, Dict

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

    return s_agg.dropna()


def merge_bid_ask(df):
    # bid と ask の平均値を計算
    return pd.DataFrame({
        "open":  (df["bid_open"]  + df["ask_open"] ) / 2,
        "high":  (df["bid_high"]  + df["ask_high"] ) / 2,
        "low":   (df["bid_low"]   + df["ask_low"]  ) / 2,
        "close": (df["bid_close"] + df["ask_close"]) / 2,
    }, index=df.index)


def resample(
    df_1min: pd.DataFrame,
    freqs: List[str],
) -> Dict[str, pd.DataFrame]:
    TIMING2OP = {"open": "first", "high": "max", "low": "min", "close": "last"}

    # 複数タイムスケールのデータ作成
    df_dict = {}
    for freq in freqs:
        if freq == "1min":
            df_dict[freq] = df_1min.copy()
        else:
            df_dict[freq] = pd.concat({
                timing: aggregate_time(df_1min[timing], freq, how=TIMING2OP[timing])
                for timing in df_1min.columns
            }, axis=1)

    return df_dict


def align_frequency(base_index: pd.DatetimeIndex, df_dict: Dict[str, pd.DataFrame]):
    # 全タイムスケールのデータをまとめる
    df_merged = pd.DataFrame(index=base_index)
    for freq, df in df_dict.items():
        if freq == "1min":
            df_aligned = df.loc[base_index]
        else:
            df_aligned = df.reindex(base_index, method="ffill")

        df_merged = pd.concat([df_merged, df_aligned.add_suffix(f"_{freq}")], axis=1)

    return df_merged


def create_time_features(index: pd.DatetimeIndex):
    df_out = {
        "hour": index.hour,
        "day_of_week": index.day_of_week,
        "month": index.month,
    }
    return pd.DataFrame(df_out, index=index)


def compute_sma(s: pd.Series, sma_window_size: int) -> pd.Series:
    sma = (
        s
        .rolling(sma_window_size)
        .mean()
        .astype(np.float32)
    )
    return sma


def compute_fraction(s: pd.Series, base: float, ndigits: int):
    return (s / base) % int(10 ** ndigits)


def create_lagged_features(df: pd.DataFrame, lag_max: int) -> pd.DataFrame:
    df_out = {}
    for column in df.columns:
        for lag_i in range(1, lag_max + 1):
            df_out[f"{column}_lag{lag_i}"] = df[column].shift(lag_i)

    return pd.DataFrame(df_out, index=df.index)


def create_features(
    df: pd.DataFrame,
    symbol: str,
    timings: List[str],
    freqs: List[str],
    lag_max: int,
    sma_timing: str,
    sma_window_sizes: List[int],
    sma_window_size_center: int,
    sma_frac_ndigits: int,
) -> Dict[str, pd.DataFrame]:
    pip_scale = common_utils.get_pip_scale(symbol)

    df_dict = resample(df, freqs)

    df_seq_dict = {}
    df_cont_dict = {}
    for freq in df_dict:
        df_sma = pd.DataFrame({
            f"sma{sma_window_size}": compute_sma(df_dict[freq][sma_timing], sma_window_size)
            for sma_window_size in sma_window_sizes
        })
        df_seq_dict[freq] = pd.concat([df_dict[freq][timings], df_sma], axis=1)

        sma_frac = compute_fraction(df_sma[f"sma{sma_window_size_center}"], base=pip_scale, ndigits=sma_frac_ndigits)
        sma_frac = sma_frac.shift(1)
        # name: sma* -> sma*_frac
        sma_frac.name = sma_frac.name + "_frac_lag1"
        # seq と形式を合わせるため、Series から DataFrame に変換
        df_cont_dict[freq] = sma_frac.to_frame()

    df_time = create_time_features(df.index)
    df_cont_dict["1min"] = pd.concat([df_cont_dict["1min"], df_time], axis=1)

    # データが足りない行を削除
    first_index = pd.Timestamp("1900-1-1 00:00:00")
    for freq in df_dict:
        nan_mask = df_seq_dict[freq].isnull().any(axis=1)
        notnan_idxs = (~nan_mask).values.nonzero()[0]
        first_idx = notnan_idxs[0] + lag_max
        first_index = max(first_index, df_seq_dict[freq].index[first_idx])

    # TODO: 学習に適した時間帯だけ抽出
    base_index = df.index[df.index >= first_index]

    return base_index, {"sequential": df_seq_dict, "continuous": df_cont_dict}


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


def create_critical_labels(
    df: pd.DataFrame,
    thresh_entry: float,
    thresh_hold: float,
) -> pd.DataFrame:
    values = (df["high"].values + df["low"].values) / 2

    critical_idxs = compute_critical_idxs(values, thresh_hold)

    values_next_critical = np.empty(len(values))
    prev_cidx = 0
    for cidx in critical_idxs:
        values_next_critical[prev_cidx:cidx] = values[cidx]
        prev_cidx = cidx
    # TODO: 最後を無理やり critical index 扱いにしているが問題ない？
    values_next_critical[prev_cidx:] = values[-1]

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

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_smadiff_labels(
    df: pd.DataFrame,
    window_size: int,
    thresh_entry: float,
    thresh_hold: float,
):
    values = pd.Series((df["high"].values + df["low"].values) / 2, index=df.index)
    sma = compute_sma(values, window_size)
    sma_before = sma.shift(1)
    sma_after = sma.shift(-window_size)
    sma_diff = sma_after.values - sma_before.values

    long_entry_labels  = sma_diff > thresh_entry
    short_entry_labels = sma_diff < -thresh_entry
    long_exit_labels   = sma_diff < -thresh_hold
    short_exit_labels  = sma_diff > thresh_hold

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_dummy1_labels(index: pd.DatetimeIndex) -> pd.DataFrame:
    long_entry_labels  = (index.hour >= 0)  & (index.hour < 6)
    short_entry_labels = (index.hour >= 6)  & (index.hour < 12)
    long_exit_labels   = (index.hour >= 12) & (index.hour < 18)
    short_exit_labels  = (index.hour >= 18) & (index.hour < 24)
    return merge_labels(index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_dummy2_labels(df_x_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    sma_colname = None
    for colname in df_x_dict["continuous"]["1min"].columns:
        if colname.startswith("sma"):
            sma_colname = colname
            break

    assert sma_colname is not None
    values = df_x_dict["continuous"]["1min"][sma_colname].values

    long_entry_labels  = (values >=  0) & (values < 25)
    short_entry_labels = (values >= 25) & (values < 50)
    long_exit_labels   = (values >= 50) & (values < 75)
    short_exit_labels  = (values >= 75) & (values < 100)

    index = df_x_dict["continuous"]["1min"].index
    return merge_labels(index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_dummy3_labels(df_x_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    lag1 = df_x_dict["sequential"]["1min"]["close"].shift(1).values
    lag2 = df_x_dict["sequential"]["1min"]["close"].shift(2).values
    lag3 = df_x_dict["sequential"]["1min"]["close"].shift(3).values

    long_entry_labels  = (lag1 >  lag2) & (lag2 >  lag3)
    short_entry_labels = (lag1 >  lag2) & (lag2 <= lag3)
    long_exit_labels   = (lag1 <= lag2) & (lag2 >  lag3)
    short_exit_labels  = (lag1 <= lag2) & (lag2 <= lag3)

    index = df_x_dict["sequential"]["1min"].index
    return merge_labels(index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def merge_labels(
    index:pd.DatetimeIndex,
    long_entry_labels: np.ndarray,
    short_entry_labels: np.ndarray,
    long_exit_labels: np.ndarray,
    short_exit_labels: np.ndarray,
):
    assert not (long_entry_labels & short_entry_labels).any()
    assert not (long_entry_labels & long_exit_labels).any()
    assert not (short_entry_labels & short_exit_labels).any()

    df_labels = pd.DataFrame({
        "long_entry": long_entry_labels,
        "short_entry": short_entry_labels,
        "long_exit": long_exit_labels,
        "short_exit": short_exit_labels,
    }, index=index)
    return df_labels
