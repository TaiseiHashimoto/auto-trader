import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
import re

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
    # TODO: config で使用する特徴量を変えられるように (month は不要？)
    df_out = {
        "hour": index.hour.astype(np.int32),
        "day_of_week": index.day_of_week.astype(np.int32),
        "month": index.month.astype(np.int32),
    }
    return pd.DataFrame(df_out, index=index)


def compute_sma(s: pd.Series, window_size: int) -> pd.Series:
    return s.rolling(window_size).mean().astype(np.float32)


def compute_ema(s: pd.Series, window_size: int) -> pd.Series:
    return s.ewm(span=window_size, adjust=False).mean().astype(np.float32)


def compute_sigma(s: pd.Series, window_size: int) -> pd.Series:
    return s.rolling(window_size).std(ddof=0).astype(np.float32)


def compute_fraction(s: pd.Series, base: float, ndigits: int):
    return (s / base) % int(10 ** ndigits)


def compute_macd(
    s: pd.Series,
    ema_window_size_short: int,
    ema_window_size_long: int,
    sma_window_size: int,
) -> Tuple[pd.Series, pd.Series]:
    ema_short = compute_ema(s, ema_window_size_short)
    ema_long = compute_ema(s, ema_window_size_long)
    macd = ema_short - ema_long
    macd_signal = compute_sma(macd, sma_window_size)
    return macd, macd_signal


def compute_rsi(s: pd.Series, window_size: int) -> pd.Series:
    diff = s.diff()
    up_diff = diff.clip(lower=0)
    down_diff = (-diff).clip(lower=0)
    up_diff_ma = compute_sma(up_diff, window_size)
    down_diff_ma = compute_sma(down_diff, window_size)
    return up_diff_ma / (up_diff_ma + down_diff_ma)


def compute_stochastics(s: pd.Series,
    k_window_size: int,
    d_window_size: int,
    sd_window_size: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    k_max = s.rolling(k_window_size).max()
    k_min = s.rolling(k_window_size).min()
    k = ((s - k_min) / (k_max - k_min)).astype(np.float32)
    d = compute_sma(k, d_window_size)
    sd = compute_sma(d, sd_window_size)
    return k, d, sd


def create_lagged_features(df: pd.DataFrame, lag_max: int) -> pd.DataFrame:
    df_out = {}
    for column in df.columns:
        for lag_i in range(1, lag_max + 1):
            df_out[f"{column}_lag{lag_i}"] = df[column].shift(lag_i)

    return pd.DataFrame(df_out, index=df.index)


def create_features(
    df: pd.DataFrame,
    df_critical: pd.DataFrame,
    symbol: str,
    timings: List[str],
    freqs: List[str],
    sma_timing: str,
    sma_window_sizes: List[int],
    sma_window_size_center: int,
    sma_frac_ndigits: int,
    sigma_timing: str,
    sigma_window_sizes: List[int],
    lag_max: int,
    start_hour: int,
    end_hour: int,
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
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
        df_sma_frac = sma_frac.shift(1).to_frame().add_suffix("_frac_lag1")

        df_sigma = pd.DataFrame({
            f"sigma{sigma_window_size}_lag1": compute_sigma(df_dict[freq][sigma_timing], sigma_window_size).shift(1)
            for sigma_window_size in sigma_window_sizes
        })

        df_cont_dict[freq] = pd.concat([df_sma_frac, df_sigma], axis=1)

    df_time = create_time_features(df.index)

    assert (df.index == df_critical.index).all()
    df_critical_values = df_critical[[c for c in df_critical.columns if re.match(r"prev[0-9]+_pre_critical_values", c)]]
    # 中心化
    df_critical_values = df_critical_values - df_seq_dict["1min"][f"sma{sma_window_size_center}"].values[:, np.newaxis]
    df_critical_idxs = df_critical[[c for c in df_critical.columns if re.match(r"prev[0-9]+_pre_critical_idxs", c)]]
    # HACK: 該当なしの場合に -1 になることを使っている
    df_critical_idxs = (df_critical_idxs.replace(-1, np.nan) - np.arange(len(df))[:, np.newaxis])
    # df_critical_uptrends = df_critical[["pre_uptrends"]].astype(np.int32)

    df_cont_dict["1min"] = pd.concat([
        df_cont_dict["1min"],
        df_time,
        df_critical_values.shift(1).add_suffix("_lag1"),
        df_critical_idxs.shift(1).add_suffix("_lag1"),
        # df_critical_uptrends.shift(1).add_suffix("_lag1"),
    ], axis=1)

    # データが足りている最初の時刻を求める
    first_index = pd.Timestamp("1900-1-1 00:00:00")
    for freq in df_dict:
        assert (df_seq_dict[freq].index == df_cont_dict[freq].index).all()

        nan_mask = df_seq_dict[freq].isnull().any(axis=1)
        notnan_idxs = (~nan_mask).values.nonzero()[0]
        first_idx = notnan_idxs[0] + lag_max
        first_index = max(first_index, df_seq_dict[freq].index[first_idx])

    available_mask = (
        (df.index >= first_index)
        & (df.index.hour >= start_hour)
        & (df.index.hour <  end_hour)
        & ~((df.index.month == 12) & (df.index.day == 25))
    )
    base_index = df.index[available_mask]

    return base_index, {"sequential": df_seq_dict, "continuous": df_cont_dict}


def compute_critical_idxs(values: np.ndarray, thresh_hold: float) -> Tuple[np.ndarray, np.ndarray]:
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
    critical_switch_idxs = []
    for i in range(1, len(values)):
        v = values[i]
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
                    # 下降に転換
                    critical_idxs.append(max_idx)
                    critical_switch_idxs.append(i)
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
                    # 上昇に転換
                    critical_idxs.append(min_idx)
                    critical_switch_idxs.append(i)
                    is_uptrend = True
                    min_idx = None
                    min_value = np.inf

    # TODO: 最後を特別扱いする必要はないかも
    if is_uptrend:
        critical_idxs.append(max_idx)
    else:
        critical_idxs.append(min_idx)

    critical_switch_idxs.append(len(values))

    assert len(critical_idxs) == len(critical_switch_idxs)

    return np.array(critical_idxs), np.array(critical_switch_idxs)


def compute_neighbor_critical_idxs(
    size: int,
    critical_idxs: np.ndarray,
    offset: int,  # 0 のとき次の critical_idx を求める
) -> np.ndarray:
    neighbor_critical_idxs = np.empty(size, dtype=np.int32)
    prev_cidx = 0
    for i in range(len(critical_idxs) + 1):
        if i < len(critical_idxs):
            cidx = critical_idxs[i]
        else:
            cidx = size

        j = i + offset
        if 0 <= j < len(critical_idxs):
            ncidx = critical_idxs[j]
        else:
            # HACK: 対応する critical_idx がない場合 -1 にしている
            ncidx = -1

        neighbor_critical_idxs[prev_cidx:cidx] = ncidx
        prev_cidx = cidx

    return neighbor_critical_idxs


def compute_neighbor_pre_critical_idxs(
    size: int,
    critical_idxs: np.ndarray,
    critical_switch_idxs: np.ndarray,
    offset: int,  # 0 のとき次の critical_idx を求める
) -> np.ndarray:
    assert critical_switch_idxs[-1] == size

    neighbor_critical_idxs = np.empty(size, dtype=np.int32)
    prev_csidx = 0
    for i in range(len(critical_switch_idxs)):
        csidx = critical_switch_idxs[i]

        j = i + offset
        if 0 <= j < len(critical_idxs):
            ncidx = critical_idxs[j]
        else:
            # HACK: 対応する critical_idx がない場合 -1 にしている
            ncidx = -1

        neighbor_critical_idxs[prev_csidx:csidx] = ncidx
        prev_csidx = csidx

    return neighbor_critical_idxs


def compute_trends(size: int, critical_idxs: np.ndarray) -> np.array:
    uptrend = True
    prev_cidx = 0
    uptrends = np.empty(size, dtype=bool)
    for cidx in critical_idxs:
        uptrends[prev_cidx:cidx] = uptrend
        uptrend = not uptrend
        prev_cidx = cidx

    if critical_idxs[-1] < size:
        uptrends[critical_idxs[-1]:size] = uptrend

    return uptrends


def compute_critical_info(
    df: pd.DataFrame,
    thresh_hold: float,
    prev_max: int,
) -> pd.DataFrame:
    critical_info = {}

    values = (df["high"].values + df["low"].values) / 2
    critical_info["values"] = values

    critical_idxs, critical_switch_idxs = compute_critical_idxs(values, thresh_hold)

    def get_values_from_idxs(values: np.ndarray, idxs: np.ndarray):
        v = values[idxs].copy()
        missing = idxs == -1
        v[missing] = np.nan
        return v

    for prev_i in range(1, prev_max + 1):
        prev_critical_idxs = compute_neighbor_critical_idxs(
            size=len(df),
            critical_idxs=critical_idxs,
            offset=-prev_i,
        )
        critical_info[f"prev{prev_i}_critical_idxs"] = prev_critical_idxs
        critical_info[f"prev{prev_i}_critical_values"] = get_values_from_idxs(values, prev_critical_idxs)

        prev_pre_critical_idxs = compute_neighbor_pre_critical_idxs(
            size=len(df),
            critical_idxs=critical_idxs,
            critical_switch_idxs=critical_switch_idxs,
            offset=-prev_i,
        )
        critical_info[f"prev{prev_i}_pre_critical_idxs"] = prev_pre_critical_idxs
        critical_info[f"prev{prev_i}_pre_critical_values"] = get_values_from_idxs(values, prev_pre_critical_idxs)

    next_critical_idxs = compute_neighbor_critical_idxs(
        size=len(df),
        critical_idxs=critical_idxs,
        offset=0
    )
    critical_info["next_critical_idxs"] = next_critical_idxs
    critical_info["next_critical_values"] = get_values_from_idxs(values, next_critical_idxs)

    critical_info["uptrends"] = compute_trends(len(df), critical_idxs)
    critical_info["pre_uptrends"] = compute_trends(len(df), critical_switch_idxs)

    return pd.DataFrame(critical_info, index=df.index)


def create_critical_labels(
    df: pd.DataFrame,
    thresh_entry: float,
    thresh_hold: float,
) -> pd.DataFrame:
    values = (df["high"].values + df["low"].values) / 2

    critical_idxs, _ = compute_critical_idxs(values, thresh_hold)

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


def create_critical2_labels(
    df_critical: pd.DataFrame,
    thresh_entry: float,
) -> pd.DataFrame:
    values = df_critical["values"].values
    next_critical_values = df_critical["next_critical_values"].values
    uptrends = df_critical["uptrends"].values
    pre_uptrends = df_critical["pre_uptrends"].values
    pre_uptrends_lag1 = df_critical["pre_uptrends"].shift(1, fill_value=True).values
    values_diff = next_critical_values - values

    # entry: 利益が出る and 暫定トレンドが順方向
    # pre_uptrends_lag1 だけでは周回遅れになる可能性がため、pre_uptrends についてもチェックする
    long_entry_labels  = (values_diff >= thresh_entry)  & pre_uptrends  & pre_uptrends_lag1
    short_entry_labels = (values_diff <= -thresh_entry) & ~pre_uptrends & ~pre_uptrends_lag1
    # exit: トレンドが逆方向
    long_exit_labels  = ~uptrends
    short_exit_labels = uptrends

    return merge_labels(df_critical.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_smadiff_labels(
    df: pd.DataFrame,
    window_size_before: int,
    window_size_after: int,
    thresh_entry: float,
    thresh_hold: float,
) -> pd.DataFrame:
    values = pd.Series((df["high"].values + df["low"].values) / 2, index=df.index)
    sma_before = compute_sma(values, window_size_before)
    sma_after = compute_sma(values.shift(-window_size_after), window_size_after)
    sma_diff = sma_after.values - sma_before.values

    long_entry_labels  = sma_diff > thresh_entry
    short_entry_labels = sma_diff < -thresh_entry
    long_exit_labels   = sma_diff < -thresh_hold
    short_exit_labels  = sma_diff > thresh_hold

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_future_labels(
    df: pd.DataFrame,
    future_step_min: int,
    future_step_max: int,
    thresh_entry: float,
    thresh_hold: float,
) -> pd.DataFrame:
    values = pd.Series((df["high"].values + df["low"].values) / 2, index=df.index)
    future_sma = compute_sma(values, future_step_max - future_step_min + 1).shift(-future_step_max)
    diff = future_sma - values

    long_entry_labels  = diff >= thresh_entry
    short_entry_labels = diff <= -thresh_entry
    long_exit_labels   = diff < -thresh_hold
    short_exit_labels  = diff > thresh_hold

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_smatrend_labels(
    df: pd.DataFrame,
    window_size: int,
    step_before: int,
    step_after: int,
    thresh_entry: float,
) -> pd.DataFrame:
    assert window_size % 2 == 1

    values = pd.Series((df["high"].values + df["low"].values) / 2, index=df.index)
    sma = compute_sma(values, window_size).shift(-(window_size//2))

    ascending  = sma > sma.shift(1)
    descending = sma < sma.shift(1)

    ascending_before  = np.ones(len(df), dtype=bool)
    descending_before = np.ones(len(df), dtype=bool)
    for i in range(step_before):
        ascending_before  &= ascending.shift(i).fillna(False).values
        descending_before &= descending.shift(i).fillna(False).values

    ascending_after  = np.ones(len(df), dtype=bool)
    descending_after = np.ones(len(df), dtype=bool)
    for i in range(step_after):
        ascending_after  &= ascending.shift(-(i+1)).fillna(False).values
        descending_after &= descending.shift(-(i+1)).fillna(False).values

    lift_before = sma - sma.shift(step_before)
    lift_after  = sma.shift(-step_after) - sma

    long_entry_labels = (
        ascending_before & ascending_after
        & (lift_before >= thresh_entry) & (lift_after >= thresh_entry)
    )
    short_entry_labels = (
        descending_before & descending_after
        & (lift_before <= -thresh_entry) & (lift_after <= -thresh_entry)
    )
    long_exit_labels  = (sma.shift(-1) <= sma).values
    short_exit_labels = (sma.shift(-1) >= sma).values

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels)


def create_gain_labels(
    df: pd.DataFrame,
    future_step_min: int,
    future_step_max: int,
    entry_bias: float,
    exit_bias: float,
):
    values = pd.Series((df["high"].values + df["low"].values) / 2, index=df.index)
    future_sma = compute_sma(values, future_step_max - future_step_min + 1).shift(-future_step_max)
    diff = future_sma - values

    long_entry_labels  = diff + entry_bias
    short_entry_labels = -diff + entry_bias
    long_exit_labels   = diff + exit_bias
    short_exit_labels  = -diff + exit_bias

    return merge_labels(df.index, long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels, validate=False)


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
    validate: bool = True,
):
    if validate:
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


def create_labels(
    label_type: str,
    df: pd.DataFrame,
    df_x_dict: Dict[str, Dict[str, pd.DataFrame]],
    df_critical,
    label_params: Dict[str, Any],
) -> pd.DataFrame:
    if label_type == "critical":
        df_y = create_critical_labels(df, **label_params)
    elif label_type == "critical2":
        df_y = create_critical2_labels(df_critical, **label_params)
    elif label_type == "smadiff":
        df_y = create_smadiff_labels(df, **label_params)
    elif label_type == "future":
        df_y = create_future_labels(df, **label_params)
    elif label_type == "smatrend":
        df_y = create_smatrend_labels(df, **label_params)
    elif label_type == "gain":
        df_y = create_gain_labels(df, **label_params)
    elif label_type == "dummy1":
        df_y = create_dummy1_labels(df.index, **label_params)
    elif label_type == "dummy2":
        df_y = create_dummy2_labels(df_x_dict, **label_params)
    elif label_type == "dummy3":
        df_y = create_dummy3_labels(df_x_dict, **label_params)
    else:
        raise ValueError(f"Unknown label_type `{label_type}`")

    return df_y


def calc_specificity(label: np.ndarray, pred: np.ndarray) -> float:
    return ((~label) & (~pred)).sum() / (~label).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
