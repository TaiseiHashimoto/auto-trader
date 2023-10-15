import os
from typing import Generator, Optional, Union

import numpy as np
import pandas as pd

from auto_trader.common import utils


def read_cleansed_data(
    symbol: str,
    yyyymm_begin: int,
    yyyymm_end: int,
    cleansed_data_dir: str,
) -> pd.DataFrame:
    """
    前処理したデータを範囲指定して読み込む
    """

    yyyymm = yyyymm_begin
    df_data = []
    while yyyymm <= yyyymm_end:
        df_data.append(
            pd.read_parquet(
                os.path.join(cleansed_data_dir, f"{symbol}-{yyyymm}.parquet")
            )
        )
        yyyymm = utils.calc_yyyymm(yyyymm, month_delta=1)

    return pd.concat(df_data, axis=0)


def merge_bid_ask(df):
    # bid と ask の平均値を計算
    return pd.DataFrame(
        {
            "open": (df["bid_open"] + df["ask_open"]) / 2,
            "high": (df["bid_high"] + df["ask_high"]) / 2,
            "low": (df["bid_low"] + df["ask_low"]) / 2,
            "close": (df["bid_close"] + df["ask_close"]) / 2,
        },
        index=df.index,
        dtype=np.float32,
    )


def resample(
    values_1min: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    if timeframe == "1min":
        return values_1min
    else:
        values_resampled = pd.DataFrame(
            {
                "open": values_1min["open"].resample(timeframe).first(),
                "high": values_1min["high"].resample(timeframe).max(),
                "min": values_1min["low"].resample(timeframe).min(),
                "close": values_1min["close"].resample(timeframe).last(),
            }
        )
        # 削除していた flat 期間が NaN になるので削除
        return values_resampled.dropna()


def calc_sma(s: pd.Series, window_size: int) -> pd.Series:
    return s.rolling(window_size).mean().astype(np.float32)


def calc_sigma(s: pd.Series, window_size: int) -> pd.Series:
    return s.rolling(window_size).std(ddof=0).astype(np.float32)


def calc_fraction(values: pd.Series, ndigits: int):
    return (values % int(10**ndigits)).astype(np.float32)


def create_features(
    values: pd.DataFrame,
    base_timing: str,
    sma_window_sizes: list[int],
    sma_window_size_center: int,
    sigma_window_sizes: list[int],
    sma_frac_ndigits: int,
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    features_rel = values.copy()

    for window_size in sma_window_sizes:
        features_rel[f"sma{window_size}"] = calc_sma(values[base_timing], window_size)

    features_abs = pd.DataFrame()

    for window_size in sigma_window_sizes:
        features_abs[f"sigma{window_size}"] = calc_sigma(
            values[base_timing], window_size
        )

    features_abs[f"sma{sma_window_size_center}_frac"] = calc_fraction(
        features_rel[f"sma{sma_window_size_center}"],
        ndigits=sma_frac_ndigits,
    )

    features_abs["hour"] = values.index.hour.astype(np.int64)
    features_abs["dow"] = values.index.day_of_week.astype(np.int64)

    return {"rel": features_rel, "abs": features_abs}


def calc_target(close_1min: pd.Series, alpha: float) -> pd.Series:
    future_values = (
        close_1min[::-1].ewm(alpha=alpha, adjust=True).mean().astype(np.float32)
    )
    return future_values[::-1].shift(-1)


def calc_gain(close_1min: pd.Series, alpha: float) -> pd.Series:
    targets = calc_target(close_1min, alpha)
    return targets - close_1min


def calc_available_index(
    features: dict[str, dict[str, pd.DataFrame]],
    gain: pd.Series,
    lag_max: int,
    start_hour: int,
    end_hour: int,
) -> pd.DatetimeIndex:
    def get_first_index(df: pd.DataFrame):
        notnan_idxs = (df.notna().all(axis=1)).values.nonzero()[0]
        first_idx = notnan_idxs[0] + lag_max - 1
        return df.index[first_idx]

    # features は前を見るため最初の方のデータは NaN になる
    first_index = pd.Timestamp("1900-1-1 00:00:00")
    for timeframe in features:
        first_index = max(
            first_index,
            get_first_index(features[timeframe]["rel"]),
            get_first_index(features[timeframe]["abs"]),
        )

    # gain は後を見るため最後のデータは NaN になる。
    last_index = gain.index[gain.notna()][-1]

    base_index = features["1min"]["rel"].index
    available_mask = (
        (base_index >= first_index)
        & (base_index <= last_index)
        & (base_index.hour >= start_hour)
        & (base_index.hour < end_hour)
        & ~((base_index.month == 12) & (base_index.day == 25))
    )
    return base_index[available_mask]


class DataLoader:
    def __init__(
        self,
        base_index: pd.DatetimeIndex,
        features: dict[str, dict[str, pd.DataFrame]],
        gain: Optional[pd.DataFrame],
        lag_max: int,
        sma_window_size_center: int,
        batch_size: int,
        shuffle: bool = False,
        rel_reverse: bool = False,
        rel_shuffle: bool = False,
    ):
        self.base_index = base_index
        self.features = features
        self.gain = gain
        self.lag_max = lag_max
        self.sma_window_size_center = sma_window_size_center
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rel_reverse = rel_reverse
        self.rel_shuffle = rel_shuffle

    def __iter__(
        self,
    ) -> Generator[tuple[dict[str, dict[str, np.ndarray]], np.ndarray], None, None]:
        index = self.base_index
        if self.shuffle:
            index = index[np.random.permutation(len(index))]

        timeframes = list(self.features.keys())

        # それぞれの freq に対応する idx を予め計算しておく
        index_dict = {}
        for timeframe in timeframes:
            index_timeframe = index.floor(pd.Timedelta(timeframe))
            index_dict[timeframe] = self.features[timeframe]["rel"].index.get_indexer(
                index_timeframe
            )

        row_count = 0
        while row_count < len(index):
            idx_batch_dict = {
                timeframe: index_dict[timeframe][
                    row_count : row_count + self.batch_size
                ]
                for timeframe in timeframes
            }

            features = {}
            for timeframe in timeframes:
                idx_batch = idx_batch_dict[timeframe]
                idx_expanded = (idx_batch[:, np.newaxis] - np.arange(self.lag_max))[
                    :, ::-1
                ]

                sma = self.features[timeframe]["rel"][
                    f"sma{self.sma_window_size_center}"
                ].values[idx_batch]
                # import pdb; pdb.set_trace()

                features[timeframe] = {}
                for feature_name in self.features[timeframe]["rel"]:
                    value = (
                        self.features[timeframe]["rel"][feature_name]
                        .values[idx_expanded.flatten()]
                        .reshape((len(idx_batch), self.lag_max))
                        - sma[:, np.newaxis]
                    )
                    if self.rel_reverse:
                        value = value * -1
                    if self.rel_shuffle:
                        rand = np.random.choice([1, -1], size=(len(value), 1))
                        value = value * rand.astype(np.float32)

                    features[timeframe][feature_name] = value

                for feature_name in self.features[timeframe]["abs"]:
                    features[timeframe][feature_name] = (
                        self.features[timeframe]["abs"][feature_name]
                        .values[idx_expanded.flatten()]
                        .reshape((len(idx_batch), self.lag_max))
                    )

            if self.gain is None:
                gain = np.zeros(len(idx_batch_dict["1min"]), dtype=np.float32)
            else:
                gain = self.gain.values[idx_batch_dict["1min"]]

            for timeframe in timeframes:
                for feature_name in features[timeframe]:
                    if np.isnan(features[timeframe][feature_name]).any():
                        import pdb

                        pdb.set_trace()
                    assert not np.isnan(
                        features[timeframe][feature_name]
                    ).any(), f"NaN found in {timeframe} {feature_name}"

            assert not np.isnan(gain).any()

            yield features, gain

            row_count += len(idx_batch_dict["1min"])


def get_feature_info(
    loader: DataLoader, batch_num: int = 100
) -> dict[str, dict[str, dict[str, Union[int, float]]]]:
    feature_info = {}
    count = 0
    for batch_idx, (features, _) in enumerate(loader):
        if batch_idx == batch_num:
            break

        for timeframe in features:
            if batch_idx == 0:
                feature_info[timeframe] = {}

            for feature_name in features[timeframe]:
                values = features[timeframe][feature_name]
                if batch_idx == 0:
                    feature_info[timeframe][feature_name] = {
                        "dtype": values.dtype,
                        "mean": 0,
                        "var": 0,
                        "cardinality": 0,
                    }

                if values.dtype == np.float32:
                    count_add = values.size
                    mean_add = np.mean(values)
                    var_add = np.var(values)

                    mean = feature_info[timeframe][feature_name]["mean"]
                    var = feature_info[timeframe][feature_name]["var"]
                    mean_new = (mean * count + mean_add * count_add) / (
                        count + count_add
                    )
                    var_new = (var * count + var_add * count_add) / (
                        count + count_add
                    ) + ((mean - mean_add) ** 2) * count * count_add / (
                        (count + count_add) ** 2
                    )
                    feature_info[timeframe][feature_name]["mean"] = float(mean_new)
                    feature_info[timeframe][feature_name]["var"] = float(var_new)
                elif values.dtype == np.int64:
                    cardinality_old = feature_info[timeframe][feature_name][
                        "cardinality"
                    ]
                    cardinality_new = max(cardinality_old, values.max())
                    feature_info[timeframe][feature_name][
                        "cardinality"
                    ] = cardinality_new
                else:
                    raise ValueError(
                        f"Data type of {timeframe} {feature_name} is not supported: "
                        f"{values.dtype}"
                    )

            count += values.size

    return feature_info
