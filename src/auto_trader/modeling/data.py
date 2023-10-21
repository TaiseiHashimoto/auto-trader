import os
from typing import Generator, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from auto_trader.common import utils

Timeframe = str
FeatureType = Literal["rel", "abs"]
FeatureName = str
FeatureValue = NDArray[Union[np.float32, np.int64]]


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


def merge_bid_ask(df: pd.DataFrame) -> pd.DataFrame:
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
    values_base: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    if timeframe == "1min":
        return values_base
    else:
        values_resampled = pd.DataFrame(
            {
                "open": values_base["open"].resample(timeframe).first(),
                "high": values_base["high"].resample(timeframe).max(),
                "low": values_base["low"].resample(timeframe).min(),
                "close": values_base["close"].resample(timeframe).last(),
            }
        )
        # 削除していた flat 期間が NaN になるので削除
        return values_resampled.dropna()


def calc_sma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).mean().astype(np.float32)


def calc_sigma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).std(ddof=0).astype(np.float32)


def calc_fraction(values: "pd.Series[float]", unit: int) -> "pd.Series[float]":
    return (values % unit).astype(np.float32)


def create_features(
    values: pd.DataFrame,
    base_timing: str,
    sma_window_sizes: list[int],
    sma_window_size_center: int,
    sigma_window_sizes: list[int],
    sma_frac_unit: int,
) -> dict[FeatureType, pd.DataFrame]:
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
        unit=sma_frac_unit,
    )

    datetime_index = cast(pd.DatetimeIndex, values.index)
    features_abs["hour"] = datetime_index.hour.astype(np.int64)
    features_abs["dow"] = datetime_index.day_of_week.astype(np.int64)

    return {"rel": features_rel, "abs": features_abs}


def calc_lift(value_base: "pd.Series[float]", alpha: float) -> "pd.Series[float]":
    future_value = (
        value_base.iloc[::-1]
        .ewm(alpha=alpha, adjust=True)
        .mean()
        .iloc[::-1]
        .shift(-1)
        .astype(np.float32)
    )
    return future_value - value_base


def calc_available_index(
    features: dict[Timeframe, dict[FeatureType, pd.DataFrame]],
    lift: "pd.Series[float]",
    hist_len: int,
    start_hour: int,
    end_hour: int,
) -> pd.DatetimeIndex:
    def get_first_index(df: pd.DataFrame) -> pd.Timestamp:
        notnan_idxs = cast(
            NDArray[np.bool_], (df.notna().all(axis=1)).values
        ).nonzero()[0]
        first_idx = notnan_idxs[0] + hist_len - 1
        return cast(pd.Timestamp, df.index[first_idx])

    # features は前を見るため最初の方のデータは NaN になる
    first_index = pd.Timestamp("1900-1-1 00:00:00")
    for timeframe in features:
        first_index = max(
            first_index,
            get_first_index(features[timeframe]["rel"]),
            get_first_index(features[timeframe]["abs"]),
        )

    # lift は後を見るため最後のデータは NaN になる。
    last_index = lift.index[lift.notna()][-1]

    base_index = cast(pd.DatetimeIndex, features["1min"]["rel"].index)
    available_mask = (
        (base_index >= first_index)
        & (base_index <= last_index)
        & (base_index.hour >= start_hour)
        & (base_index.hour < end_hour)
        # クリスマスは一部時間がデータに含まれるが、傾向が特殊なので除外
        & ~((base_index.month == 12) & (base_index.day == 25))
    )
    return cast(pd.DatetimeIndex, base_index[available_mask])


class DataLoader:
    def __init__(
        self,
        base_index: pd.DatetimeIndex,
        features: dict[str, dict[FeatureType, pd.DataFrame]],
        lift: Optional[pd.DataFrame],
        hist_len: int,
        sma_window_size_center: int,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self.base_index = base_index
        self.features = features
        self.lift = lift
        self.hist_len = hist_len
        self.sma_window_size_center = sma_window_size_center
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(
        self,
    ) -> Generator[
        tuple[dict[Timeframe, dict[FeatureName, FeatureValue]], NDArray[np.float32]],
        None,
        None,
    ]:
        index = self.base_index
        if self.shuffle:
            index = index[np.random.permutation(len(index))]

        timeframes = list(self.features.keys())

        # それぞれの freq に対応する idx を予め計算しておく
        index_dict = {}
        for timeframe in timeframes:
            index_timeframe = index.floor(freq=timeframe)
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

            features: dict[Timeframe, dict[FeatureName, FeatureValue]] = {}
            for timeframe in timeframes:
                idx_batch = idx_batch_dict[timeframe]
                idx_expanded = (idx_batch[:, np.newaxis] - np.arange(self.hist_len))[
                    :, ::-1
                ]

                sma = self.features[timeframe]["rel"][
                    f"sma{self.sma_window_size_center}"
                ].values[idx_batch]

                features[timeframe] = {}
                for feature_type in self.features[timeframe]:
                    for feature_name in self.features[timeframe][feature_type]:
                        value = (
                            self.features[timeframe][feature_type][feature_name]
                            .values[idx_expanded.flatten()]
                            .reshape((len(idx_batch), self.hist_len))
                        )
                        if feature_type == "rel":
                            value -= sma[:, np.newaxis]

                        assert feature_name not in features[timeframe]
                        features[timeframe][feature_name] = value

            if self.lift is None:
                lift = np.zeros(len(idx_batch_dict["1min"]), dtype=np.float32)
            else:
                lift = self.lift.values[idx_batch_dict["1min"]]

            for timeframe in timeframes:
                for feature_name in features[timeframe]:
                    assert not np.isnan(
                        features[timeframe][feature_name]
                    ).any(), f"NaN found in {timeframe} {feature_name}"

            assert not np.isnan(lift).any()

            yield features, lift

            row_count += len(idx_batch_dict["1min"])


class FeatureInfo:
    def __init__(self, dtype: DTypeLike):
        self._dtype = dtype
        if dtype == np.float32:
            self._count = 0
            self._mean = 0.0
            self._var = 0.0
        elif dtype == np.int64:
            self._max = 0
        else:
            raise ValueError(f"Data type {dtype} is not supported")

    def update(self, values: FeatureValue) -> None:
        if self._dtype == np.float32:
            count_add = values.size
            mean_add = np.mean(values)
            var_add = np.var(values)
            count_new = self._count + count_add
            mean_new = (self._mean * self._count + mean_add * count_add) / count_new
            var_new = (self._var * self._count + var_add * count_add) / count_new + (
                (self._mean - mean_add) ** 2
            ) * self._count * count_add / count_new**2
            self._count = count_new
            self._mean = float(mean_new)
            self._var = float(var_new)
        else:
            self._max = max(self._max, values.max())

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def mean(self) -> float:
        assert self.dtype == np.float32
        return self._mean

    @property
    def var(self) -> float:
        assert self.dtype == np.float32
        return self._var

    @property
    def max(self) -> int:
        assert self.dtype == np.int64
        return self._max


def get_feature_info(
    loader: DataLoader, batch_num: int = 100
) -> dict[Timeframe, dict[FeatureName, FeatureInfo]]:
    feature_info: dict[Timeframe, dict[FeatureName, FeatureInfo]] = {}
    for batch_idx, (features, _) in enumerate(loader):
        if batch_idx == batch_num:
            break

        for timeframe in features:
            if batch_idx == 0:
                feature_info[timeframe] = {}

            for feature_name in features[timeframe]:
                values = features[timeframe][feature_name]
                if batch_idx == 0:
                    feature_info[timeframe][feature_name] = FeatureInfo(values.dtype)

                feature_info[timeframe][feature_name].update(values)

    return feature_info
