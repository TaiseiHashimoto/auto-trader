import os
from typing import Generator, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from auto_trader.common import utils

Symbol = str
Timeframe = str
FeatureType = Literal["rel", "abs"]
FeatureName = str
FeatureValue = NDArray[Union[np.float32, np.int64]]


def read_cleansed_data(
    symbol: Symbol,
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
                os.path.join(cleansed_data_dir, symbol, f"{yyyymm}.parquet")
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
    timeframe: Timeframe,
) -> pd.DataFrame:
    if timeframe == "1min":
        values_resampled = values_base
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
        values_resampled = values_resampled.dropna()

    # 未来の値を見ないようにシフト
    return values_resampled.shift(1)


def calc_sma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).mean().astype(np.float32)


def calc_moving_max(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).max().astype(np.float32)


def calc_moving_min(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).min().astype(np.float32)


def calc_sigma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).std(ddof=0).astype(np.float32)


def calc_fraction(values: "pd.Series[float]", unit: int) -> "pd.Series[float]":
    return (values % unit).astype(np.float32)


def create_features(
    values: pd.DataFrame,
    base_timing: str,
    moving_window_sizes: list[int],
    moving_window_size_center: int,
    use_sma_frac: bool,
    sma_frac_unit: int,
    use_hour: bool,
    use_dow: bool,
) -> dict[FeatureType, pd.DataFrame]:
    features_rel = values.copy()
    features_abs = pd.DataFrame()

    for window_size in moving_window_sizes:
        features_rel[f"sma{window_size}"] = calc_sma(values[base_timing], window_size)
        features_rel[f"moving_max{window_size}"] = calc_moving_max(
            values[base_timing], window_size
        )
        features_rel[f"moving_min{window_size}"] = calc_moving_min(
            values[base_timing], window_size
        )
        features_abs[f"sigma{window_size}"] = calc_sigma(
            values[base_timing], window_size
        )

    if use_sma_frac:
        features_abs[f"sma{moving_window_size_center}_frac"] = calc_fraction(
            features_rel[f"sma{moving_window_size_center}"],
            unit=sma_frac_unit,
        )

    datetime_index = cast(pd.DatetimeIndex, values.index)
    if use_hour:
        features_abs["hour"] = datetime_index.hour.astype(np.int64)
    if use_dow:
        features_abs["dow"] = datetime_index.day_of_week.astype(np.int64)

    return {"rel": features_rel, "abs": features_abs}


def calc_lift(value_close: "pd.Series[float]", alpha: float) -> "pd.Series[float]":
    future_value = (
        value_close.iloc[::-1]
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .iloc[::-1]
        .astype(np.float32)
    )
    return future_value - value_close.shift(1)


def calc_available_index(
    features: dict[Timeframe, dict[FeatureType, pd.DataFrame]], hist_len: int
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

    base_index = cast(pd.DatetimeIndex, features["1min"]["rel"].index)
    available_mask = (
        (base_index >= first_index)
        # クリスマスは一部時間がデータに含まれるが、傾向が特殊なので除外
        & ~((base_index.month == 12) & (base_index.day == 25))
    )
    return base_index[available_mask]


def split_block_idxs(
    size: int, block_size: int, valid_ratio: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    block_start_idxs = np.arange(0, size, block_size)
    block_start_idxs_valid = np.random.choice(
        block_start_idxs, size=int(len(block_start_idxs) * valid_ratio), replace=False
    )
    idxs_valid_content: list[int] = []
    for idx in block_start_idxs_valid:
        idxs_valid_content.extend(range(idx, min(idx + block_size, size)))

    idxs_valid = np.random.permutation(idxs_valid_content).astype(np.int64)
    idxs_train = np.random.permutation(
        list(set(range(size)) - set(idxs_valid_content))
    ).astype(np.int64)
    return idxs_train, idxs_valid


class RawLoader:
    def __init__(
        self,
        base_index: pd.DatetimeIndex,
        features: dict[Timeframe, dict[FeatureType, pd.DataFrame]],
        lift: Optional["pd.Series[float]"],
        hist_len: int,
        moving_window_size_center: int,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self.base_index = base_index
        self.features = features
        self.lift = lift
        self.hist_len = hist_len
        self.moving_window_size_center = moving_window_size_center
        self.batch_size = batch_size
        self.shuffle = shuffle

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.base_index) + self.batch_size - 1) // self.batch_size

    @property
    def size(self) -> int:
        return len(self.base_index)

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            dict[Timeframe, dict[FeatureName, FeatureValue]],
            NDArray[np.float32],
        ],
        None,
        None,
    ]:
        index = self.base_index
        if self.shuffle:
            index = index[np.random.permutation(len(index))]

        timeframes = list(self.features.keys())

        # それぞれの timeframe に対応する idx を予め計算しておく
        idx_dict = {}
        for timeframe in timeframes:
            index_timeframe = index.floor(freq=timeframe)
            idx_dict[timeframe] = self.features[timeframe]["rel"].index.get_indexer(
                index_timeframe
            )  # type: ignore

        row_count = 0
        while row_count < len(index):
            idx_batch_dict = {
                timeframe: idx_dict[timeframe][row_count : row_count + self.batch_size]
                for timeframe in timeframes
            }

            features: dict[Timeframe, dict[FeatureName, FeatureValue]] = {}
            for timeframe in timeframes:
                idx_batch = idx_batch_dict[timeframe]
                # 時刻の昇順に展開
                idx_expanded = idx_batch[:, np.newaxis] - np.arange(self.hist_len)[::-1]

                sma = self.features[timeframe]["rel"][
                    f"sma{self.moving_window_size_center}"
                ].values[idx_batch]

                features[timeframe] = {}
                for feature_type in self.features[timeframe]:
                    for feature_name in self.features[timeframe][feature_type].columns:
                        value = (
                            self.features[timeframe][feature_type][feature_name]
                            .values[idx_expanded.flatten()]
                            .reshape((len(idx_batch), self.hist_len))
                        )
                        if feature_type == "rel":
                            value -= sma[:, np.newaxis]

                        assert feature_name not in features[timeframe]
                        features[timeframe][feature_name] = value

            if self.lift is not None:
                lift = self.lift.values[idx_batch_dict["1min"]]
            else:
                lift = np.zeros(len(idx_batch_dict["1min"]), dtype=np.float32)

            # for timeframe in timeframes:
            #     for feature_name in features[timeframe]:
            #         assert not np.isnan(
            #             features[timeframe][feature_name]
            #         ).any(), f"NaN found in {timeframe} {feature_name}"

            # assert not np.isnan(lift).any()

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

    def __str__(self) -> str:
        if self._dtype == np.float32:
            return f"{self._mean:.6f} +- {self._var ** 0.5:.6f}"
        else:
            return f"<= {self._max}"


def get_feature_info(
    loader: RawLoader, batch_num: int = 100
) -> tuple[dict[Timeframe, dict[FeatureName, FeatureInfo]], FeatureInfo]:
    feature_info: dict[Timeframe, dict[FeatureName, FeatureInfo]] = {}
    lift_info = FeatureInfo(np.float32)
    for batch_idx, (features, lift) in enumerate(loader):
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

        lift_info.update(lift)

    return feature_info, lift_info


class NormalizedLoader:
    def __init__(
        self,
        loader: RawLoader,
        feature_info: dict[Timeframe, dict[FeatureName, FeatureInfo]],
    ):
        self._loader = loader
        self._feature_info = feature_info

    def set_batch_size(self, batch_size: int) -> None:
        self._loader.set_batch_size(batch_size)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def size(self) -> int:
        return self._loader.size

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            dict[Timeframe, dict[FeatureName, FeatureValue]],
            NDArray[np.float32],
        ],
        None,
        None,
    ]:
        for features, lift in self._loader:
            features_norm: dict[Timeframe, dict[FeatureName, FeatureValue]] = {}
            for timeframe in features:
                features_norm[timeframe] = {}
                for feature_name in features[timeframe]:
                    feature_value = features[timeframe][feature_name]
                    feature_info = self._feature_info[timeframe][feature_name]
                    if feature_info.dtype == np.float32:
                        feature_value_norm = (feature_value - feature_info.mean) / (
                            feature_info.var**0.5 + 1e-6
                        )
                    elif feature_info.dtype == np.int64:
                        feature_value_norm = np.clip(feature_value, 0, feature_info.max)
                    features_norm[timeframe][feature_name] = feature_value_norm

            yield features_norm, lift


class CombinedLoader:
    def __init__(
        self, loaders: dict[Symbol, NormalizedLoader], key_map: dict[Symbol, int]
    ):
        self._loaders = {key: loader for key, loader in loaders.items()}
        self._key_map = key_map

    def __len__(self) -> int:
        return max(len(loader) for loader in self._loaders.values())

    @property
    def size(self) -> int:
        return sum(loader.size for loader in self._loaders.values())

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            NDArray[np.int64],
            dict[Timeframe, dict[FeatureName, FeatureValue]],
            NDArray[np.float32],
        ],
        None,
        None,
    ]:
        iterators = {key: iter(loader) for key, loader in self._loaders.items()}
        while True:
            key_idx_list = []
            features_list: dict[Timeframe, dict[FeatureName, list[FeatureValue]]] = {}
            lift_list = []
            for key, iterator in iterators.items():
                batch = next(iterator, None)
                if batch is None:
                    continue

                features, lift = batch

                key_idx_list.append([self._key_map[key]] * len(lift))

                for timeframe in features:
                    if timeframe not in features_list:
                        features_list[timeframe] = {}

                    for feature_name in features[timeframe]:
                        if feature_name not in features_list[timeframe]:
                            features_list[timeframe][feature_name] = []

                        features_list[timeframe][feature_name].append(
                            features[timeframe][feature_name]
                        )

                lift_list.append(lift)

            if len(key_idx_list) == 0:
                return

            key_idx_combined = np.concatenate(key_idx_list, axis=0)
            batch_combined = {
                timeframe: {
                    feature_name: np.concatenate(
                        features_list[timeframe][feature_name], axis=0
                    )
                    for feature_name in features_list[timeframe]
                }
                for timeframe in features_list
            }
            lift_combined = np.concatenate(lift_list, axis=0)
            yield key_idx_combined, batch_combined, lift_combined
