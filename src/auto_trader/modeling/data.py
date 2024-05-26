import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from auto_trader.common import utils

Symbol = str
FeatureName = str
FeatureValue = Union[NDArray[np.float32], NDArray[np.int64]]


def read_cleansed_data(
    cleansed_data_dir: Path,
    symbol: Symbol,
    yyyymm_begin: int,
    yyyymm_end: int,
) -> pd.DataFrame:
    """
    前処理したデータを範囲指定して読み込む
    """

    yyyymm = yyyymm_begin
    df_data = []
    while yyyymm <= yyyymm_end:
        df_data.append(
            pd.read_parquet(cleansed_data_dir / symbol / f"{yyyymm}.parquet")
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


def calc_sma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).mean().astype(np.float32)


def calc_moving_max(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).max().astype(np.float32)


def calc_moving_min(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).min().astype(np.float32)


def calc_sigma(s: "pd.Series[float]", window_size: int) -> "pd.Series[float]":
    return s.rolling(window_size).std(ddof=0).astype(np.float32)


def calc_fraction(rates: "pd.Series[float]", unit: int) -> "pd.Series[float]":
    return (rates % unit).astype(np.float32)


def is_relative_feature(name: str) -> bool:
    return re.fullmatch(r"(sma|moving_(max|min))\d+", name) is not None


def create_features(
    rates: pd.DataFrame,
    base_timing: str,
    window_sizes: list[int],
    window_size_center: int,
    use_sma_frac: bool,
    sma_frac_unit: int,
    use_hour: bool,
    use_dow: bool,
    center: bool = True,
) -> pd.DataFrame:
    features = rates.copy()

    sma_center = calc_sma(features[base_timing], window_size_center)
    abs_feature_names = set()
    for window_size in window_sizes:
        features[f"sma{window_size}"] = calc_sma(features[base_timing], window_size)
        features[f"moving_max{window_size}"] = calc_moving_max(
            features[base_timing], window_size
        )
        features[f"moving_min{window_size}"] = calc_moving_min(
            features[base_timing], window_size
        )
        features[f"sigma{window_size}"] = calc_sigma(features[base_timing], window_size)
        abs_feature_names.add(f"sigma{window_size}")

    if use_sma_frac:
        features[f"sma_frac{window_size_center}"] = calc_fraction(
            sma_center,
            unit=sma_frac_unit,
        )
        abs_feature_names.add(f"sma_frac{window_size_center}")

    datetime_index = cast(pd.DatetimeIndex, features.index)
    if use_hour:
        features["hour"] = datetime_index.hour.astype(np.int64)
        abs_feature_names.add("hour")
    if use_dow:
        features["dow"] = datetime_index.day_of_week.astype(np.int64)
        abs_feature_names.add("dow")

    if center:
        for feature_name in features.columns:
            if is_relative_feature(feature_name):
                features[feature_name] -= sma_center

    return features


@dataclass
class ContinuousFeatureStats:
    mean: float
    std: float

    def __str__(self) -> str:
        return f"{self.mean:.6f} +- {self.std:.6f}"


@dataclass
class CategoricalFeatureStats:
    vocab_counts: dict[int, int]

    def __post_init__(self) -> None:
        vocab = sorted(self.vocab_counts.keys())
        if vocab != list(range(len(vocab))):
            raise ValueError(
                "Categorical features must be contiguous integers "
                "starting from zero."
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_counts)

    def __str__(self) -> str:
        return str(self.vocab_counts)


FeatureStats = Union[ContinuousFeatureStats, CategoricalFeatureStats]


def get_feature_stats(features: pd.DataFrame) -> dict[FeatureName, FeatureStats]:
    stats: dict[FeatureName, FeatureStats] = {}
    for col in features.columns:
        val = features[col]
        if val.dtype == np.float32:
            stats[col] = ContinuousFeatureStats(val.mean(), val.std(ddof=0))
        elif val.dtype == np.int64:
            stats[col] = CategoricalFeatureStats(
                val.value_counts().sort_index().to_dict()
            )

    return stats


def normalize_features(
    features: pd.DataFrame, feature_stats: dict[FeatureName, FeatureStats]
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=features.index)
    for col in features.columns:
        val = features[col]
        stats = feature_stats[col]
        if isinstance(stats, ContinuousFeatureStats):
            normalized[col] = (val - stats.mean) / (stats.std + 1e-6)
        elif isinstance(stats, CategoricalFeatureStats):
            # OOV は vocab_size に変換する
            oov_mask = (val < 0) | (val >= stats.vocab_size)
            normalized[col] = val.mask(oov_mask, stats.vocab_size)

    return normalized


def calc_lift(
    rate: "pd.Series[float]", future_begin: int, future_end: int
) -> "pd.Series[float]":
    future_value = calc_sma(rate, future_end - future_begin).shift(-(future_end - 1))
    return future_value - rate


def create_label(lift: "pd.Series[float]", bin_boundary: float) -> "pd.Series[float]":
    return pd.cut(
        lift,
        bins=[lift.min(), -bin_boundary, bin_boundary, lift.max()],
        labels=False,
        include_lowest=True,
    ).astype(np.float32)


def calc_available_index(
    features: pd.DataFrame, label: "pd.Series[float]", hist_len: int
) -> pd.DatetimeIndex:
    available_index_features = features.index[features.notna().all(axis=1)]
    available_index_label = label.index[label.notna()]
    first_datetime = max(
        available_index_features[hist_len - 1], available_index_label[hist_len - 1]
    )
    last_datetime = min(available_index_features[-1], available_index_label[-1])

    base_index = cast(pd.DatetimeIndex, features.index)
    available_mask = (
        (base_index >= first_datetime)
        & (base_index <= last_datetime)
        # クリスマスは一部時間がデータに含まれるが、傾向が特殊なので除外
        & ~((base_index.month == 12) & (base_index.day == 25))
    )
    return cast(pd.DatetimeIndex, base_index[available_mask])


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


class SequentialLoader:
    def __init__(
        self,
        available_index: pd.DatetimeIndex,
        features: pd.DataFrame,
        label: Optional["pd.Series[float]"],
        hist_len: int,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        if label is not None:
            assert (features.index == label.index).all()

        self.available_index = available_index
        self.features = features
        self.label = label
        self.hist_len = hist_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.available_index) + self.batch_size - 1) // self.batch_size

    @property
    def size(self) -> int:
        return len(self.available_index)

    def __iter__(self) -> Generator[
        tuple[dict[FeatureName, FeatureValue], NDArray[np.float32]],
        None,
        None,
    ]:
        index = self.available_index
        if self.shuffle:
            index = index[np.random.permutation(len(index))]

        # index に対応する idx を予め計算しておく
        idx = self.features.index.get_indexer(index)  # type: ignore

        row_count = 0
        while row_count < len(index):
            idx_batch = idx[row_count : row_count + self.batch_size]
            # 時刻の昇順に展開 (B, T)
            idx_expanded = cast(
                NDArray[np.int64],
                idx_batch[:, np.newaxis] - np.arange(self.hist_len)[::-1],
            )
            features = {
                col: (
                    self.features[col]
                    .to_numpy()[idx_expanded.flatten()]
                    .reshape((len(idx_batch), self.hist_len))
                )
                for col in self.features.columns
            }

            if self.label is not None:
                label = self.label.values[idx_batch].astype(np.int64)
            else:
                label = np.zeros(len(idx_batch), dtype=np.float64)

            # for feature_name in features:
            #     assert not np.isnan(
            #         features[feature_name]
            #     ).any(), f"NaN found in {feature_name}"

            yield features, label

            row_count += len(idx_batch)
