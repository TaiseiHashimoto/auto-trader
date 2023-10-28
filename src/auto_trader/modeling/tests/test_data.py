from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pytest import approx

from auto_trader.modeling import data


def test_read_cleansed_data(tmp_path: Path) -> None:
    df_202301 = pd.DataFrame({"x": [1, 2, 3]})
    df_202302 = pd.DataFrame({"x": [4, 5]})
    df_202301.to_parquet(tmp_path / "usdjpy-202301.parquet")
    df_202302.to_parquet(tmp_path / "usdjpy-202302.parquet")

    df_actual = data.read_cleansed_data(
        symbol="usdjpy",
        yyyymm_begin=202301,
        yyyymm_end=202302,
        cleansed_data_dir=str(tmp_path),
    )

    df_expected = pd.concat([df_202301, df_202302], axis=0)
    pd.testing.assert_frame_equal(df_actual, df_expected)


def test_merge_bid_ask() -> None:
    index = pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:03:00", freq="1min")
    df = pd.DataFrame(
        {
            "bid_open": [0, 1, 2, 3],
            "ask_open": [2, 3, 4, 5],
            "bid_high": [0, 2, 4, 6],
            "ask_high": [2, 4, 6, 8],
            "bid_low": [0, 3, 6, 9],
            "ask_low": [2, 5, 8, 11],
            "bid_close": [0, 4, 8, 12],
            "ask_close": [2, 6, 10, 14],
        },
        index=index,
    )
    actual_result = data.merge_bid_ask(df)
    expected_result = pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [1, 3, 5, 7],
            "low": [1, 4, 7, 10],
            "close": [1, 5, 9, 13],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(actual_result, expected_result, check_dtype=False)


def test_resample() -> None:
    df_base = pd.DataFrame(
        {
            "open": [0, 1, 2, 3, 4, 5, 6, 7],
            "high": [0, 10, 20, 30, 40, 50, 60, 70],
            "low": [0, -10, -20, -30, -40, -50, -60, -70],
            "close": [0, -1, -2, -3, -4, -5, -6, -7],
        },
        index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"),
    )
    pd.testing.assert_frame_equal(
        data.resample(df_base, timeframe="1min"),
        pd.DataFrame(
            {
                "open": [np.nan, 0, 1, 2, 3, 4, 5, 6],
                "high": [np.nan, 0, 10, 20, 30, 40, 50, 60],
                "low": [np.nan, 0, -10, -20, -30, -40, -50, -60],
                "close": [np.nan, 0, -1, -2, -3, -4, -5, -6],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"
            ),
        ),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        data.resample(df_base, timeframe="2min"),
        pd.DataFrame(
            {
                "open": [np.nan, 0, 2, 4],
                "high": [np.nan, 10, 30, 50],
                "low": [np.nan, -10, -30, -50],
                "close": [np.nan, -1, -3, -5],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="2min"
            ),
        ),
        check_dtype=False,
    )


def test_calc_sma() -> None:
    s = pd.Series([0.0, 4.0, 2.0, 3.0, 6.0, 4.0, 6.0, 9.0], dtype=np.float32)
    actual_result = data.calc_sma(s, window_size=4)
    expected_result = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            (0 + 4 + 2 + 3) / 4,
            (4 + 2 + 3 + 6) / 4,
            (2 + 3 + 6 + 4) / 4,
            (3 + 6 + 4 + 6) / 4,
            (6 + 4 + 6 + 9) / 4,
        ],
        dtype=np.float32,
    )
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_calc_sigma() -> None:
    s = pd.Series([0.0, 4.0, 2.0, 3.0, 6.0, 4.0, 6.0, 9.0], dtype=np.float32)
    actual_result = data.calc_sigma(s, window_size=4)
    expected_result = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            ((0**2 + 4**2 + 2**2 + 3**2) / 4 - ((0 + 4 + 2 + 3) / 4) ** 2)
            ** 0.5,
            ((4**2 + 2**2 + 3**2 + 6**2) / 4 - ((4 + 2 + 3 + 6) / 4) ** 2)
            ** 0.5,
            ((2**2 + 3**2 + 6**2 + 4**2) / 4 - ((2 + 3 + 6 + 4) / 4) ** 2)
            ** 0.5,
            ((3**2 + 6**2 + 4**2 + 6**2) / 4 - ((3 + 6 + 4 + 6) / 4) ** 2)
            ** 0.5,
            ((6**2 + 4**2 + 6**2 + 9**2) / 4 - ((6 + 4 + 6 + 9) / 4) ** 2)
            ** 0.5,
        ],
        dtype=np.float32,
    )
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_calc_fraction() -> None:
    values = pd.Series([12345.67, 9876.54])

    actual = data.calc_fraction(values, unit=100)
    expected = pd.Series([45.67, 76.54], dtype=np.float32)
    pd.testing.assert_series_equal(expected, actual)

    actual = data.calc_fraction(values, unit=1000)
    expected = pd.Series([345.67, 876.54], dtype=np.float32)
    pd.testing.assert_series_equal(expected, actual)


def test_create_features() -> None:
    values = pd.DataFrame(
        {
            "open": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "high": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "low": [0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110],
            "close": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11],
        },
        index=pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:11:00", freq="1min"),
    ).astype(np.float32)
    actual = data.create_features(
        values=values,
        base_timing="close",
        sma_window_sizes=[5, 10],
        sma_window_size_center=5,
        sigma_window_sizes=[6, 12],
        sma_frac_unit=100,
    )
    expected = {
        "rel": pd.DataFrame(
            {
                "open": values["open"],
                "high": values["high"],
                "low": values["low"],
                "close": values["close"],
                "sma5": data.calc_sma(values["close"], window_size=5),
                "sma10": data.calc_sma(values["close"], window_size=10),
            }
        ),
        "abs": pd.DataFrame(
            {
                "sigma6": data.calc_sigma(values["close"], window_size=6),
                "sigma12": data.calc_sigma(values["close"], window_size=12),
                "sma5_frac": data.calc_fraction(
                    data.calc_sma(values["close"], window_size=5), unit=100
                ),
                "hour": np.full(12, 0),
                "dow": np.full(12, date(2023, 1, 1).weekday()),
            }
        ),
    }
    pd.testing.assert_frame_equal(actual["rel"], expected["rel"])
    pd.testing.assert_frame_equal(actual["abs"], expected["abs"])


def test_calc_lift() -> None:
    value_close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    pd.testing.assert_series_equal(
        data.calc_lift(value_close, alpha=0.1),
        pd.Series(
            [
                np.nan,
                (2 * 0.1 + 3 * 0.1 * 0.9 + 4 * 0.1 * 0.9**2 + 5 * 0.9**3) - 1,
                (3 * 0.1 + 4 * 0.1 * 0.9 + 5 * 0.9**2) - 2,
                (4 * 0.1 + 5 * 0.9**1) - 3,
                5 * 0.9**0 - 4,
            ],
            dtype=np.float32,
        ),
    )


def test_calc_losscut_idxs() -> None:
    value_close = np.array(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            102.0,
            101.0,
            100.0,
            99.0,
            98.0,
            97.0,
            98.0,
            99.0,
            100.0,
        ]
    )
    np.testing.assert_array_equal(
        data.calc_losscut_idxs(value_close, thresh_losscut=2.0),
        np.array([8, 7, 6, 5, 6, 7, 8, 9, 12, 12, 12, 12, 12]),
    )
    np.testing.assert_array_equal(
        data.calc_losscut_idxs(-value_close, thresh_losscut=2.0),
        np.array([2, 3, 12, 12, 12, 12, 12, 12, 12, 11, 12, 12, 12]),
    )


def test_calc_gains() -> None:
    value_close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    gain_long, gain_short = data.calc_gains(value_close, alpha=0.1, thresh_losscut=2.0)
    pd.testing.assert_series_equal(
        gain_long,
        pd.Series(
            [
                np.nan,
                (2 * 0.1 + 3 * 0.1 * 0.9 + 4 * 0.1 * 0.9**2 + 5 * 0.9**3) - 1,
                (3 * 0.1 + 4 * 0.1 * 0.9 + 5 * 0.9**2) - 2,
                (4 * 0.1 + 5 * 0.9**1) - 3,
                5 * 0.9**0 - 4,
            ],
            dtype=np.float32,
        ),
    )
    pd.testing.assert_series_equal(
        gain_short,
        -pd.Series(
            [
                np.nan,
                (2 * 0.1 + 3 * 0.9) - 1,
                (3 * 0.1 + 4 * 0.9) - 2,
                (4 * 0.1 + 5 * 0.9) - 3,
                5 - 4,
            ],
            dtype=np.float32,
        ),
    )


def test_calc_available_index_nan() -> None:
    features = cast(
        dict[data.Timeframe, dict[data.FeatureType, pd.DataFrame]],
        {
            "1min": {
                "rel": pd.DataFrame(
                    {"x": [0] * 20},
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:19", freq="1min"
                    ),
                ),
                "abs": pd.DataFrame(
                    {"y": [0] * 20},
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:19", freq="1min"
                    ),
                ),
            },
            "5min": {
                "rel": pd.DataFrame(
                    {"x": [0] * 4},
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:19", freq="5min"
                    ),
                ),
                "abs": pd.DataFrame(
                    {"y": [np.nan] + [0] * 3},
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:19", freq="5min"
                    ),
                ),
            },
        },
    )
    actual = data.calc_available_index(
        features=features,
        hist_len=2,
        start_hour=0,
        end_hour=24,
    )
    expected = pd.date_range("2023-1-1 00:10", "2023-1-1 00:19", freq="1min")
    pd.testing.assert_index_equal(actual, expected)


def test_split_block_idxs() -> None:
    idxs_train, idxs_valid = data.split_block_idxs(
        size=10,
        block_size=2,
        valid_ratio=0.2,
    )
    assert len(idxs_train) == 8
    assert len(idxs_valid) == 2
    assert set(idxs_train) | set(idxs_valid) == set(range(10))
    assert abs(idxs_valid[0] - idxs_valid[1]) == 1
    assert (0 in idxs_train and 1 in idxs_train) or (
        0 in idxs_valid and 1 in idxs_valid
    )
    assert (2 in idxs_train and 3 in idxs_train) or (
        2 in idxs_valid and 3 in idxs_valid
    )
    assert (4 in idxs_train and 5 in idxs_train) or (
        4 in idxs_valid and 5 in idxs_valid
    )
    assert (6 in idxs_train and 7 in idxs_train) or (
        6 in idxs_valid and 7 in idxs_valid
    )
    assert (8 in idxs_train and 9 in idxs_train) or (
        8 in idxs_valid and 9 in idxs_valid
    )

    idxs_train, idxs_valid = data.split_block_idxs(
        size=7,
        block_size=3,
        valid_ratio=0.5,
    )
    assert set(idxs_train) | set(idxs_valid) == set(range(7))
    assert (0 in idxs_train and 1 in idxs_train and 2 in idxs_train) or (
        0 in idxs_valid and 1 in idxs_valid and 2 in idxs_valid
    )
    assert (3 in idxs_train and 4 in idxs_train and 5 in idxs_train) or (
        3 in idxs_valid and 4 in idxs_valid and 5 in idxs_valid
    )
    assert (6 in idxs_train) or (6 in idxs_valid)


def test_dataloader() -> None:
    base_index = pd.date_range("2023-1-1 00:02", "2023-1-1 00:05", freq="1min")
    features = cast(
        dict[data.Timeframe, dict[data.FeatureType, pd.DataFrame]],
        {
            "1min": {
                "rel": pd.DataFrame(
                    {
                        "sma5": np.array(
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                        ),
                        "x": np.array(
                            [-0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32
                        ),
                    },
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:05", freq="1min"
                    ),
                ),
                "abs": pd.DataFrame(
                    {
                        "sigma": np.array(
                            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32
                        ),
                        "minute": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
                    },
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:05", freq="1min"
                    ),
                ),
            },
            "2min": {
                "rel": pd.DataFrame(
                    {
                        "sma5": np.array([0.5, 1.5, 2.5], dtype=np.float32),
                        "x": np.array([-0.5, -1.5, -2.5], dtype=np.float32),
                    },
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:05", freq="2min"
                    ),
                ),
                "abs": pd.DataFrame(
                    {
                        "sigma": np.array([10.5, 11.5, 12.5], dtype=np.float32),
                        "minute": np.array([0, 2, 4], dtype=np.int64),
                    },
                    index=pd.date_range(
                        "2023-1-1 00:00", "2023-1-1 00:05", freq="2min"
                    ),
                ),
            },
        },
    )
    gain_long = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        index=pd.date_range("2023-1-1 00:00", "2023-1-1 00:05", freq="1min"),
        dtype=np.float32,
    )
    gain_short = -gain_long

    loader = data.DataLoader(
        base_index=base_index,
        features=features,
        gain_long=gain_long,
        gain_short=gain_short,
        hist_len=2,
        sma_window_size_center=5,
        batch_size=2,
    )

    expected_features_list = [
        {
            "1min": {
                "sma5": np.array([[1.0, 2.0], [2.0, 3.0]]) - np.array([[2.0], [3.0]]),
                "x": np.array([[-1.0, -2.0], [-2.0, -3.0]]) - np.array([[2.0], [3.0]]),
                "sigma": np.array([[11.0, 12.0], [12.0, 13.0]]),
                "minute": np.array([[1, 2], [2, 3]]),
            },
            "2min": {
                "sma5": np.array([[0.5, 1.5], [0.5, 1.5]]) - np.array([[1.5], [1.5]]),
                "x": np.array([[-0.5, -1.5], [-0.5, -1.5]]) - np.array([[1.5], [1.5]]),
                "sigma": np.array([[10.5, 11.5], [10.5, 11.5]]),
                "minute": np.array([[0, 2], [0, 2]]),
            },
        },
        {
            "1min": {
                "sma5": np.array([[3.0, 4.0], [4.0, 5.0]]) - np.array([[4.0], [5.0]]),
                "x": np.array([[-3.0, -4.0], [-4.0, -5.0]]) - np.array([[4.0], [5.0]]),
                "sigma": np.array([[13.0, 14.0], [14.0, 15.0]]),
                "minute": np.array([[3, 4], [4, 5]]),
            },
            "2min": {
                "sma5": np.array([[1.5, 2.5], [1.5, 2.5]]) - np.array([[2.5], [2.5]]),
                "x": np.array([[-1.5, -2.5], [-1.5, -2.5]]) - np.array([[2.5], [2.5]]),
                "sigma": np.array([[11.5, 12.5], [11.5, 12.5]]),
                "minute": np.array([[2, 4], [2, 4]]),
            },
        },
    ]
    expected_gain_long_list = [
        np.array([102.0, 103.0]),
        np.array([104.0, 105.0]),
    ]
    expected_gain_short_list = [-g for g in expected_gain_long_list]

    # iterator のテスト
    for batch_idx, (actual_features, actual_gains) in enumerate(loader):
        for timeframe in ["1min", "2min"]:
            for feature_name in ["sma5", "x", "sigma", "minute"]:
                np.testing.assert_allclose(
                    actual_features[timeframe][feature_name],
                    expected_features_list[batch_idx][timeframe][feature_name],
                )

        np.testing.assert_allclose(actual_gains[0], expected_gain_long_list[batch_idx])
        np.testing.assert_allclose(actual_gains[1], expected_gain_short_list[batch_idx])

    # get_feature_info のテスト
    feature_info, (gain_long_info, gain_short_info) = data.get_feature_info(loader)

    for timeframe in ["1min", "2min"]:
        for feature_name in ["sma5", "x", "sigma"]:
            values = [f[timeframe][feature_name] for f in expected_features_list]
            assert feature_info[timeframe][feature_name].mean == approx(
                np.mean(values)
            ), f"{timeframe} {feature_name}"
            assert feature_info[timeframe][feature_name].var == approx(
                np.var(values)
            ), f"{timeframe} {feature_name}"

        for feature_name in ["minute"]:
            values = [f[timeframe][feature_name] for f in expected_features_list]
            assert feature_info[timeframe][feature_name].max == np.max(
                values
            ), f"{timeframe} {feature_name}"

    assert gain_long_info.mean == approx(np.mean(expected_gain_long_list))
    assert gain_long_info.var == approx(np.var(expected_gain_long_list))
    assert gain_short_info.mean == approx(np.mean(expected_gain_short_list))
    assert gain_short_info.var == approx(np.var(expected_gain_short_list))
