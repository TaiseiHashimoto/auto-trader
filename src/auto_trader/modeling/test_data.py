import numpy as np
import pandas as pd

from auto_trader.modeling import utils


def test_merge_bid_ask():
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
    actual_result = utils.merge_bid_ask(df)
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


def test_resample():
    df_1min = pd.DataFrame(
        {
            "open": [0, 1, 2, 3, 4, 5, 6, 7],
            "high": [0, 10, 20, 30, 40, 50, 60, 70],
            "low": [0, -10, -20, -30, -40, -50, -60, -70],
            "close": [0, -1, -2, -3, -4, -5, -6, -7],
        },
        index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"),
    )
    actual_result = utils.resample(df_1min, freqs=["1min", "2min", "4min"])
    expected_result = {
        "1min": pd.DataFrame(
            {
                "open": [0, 1, 2, 3, 4, 5, 6, 7],
                "high": [0, 10, 20, 30, 40, 50, 60, 70],
                "low": [0, -10, -20, -30, -40, -50, -60, -70],
                "close": [0, -1, -2, -3, -4, -5, -6, -7],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"
            ),
            dtype=np.float32,
        ),
        "2min": pd.DataFrame(
            {
                "open": [0, 2, 4, 6],
                "high": [10, 30, 50, 70],
                "low": [-10, -30, -50, -70],
                "close": [-1, -3, -5, -7],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="2min"
            ),
            dtype=np.float32,
        ),
        "4min": pd.DataFrame(
            {
                "open": [0, 4],
                "high": [30, 70],
                "low": [-30, -70],
                "close": [-3, -7],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="4min"
            ),
            dtype=np.float32,
        ),
    }
    assert_df_dict_equal(actual_result, expected_result)


def test_calc_sma():
    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9], dtype=np.float32)
    actual_result = utils.calc_sma(s, window_size=4)
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


def test_calc_sigma():
    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9])
    actual_result = utils.calc_sigma(s, window_size=4)
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


def test_calc_fraction():
    s = pd.Series([100.1234, 104.4567, 90.7890])

    actual_result = utils.calc_fraction(s, base=0.01, ndigits=2)
    expected_result = pd.Series([12.34, 45.67, 78.90], dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)

    actual_result = utils.calc_fraction(s, base=0.001, ndigits=1)
    expected_result = pd.Series([3.4, 6.7, 9.0], dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_create_features():
    actual_base_index, actual_data = utils.create_features(
        df=pd.DataFrame(
            {
                "open": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "high": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                "low": [0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110],
                "close": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
            ),
        ),
        df_dict_critical={
            "1min": pd.DataFrame(
                {
                    "prev1_pre_critical_values": [-1, 0, 0, 0, 1, 1, 5, 5, 5, 5, 8, 9],
                    "prev2_pre_critical_values": [
                        -1,
                        -1,
                        -1,
                        -1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        5,
                        8,
                    ],
                    "prev3_pre_critical_values": [
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        5,
                    ],
                    "prev1_pre_critical_idxs": [-1, 0, 0, 0, 1, 1, 5, 5, 5, 5, 8, 9],
                    "prev2_pre_critical_idxs": [-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 5, 8],
                    "prev3_pre_critical_idxs": [
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        5,
                    ],
                    "pre_uptrends": [
                        True,
                        True,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                    ],
                },
                index=pd.date_range(
                    "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
                ),
            ),
            "2min": pd.DataFrame(
                {
                    "prev1_pre_critical_values": [-1, 0, 1, 5, 5, 8],
                    "prev2_pre_critical_values": [-1, -1, 0, 1, 1, 5],
                    "prev3_pre_critical_values": [-1, -1, -1, 0, 0, 1],
                    "prev1_pre_critical_idxs": [-1, 0, 1, 5, 5, 5],
                    "prev2_pre_critical_idxs": [-1, -1, 0, 0, 1, 1],
                    "prev3_pre_critical_idxs": [-1, -1, -1, 0, 0, 1],
                    "pre_uptrends": [True, False, True, True, False, False],
                },
                index=pd.date_range(
                    "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min"
                ),
            ),
        },
        symbol="usdjpy",
        freqs=["1min", "2min"],
        base_timing="open",
        candle_usage="sequential",
        sma_usage="sequential",
        sma_window_sizes=[2, 4],
        sma_window_size_center=2,
        sigma_usage="sequential",
        sigma_window_size=3,
        macd_usage="sequential",
        macd_ema_window_size_short=1,
        macd_ema_window_size_long=2,
        macd_sma_window_size=1,
        rsi_usage="sequential",
        rsi_window_size=2,
        stochastics_usage="continuous",
        stochastics_k_window_size=2,
        stochastics_d_window_size=1,
        stochastics_sd_window_size=1,
        sma_frac_usage="continuous",
        sma_frac_ndigits=2,
        critical_values_usage="continuous",
        critical_idxs_usage="continuous",
        critical_trends_usage="continuous",
        time_usage="continuous",
        lag_max=2,
        start_hour=0,
        end_hour=1,
    )

    expected_base_index = pd.date_range(
        "2022-01-01 00:10:00", "2022-01-01 00:11:00", freq="1min"
    )
    assert (actual_base_index == expected_base_index).all()

    assert list(actual_data.keys()) == ["sequential", "continuous"]
    assert list(actual_data["sequential"].keys()) == ["center", "nocenter"]
    assert list(actual_data["sequential"]["center"].keys()) == ["1min", "2min"]
    assert list(actual_data["sequential"]["nocenter"].keys()) == ["1min", "2min"]
    assert list(actual_data["continuous"].keys()) == ["1min", "2min"]
    for freq in ["1min", "2min"]:
        index = pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq=freq)
        assert (actual_data["sequential"]["center"][freq].index == index).all()
        assert (
            actual_data["sequential"]["center"][freq].columns
            == [
                "open",
                "sma2",
                "sma4",
            ]
        ).all()
        assert (
            actual_data["sequential"]["center"][freq].dtypes
            == [
                np.float32,
                np.float32,
                np.float32,
            ]
        ).all()
        assert (
            actual_data["sequential"]["nocenter"][freq].columns
            == [
                "body",
                "upper_shadow",
                "lower_shadow",
                "sigma",
                "macd",
                "macd_signal",
                "rsi",
            ]
        ).all()
        assert (
            actual_data["sequential"]["nocenter"][freq].dtypes
            == [
                np.float32,
                np.float32,
                np.float32,
                np.float32,
                np.float32,
                np.float32,
                np.float32,
            ]
        ).all()
        if freq == "1min":
            assert (
                actual_data["continuous"][freq].columns
                == [
                    "stochastics_k",
                    "stochastics_d",
                    "stochastics_sd",
                    "sma2_frac_lag1",
                    "prev1_pre_critical_values_lag1",
                    "prev2_pre_critical_values_lag1",
                    "prev3_pre_critical_values_lag1",
                    "prev1_pre_critical_idxs_lag1",
                    "prev2_pre_critical_idxs_lag1",
                    "prev3_pre_critical_idxs_lag1",
                    "pre_uptrends",
                    "hour",
                    "day_of_week",
                ]
            ).all()
            assert (
                actual_data["continuous"][freq].dtypes
                == [
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                ]
            ).all()
        else:
            assert (
                actual_data["continuous"][freq].columns
                == [
                    "stochastics_k",
                    "stochastics_d",
                    "stochastics_sd",
                    "sma2_frac_lag1",
                    "prev1_pre_critical_values_lag1",
                    "prev2_pre_critical_values_lag1",
                    "prev3_pre_critical_values_lag1",
                    "prev1_pre_critical_idxs_lag1",
                    "prev2_pre_critical_idxs_lag1",
                    "prev3_pre_critical_idxs_lag1",
                    "pre_uptrends",
                ]
            ).all()
            assert (
                actual_data["continuous"][freq].dtypes
                == [
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                ]
            ).all()

    size = 60 * 72
    actual_base_index, _ = utils.create_features(
        df=pd.DataFrame(
            {
                "open": np.zeros(size),
                "high": np.zeros(size),
                "low": np.zeros(size),
                "close": np.zeros(size),
            },
            index=pd.date_range(
                "2022-12-24 00:00:00", "2022-12-26 23:59:59", freq="1min"
            ),
        ),
        df_dict_critical={
            "1min": pd.DataFrame(
                {
                    "prev1_pre_critical_values": [0] * size,
                    "prev2_pre_critical_values": [1] * size,
                    "prev3_pre_critical_values": [2] * size,
                    "prev1_pre_critical_idxs": [0] * size,
                    "prev2_pre_critical_idxs": [1] * size,
                    "prev3_pre_critical_idxs": [2] * size,
                    "pre_uptrends": [True] * size,
                },
                index=pd.date_range(
                    "2022-12-24 00:00:00", "2022-12-26 23:59:59", freq="1min"
                ),
            ),
            "2min": pd.DataFrame(
                {
                    "prev1_pre_critical_values": [0] * (size // 2),
                    "prev2_pre_critical_values": [1] * (size // 2),
                    "prev3_pre_critical_values": [2] * (size // 2),
                    "prev1_pre_critical_idxs": [0] * (size // 2),
                    "prev2_pre_critical_idxs": [1] * (size // 2),
                    "prev3_pre_critical_idxs": [2] * (size // 2),
                    "pre_uptrends": [True] * (size // 2),
                },
                index=pd.date_range(
                    "2022-12-24 00:00:00", "2022-12-26 23:59:59", freq="2min"
                ),
            ),
        },
        symbol="usdjpy",
        freqs=["1min", "2min"],
        base_timing="open",
        candle_usage="sequential",
        sma_usage="sequential",
        sma_window_sizes=[2, 4],
        sma_window_size_center=2,
        sigma_usage="sequential",
        sigma_window_size=3,
        macd_usage="sequential",
        macd_ema_window_size_short=1,
        macd_ema_window_size_long=2,
        macd_sma_window_size=1,
        rsi_usage="sequential",
        rsi_window_size=2,
        stochastics_usage="continuous",
        stochastics_k_window_size=2,
        stochastics_d_window_size=1,
        stochastics_sd_window_size=1,
        sma_frac_usage="continuous",
        sma_frac_ndigits=2,
        critical_values_usage="continuous",
        critical_idxs_usage="continuous",
        critical_trends_usage="continuous",
        time_usage="continuous",
        lag_max=2,
        start_hour=2,
        end_hour=22,
    )
    expected_base_index = pd.DatetimeIndex(
        [
            *pd.date_range("2022-12-24 02:00:00", "2022-12-24 21:59:59", freq="1min"),
            *pd.date_range("2022-12-26 02:00:00", "2022-12-26 21:59:59", freq="1min"),
        ]
    )
    assert (actual_base_index == expected_base_index).all()


def test_calc_gain():
    # TODO: テスト追加
    pass


def assert_df_dict_equal(df_dict1, df_dict2, **kwargs):
    if isinstance(df_dict1, pd.DataFrame):
        pd.testing.assert_frame_equal(df_dict1, df_dict2, **kwargs)
    else:
        assert df_dict1.keys() == df_dict2.keys()
        for key in df_dict1:
            assert_df_dict_equal(df_dict1[key], df_dict2[key], **kwargs)
