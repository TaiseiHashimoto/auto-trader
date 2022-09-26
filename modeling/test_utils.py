import numpy as np
import pandas as pd
from pytest import approx

import utils


def test_download_preprocessed_data_range(mocker):
    download_preprocessed_data = mocker.patch("utils.download_preprocessed_data", return_value=None)

    gcs = "GCS"
    symbol = "symbol"
    data_directory = "./dir"
    utils.download_preprocessed_data_range(
        gcs, symbol, 2019, 10, 2020, 2, data_directory
    )

    assert download_preprocessed_data.call_args_list == [
        mocker.call(gcs, symbol, 2019, 10, data_directory),
        mocker.call(gcs, symbol, 2019, 11, data_directory),
        mocker.call(gcs, symbol, 2019, 12, data_directory),
        mocker.call(gcs, symbol, 2020, 1, data_directory),
        mocker.call(gcs, symbol, 2020, 2, data_directory),
    ]


def test_read_preprocessed_data_range(mocker):
    read_preprocessed_data = mocker.patch("utils.read_preprocessed_data", return_value=pd.DataFrame())

    symbol = "symbol"
    data_directory = "./dir"
    returned = utils.read_preprocessed_data_range(
        symbol, 2019, 10, 2020, 2, data_directory
    )

    assert isinstance(returned, pd.DataFrame)

    assert read_preprocessed_data.call_args_list == [
        mocker.call(symbol, 2019, 10, data_directory),
        mocker.call(symbol, 2019, 11, data_directory),
        mocker.call(symbol, 2019, 12, data_directory),
        mocker.call(symbol, 2020, 1, data_directory),
        mocker.call(symbol, 2020, 2, data_directory),
    ]


def test_aggregate_time():
    s = pd.Series(
        data=np.arange(15),
        index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:14:00", freq="1min")
    )
    actual_result = utils.aggregate_time(s, "5min", "min")
    expected_result = pd.Series(
        data=np.array([0, 5, 10]),
        index=pd.DatetimeIndex([
            "2022-01-01 00:00:00",
            "2022-01-01 00:05:00",
            "2022-01-01 00:10:00",
        ])
    )
    pd.testing.assert_series_equal(actual_result, expected_result, check_freq=False)

    s = pd.Series(
        data=np.arange(15),
        index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:14:00", freq="1min")
    )
    actual_result = utils.aggregate_time(s, "5min", "max")
    expected_result = pd.Series(
        data=np.array([4, 9, 14]),
        index=pd.DatetimeIndex([
            "2022-01-01 00:00:00",
            "2022-01-01 00:05:00",
            "2022-01-01 00:10:00",
        ])
    )
    pd.testing.assert_series_equal(actual_result, expected_result, check_freq=False)


def test_merge_bid_ask():
    index = pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:03:00", freq="1min")
    df = pd.DataFrame({
        "bid_open": [0, 1, 2, 3],
        "ask_open": [2, 3, 4, 5],
        "bid_high": [0, 2, 4, 6],
        "ask_high": [2, 4, 6, 8],
        "bid_low": [0, 3, 6, 9],
        "ask_low": [2, 5, 8, 11],
        "bid_close": [0, 4, 8, 12],
        "ask_close": [2, 6, 10, 14],
    }, index=index)
    actual_result = utils.merge_bid_ask(df)
    expected_result = pd.DataFrame({
        "open": [1, 2, 3, 4],
        "high": [1, 3, 5, 7],
        "low": [1, 4, 7, 10],
        "close": [1, 5, 9, 13],
    }, index=index)
    pd.testing.assert_frame_equal(actual_result, expected_result, check_dtype=False)


def test_resample():
    df_1min = pd.DataFrame({
        "open": [0, 1, 2, 3, 4, 5, 6, 7],
        "high": [0, 10, 20, 30, 40, 50, 60, 70],
        "low": [0, -10, -20, -30, -40, -50, -60, -70],
        "close": [0, -1, -2, -3, -4, -5, -6, -7],
    }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"))
    actual_result = utils.resample(df_1min, freqs=["1min", "2min", "4min"])
    expected_result = {
        "1min": pd.DataFrame({
            "open": [0, 1, 2, 3, 4, 5, 6, 7],
            "high": [0, 10, 20, 30, 40, 50, 60, 70],
            "low": [0, -10, -20, -30, -40, -50, -60, -70],
            "close": [0, -1, -2, -3, -4, -5, -6, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min")),
        "2min": pd.DataFrame({
            "open": [0, 2, 4, 6],
            "high": [10, 30, 50, 70],
            "low": [-10, -30, -50, -70],
            "close": [-1, -3, -5, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="2min")),
        "4min": pd.DataFrame({
            "open": [0, 4],
            "high": [30, 70],
            "low": [-30, -70],
            "close": [-3, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="4min")),
    }
    assert_df_dict_equal(actual_result, expected_result)


def test_align_frequency():
    base_index = pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min")
    df_dict = {
        "1min": pd.DataFrame({
            "open": [0, 1, 2, 3, 4, 5, 6, 7],
            "high": [0, 10, 20, 30, 40, 50, 60, 70],
            "low": [0, -10, -20, -30, -40, -50, -60, -70],
            "close": [0, -1, -2, -3, -4, -5, -6, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min")),
        "2min": pd.DataFrame({
            "open": [0, 2, 4, 6],
            "high": [10, 30, 50, 70],
            "low": [-10, -30, -50, -70],
            "close": [-1, -3, -5, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="2min")),
        "4min": pd.DataFrame({
            "open": [0, 4],
            "high": [30, 70],
            "low": [-30, -70],
            "close": [-3, -7],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="4min")),
    }
    actual_result = utils.align_frequency(base_index, df_dict)
    expected_result = pd.DataFrame({
        "open_1min": [0, 1, 2, 3, 4, 5, 6, 7],
        "high_1min": [0, 10, 20, 30, 40, 50, 60, 70],
        "low_1min": [0, -10, -20, -30, -40, -50, -60, -70],
        "close_1min": [0, -1, -2, -3, -4, -5, -6, -7],
        "open_2min": [0, 0, 2, 2, 4, 4, 6, 6],
        "high_2min": [10, 10, 30, 30, 50, 50, 70, 70],
        "low_2min": [-10, -10, -30, -30, -50, -50, -70, -70],
        "close_2min": [-1, -1, -3, -3, -5, -5, -7, -7],
        "open_4min": [0, 0, 0, 0, 4, 4, 4, 4],
        "high_4min": [30, 30, 30, 30, 70, 70, 70, 70],
        "low_4min": [-30, -30, -30, -30, -70, -70, -70, -70],
        "close_4min": [-3, -3, -3, -3, -7, -7, -7, -7],
    }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min"))
    pd.testing.assert_frame_equal(expected_result, actual_result)


def test_create_time_features():
    index = pd.date_range("2022-01-01 00:00:00", "2022-01-04 23:59:59", freq="12h")
    actual_result = utils.create_time_features(index)
    expected_result = pd.DataFrame({
        "hour": [0, 12, 0, 12, 0, 12, 0, 12],
        "day_of_week": [5, 5, 6, 6, 0, 0, 1, 1],
        "month": [1, 1, 1, 1, 1, 1, 1, 1],
    }, index=index)
    pd.testing.assert_frame_equal(expected_result, actual_result)


def test_compute_sma():
    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9])
    actual_result = utils.compute_sma(s, window_size=4)
    expected_result = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        (0+4+2+3)/4,
        (4+2+3+6)/4,
        (2+3+6+4)/4,
        (3+6+4+6)/4,
        (6+4+6+9)/4,
    ], dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_compute_sigma():
    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9])
    actual_result = utils.compute_sigma(s, window_size=4)
    expected_result = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        ((0**2 + 4**2 + 2**2 + 3**2)/4 - ((0+4+2+3)/4) ** 2) ** 0.5,
        ((4**2 + 2**2 + 3**2 + 6**2)/4 - ((4+2+3+6)/4) ** 2) ** 0.5,
        ((2**2 + 3**2 + 6**2 + 4**2)/4 - ((2+3+6+4)/4) ** 2) ** 0.5,
        ((3**2 + 6**2 + 4**2 + 6**2)/4 - ((3+6+4+6)/4) ** 2) ** 0.5,
        ((6**2 + 4**2 + 6**2 + 9**2)/4 - ((6+4+6+9)/4) ** 2) ** 0.5,
    ], dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_compute_ema():
    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9])
    actual_result = utils.compute_ema(s, window_size=4)
    alpha = 2 / (4 + 1)
    ema = 0
    expected_result = []
    for i in range(len(s)):
        ema = s[i] * alpha + ema * (1 - alpha)
        expected_result.append(ema)
    expected_result = pd.Series(expected_result, dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_compute_macd(mocker):
    compute_ema = mocker.spy(utils, "compute_ema")
    compute_sma = mocker.spy(utils, "compute_sma")

    s = pd.Series([0, 4, 2, 3, 6, 4, 6, 9])
    ema_window_size_short = 10
    ema_window_size_long = 20
    sma_window_size = 5
    utils.compute_macd(s, ema_window_size_short, ema_window_size_long, sma_window_size)

    assert compute_ema.call_args_list == [mocker.call(s, ema_window_size_short), mocker.call(s, ema_window_size_long)]
    assert compute_sma.call_count == 1 and compute_sma.call_args_list[0][0][1] == sma_window_size


def test_compute_rsi():
    s = pd.Series([0,      4, 2,  3, 6, 4,  6, 9])
    # diff        [np.nan, 4, -2, 1, 3, -2, 2, 3]
    actual_result = utils.compute_rsi(s, window_size=3)
    expected_result = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        (4+1)/(4+2+1),
        (1+3)/(2+1+3),
        (1+3)/(1+3+2),
        (3+2)/(3+2+2),
        (2+3)/(2+2+3),
    ], dtype=np.float32)
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_compute_fraction():
    s = pd.Series([100.1234, 104.4567, 90.7890])

    actual_result = utils.compute_fraction(s, base=0.01, ndigits=2)
    expected_result = pd.Series([12.34, 45.67, 78.90])
    pd.testing.assert_series_equal(expected_result, actual_result)

    actual_result = utils.compute_fraction(s, base=0.001, ndigits=1)
    expected_result = pd.Series([3.4, 6.7, 9.0])
    pd.testing.assert_series_equal(expected_result, actual_result)


def test_create_lagged_features():
    index = pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:07:00", freq="1min")
    df = pd.DataFrame({
        "open": [0, 1, 2, 3, 4, 5, 6, 7],
        "high": [0, 10, 20, 30, 40, 50, 60, 70],
        "low": [0, -10, -20, -30, -40, -50, -60, -70],
        "close": [0, -1, -2, -3, -4, -5, -6, -7],
    }, index=index)
    actual_result = utils.create_lagged_features(df, lag_max=2)
    expected_result = pd.DataFrame({
        "open_lag1": [np.nan, 0, 1, 2, 3, 4, 5, 6],
        "open_lag2": [np.nan, np.nan, 0, 1, 2, 3, 4, 5],
        "high_lag1": [np.nan, 0, 10, 20, 30, 40, 50, 60],
        "high_lag2": [np.nan, np.nan, 0, 10, 20, 30, 40, 50],
        "low_lag1": [np.nan, 0, -10, -20, -30, -40, -50, -60],
        "low_lag2": [np.nan, np.nan, 0, -10, -20, -30, -40, -50],
        "close_lag1": [np.nan, 0, -1, -2, -3, -4, -5, -6],
        "close_lag2": [np.nan, np.nan, 0, -1, -2, -3, -4, -5],
    }, index=index)
    pd.testing.assert_frame_equal(expected_result, actual_result)


def test_create_features():
    actual_base_index, actual_data = utils.create_features(
        df = pd.DataFrame({
            "open":  [0, 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11],
            "high":  [0, 10,  20,  30,  40,  50,  60,  70,  80,  90,  100,  110],
            "low":   [0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110],
            "close": [0, -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10,  -11],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min")),
        df_critical = pd.DataFrame({
            "prev1_pre_critical_values": [np.nan, 0, 0, 0, 1, 1, 5, 5, 5, 5, 8, 9],
            "prev2_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, 0, 0, 1, 1, 1, 1, 5, 8],
            "prev3_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 1, 5],
            "prev1_pre_critical_idxs": [-1, 0, 0, 0, 1, 1, 5, 5, 5, 5, 8, 9],
            "prev2_pre_critical_idxs": [-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 5, 8],
            "prev3_pre_critical_idxs": [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 5],
            "pre_uptrends": [True, True, False, False, True, True, True, True, False, False, False, True],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min")),
        symbol = "usdjpy",
        timings = ["open", "low"],
        freqs = ["1min", "2min"],
        sma_timing = "open",
        sma_window_sizes = [2, 4],
        sma_window_size_center = 2,
        sma_frac_ndigits = 2,
        sigma_timing = "close",
        sigma_window_sizes = [1, 3],
        lag_max = 2,
        start_hour = 0,
        end_hour = 1,
    )
    expected_base_index = pd.date_range("2022-01-01 00:10:00", "2022-01-01 00:11:00", freq="1min")
    expected_data = {
        "sequential": {
            "1min": pd.DataFrame({
                "open": [0,      1,      2,      3,   4,   5,   6,   7,   8,   9,   10,   11],
                "low":  [0,      -10,    -20,    -30, -40, -50, -60, -70, -80, -90, -100, -110],
                "sma2": [np.nan, 0.5,    1.5,    2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,  9.5, 10.5],
                "sma4": [np.nan, np.nan, np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,  8.5, 9.5],
            }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min")),
            "2min": pd.DataFrame({
                "open": [0,      2,      4,      6,   8,   10],
                "low":  [-10,    -30,    -50,    -70, -90, -110],
                "sma2": [np.nan, 1,      3,      5,   7,   9],
                "sma4": [np.nan, np.nan, np.nan, 3,   5,   7],
            }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min")),
        },
        "continuous": {
            "1min": pd.DataFrame({
                "sma2_frac_lag1": [np.nan, np.nan, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                "sigma1_lag1": [np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "sigma3_lag1": [np.nan, np.nan, np.nan, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5, (2/3)**0.5],
                "hour": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "day_of_week": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                "month": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "prev1_pre_critical_values_lag1": [np.nan, np.nan, 0 - 0.5, 0 - 1.5, 0 - 2.5, 1 - 3.5, 1 - 4.5, 5 - 5.5, 5 - 6.5, 5 - 7.5, 5 - 8.5, 8 - 9.5],
                "prev2_pre_critical_values_lag1": [np.nan, np.nan, np.nan, np.nan, np.nan, 0 - 3.5, 0 - 4.5, 1 - 5.5, 1 - 6.5, 1 - 7.5, 1 - 8.5, 5 - 9.5],
                "prev3_pre_critical_values_lag1": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0 - 5.5, 0 - 6.5, 0 - 7.5, 0 - 8.5, 1 - 9.5],
                "prev1_pre_critical_idxs_lag1": [np.nan, np.nan, -1, -2, -3, -3, -4, -1, -2, -3, -4, -2],
                "prev2_pre_critical_idxs_lag1": [np.nan, np.nan, np.nan, np.nan, np.nan, -4, -5, -5, -6, -7, -8, -5],
                "prev3_pre_critical_idxs_lag1": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -6, -7, -8, -9, -9],
                # "pre_uptrends_lag1": [np.nan, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min")),
            "2min": pd.DataFrame({
                "sma2_frac_lag1": [np.nan, np.nan, 0, 0, 0, 0],
                "sigma1_lag1": [np.nan, 0, 0, 0, 0, 0],
                "sigma3_lag1": [np.nan, np.nan, np.nan, (8/3)**0.5, (8/3)**0.5, (8/3)**0.5],
            }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min")),
        }
    }
    assert (actual_base_index == expected_base_index).all()
    assert_df_dict_equal(actual_data, expected_data, check_dtype=False)

    size = 60 * 72
    actual_base_index, _ = utils.create_features(
        df = pd.DataFrame({
            "open":  np.zeros(size),
            "high":  np.zeros(size),
            "low":   np.zeros(size),
            "close": np.zeros(size),
        }, index=pd.date_range("2022-12-24 00:00:00", "2022-12-26 23:59:59", freq="1min")),
        df_critical = pd.DataFrame({
            "prev1_pre_critical_values": [0] * size,
            "prev2_pre_critical_values": [1] * size,
            "prev3_pre_critical_values": [2] * size,
            "prev1_pre_critical_idxs": [0] * size,
            "prev2_pre_critical_idxs": [1] * size,
            "prev3_pre_critical_idxs": [2] * size,
            "pre_uptrends": [True] * size,
        }, index=pd.date_range("2022-12-24 00:00:00", "2022-12-26 23:59:59", freq="1min")),
        symbol = "usdjpy",
        timings = ["open", "low"],
        freqs = ["1min", "2min"],
        sma_timing = "open",
        sma_window_sizes = [2, 4],
        sma_window_size_center = 2,
        sma_frac_ndigits = 2,
        sigma_timing = "close",
        sigma_window_sizes = [1, 3],
        lag_max = 2,
        start_hour = 2,
        end_hour = 22,
    )
    expected_base_index = pd.DatetimeIndex([
        *pd.date_range("2022-12-24 02:00:00", "2022-12-24 21:59:59", freq="1min"),
        *pd.date_range("2022-12-26 02:00:00", "2022-12-26 21:59:59", freq="1min"),
    ])
    assert (actual_base_index == expected_base_index).all()


def test_compute_ctirical_idxs():
    actual_critical_idxs, actual_critical_switch_idxs = utils.compute_critical_idxs(
        values = np.array([1., 2., 3., 0., 2., 1., 5., 3., 1.]),
        thresh_hold = 1.5
    )
    np.testing.assert_equal(actual_critical_idxs, np.array([2, 3, 6, 8]))
    np.testing.assert_equal(actual_critical_switch_idxs, np.array([3, 4, 7, 9]))

    actual_critical_idxs, actual_critical_switch_idxs = utils.compute_critical_idxs(
        values = np.array([3., 2., 1., 0., 1., 2., 1., 0.9, 0.8]),
        thresh_hold = 1.5
    )
    np.testing.assert_equal(actual_critical_idxs, np.array([0, 3, 5]))
    np.testing.assert_equal(actual_critical_switch_idxs, np.array([2, 5, 9]))


def test_compute_neighbor_critical_idxs():
    actual = utils.compute_neighbor_critical_idxs(
        size = 9,
        critical_idxs = np.array([2, 3, 6, 8]),
        offset = 0,
    )
    np.testing.assert_equal(actual, np.array([2, 2, 3, 6, 6, 6, 8, 8, -1]))

    actual = utils.compute_neighbor_critical_idxs(
        size = 9,
        critical_idxs = np.array([2, 3, 6, 8]),
        offset = -1,
    )
    np.testing.assert_equal(actual, np.array([-1, -1, 2, 3, 3, 3, 6, 6, 8]))

    actual = utils.compute_neighbor_critical_idxs(
        size = 9,
        critical_idxs = np.array([0, 3, 5]),
        offset = 0,
    )
    np.testing.assert_equal(actual, np.array([3, 3, 3, 5, 5, -1, -1, -1, -1]))

    actual = utils.compute_neighbor_critical_idxs(
        size = 9,
        critical_idxs = np.array([0, 3, 5]),
        offset = 1,
    )
    np.testing.assert_equal(actual, np.array([5, 5, 5, -1, -1, -1, -1, -1, -1]))


def test_compute_neighbor_pre_critical_idxs():
    actual = utils.compute_neighbor_pre_critical_idxs(
        size = 9,
        critical_idxs = np.array([2, 3, 6, 8]),
        critical_switch_idxs = np.array([3, 4, 7, 9]),
        offset = 0,
    )
    np.testing.assert_equal(actual, np.array([2, 2, 2, 3, 6, 6, 6, 8, 8]))

    actual = utils.compute_neighbor_pre_critical_idxs(
        size = 9,
        critical_idxs = np.array([2, 3, 6, 8]),
        critical_switch_idxs = np.array([3, 4, 7, 9]),
        offset = -1,
    )
    np.testing.assert_equal(actual, np.array([-1, -1, -1, 2, 3, 3, 3, 6, 6]))

    actual = utils.compute_neighbor_pre_critical_idxs(
        size = 9,
        critical_idxs = np.array([0, 3, 5]),
        critical_switch_idxs = np.array([2, 5, 9]),
        offset = 0,
    )
    np.testing.assert_equal(actual, np.array([0, 0, 3, 3, 3, 5, 5, 5, 5]))

    actual = utils.compute_neighbor_pre_critical_idxs(
        size = 9,
        critical_idxs = np.array([0, 3, 5]),
        critical_switch_idxs = np.array([2, 5, 9]),
        offset = 1,
    )
    np.testing.assert_equal(actual, np.array([3, 3, 5, 5, 5, -1, -1, -1, -1]))


def test_compute_trends():
    critical_idxs = np.array([2, 3, 6, 8])
    actual_uptrends = utils.compute_trends(size=9, critical_idxs=critical_idxs)
    expected_uptrends = np.array([True, True, False, True, True, True, False, False, True])
    np.testing.assert_equal(actual_uptrends, expected_uptrends)

    critical_idxs = np.array([0, 3, 5])
    actual_uptrends = utils.compute_trends(size=9, critical_idxs=critical_idxs)
    expected_uptrends = np.array([False, False, False, True, True, False, False, False, False])
    np.testing.assert_equal(actual_uptrends, expected_uptrends)


def test_compute_critical_info():
    values = np.array([1., 2., 3., 0., 2., 1., 5., 3., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_critical_info = utils.compute_critical_info(
        df,
        thresh_hold=1.5,
        prev_max=3,
    )
    expected_critical_info = pd.DataFrame({
        "values": values,
        "prev1_critical_idxs": [-1, -1, 2, 3,  3,  3,  6,  6, 8],
        "prev1_critical_values": [np.nan, np.nan, 3, 0, 0, 0, 5, 5, 1],
        "prev1_pre_critical_idxs": [-1, -1, -1, 2, 3, 3, 3,  6, 6],
        "prev1_pre_critical_values": [np.nan, np.nan, np.nan, 3, 0, 0, 0, 5, 5],
        "prev2_critical_idxs": [-1, -1, -1, 2, 2, 2, 3, 3, 6],
        "prev2_critical_values": [np.nan, np.nan, np.nan, 3, 3, 3, 0, 0, 5],
        "prev2_pre_critical_idxs": [-1, -1, -1, -1, 2, 2, 2, 3, 3],
        "prev2_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, 3, 3, 3, 0, 0],
        "prev3_critical_idxs": [-1, -1, -1, -1, -1, -1, 2, 2, 3],
        "prev3_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, 3, 0],
        "prev3_pre_critical_idxs": [-1, -1, -1, -1, -1, -1, -1, 2, 2],
        "prev3_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, 3],
        "next_critical_idxs": [2, 2, 3, 6, 6, 6, 8, 8, -1],
        "next_critical_values": [3, 3, 0, 5, 5, 5, 1, 1, np.nan],
        "uptrends": [True, True, False, True,  True, True, False, False, True],
        "pre_uptrends": [True, True, True,  False, True, True, True,  False, False],
    })
    pd.testing.assert_frame_equal(expected_critical_info, actual_critical_info, check_dtype=False)

    values = np.array([3., 2., 1., 0., 1., 2., 1., 0.9, 0.8])
    df = pd.DataFrame({"high": values + 2, "low": values - 2})
    actual_critical_info = utils.compute_critical_info(
        df,
        thresh_hold=1.5,
        prev_max=3,
    )
    expected_critical_info = pd.DataFrame({
        "values": values,
        "prev1_critical_idxs": [0, 0, 0, 3, 3, 5, 5, 5, 5],
        "prev1_critical_values": [3, 3, 3, 0, 0, 2, 2, 2, 2],
        "prev1_pre_critical_idxs": [-1, -1, 0, 0, 0, 3, 3, 3, 3],
        "prev1_pre_critical_values": [np.nan, np.nan, 3, 3, 3, 0, 0, 0, 0],
        "prev2_critical_idxs": [-1, -1, -1, 0, 0, 3, 3, 3, 3],
        "prev2_critical_values": [np.nan, np.nan, np.nan, 3, 3, 0, 0, 0, 0],
        "prev2_pre_critical_idxs": [-1, -1, -1, -1, -1, 0,  0,  0,  0],
        "prev2_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 3, 3, 3],
        "prev3_critical_idxs": [-1, -1, -1, -1, -1, 0, 0, 0, 0],
        "prev3_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 3, 3, 3],
        "prev3_pre_critical_idxs": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        "prev3_pre_critical_values": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "next_critical_idxs": [3, 3, 3, 5, 5, -1, -1, -1, -1],
        "next_critical_values":[0, 0, 0, 2, 2, np.nan, np.nan, np.nan, np.nan],
        "uptrends": [False, False, False, True, True, False, False, False, False],
        "pre_uptrends": [True, True, False, False, False, True, True, True, True],
    })
    pd.testing.assert_frame_equal(expected_critical_info, actual_critical_info, check_dtype=False)


def test_create_critical_labels():
    values = np.array([1., 2., 3., 0., 2., 1., 5., 3., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_critical_labels(df, thresh_entry=2.5, thresh_hold=1.5)
    assert_bool_array(actual_labels["long_entry"].values,  [3, 5])
    assert_bool_array(actual_labels["short_entry"].values, [2, 6])
    assert_bool_array(actual_labels["long_exit"].values,   [2, 6, 7])
    assert_bool_array(actual_labels["short_exit"].values,  [0, 3, 4, 5])

    values = np.array([3., 2., 1., 0., 1., 2., 1., 0.9, 0.8])
    df = pd.DataFrame({"high": values + 2, "low": values - 2})
    actual_labels = utils.create_critical_labels(df, thresh_entry=2.5, thresh_hold=1.5)
    assert_bool_array(actual_labels["long_entry"].values,  [])
    assert_bool_array(actual_labels["short_entry"].values, [0])
    assert_bool_array(actual_labels["long_exit"].values,   [0, 1])
    assert_bool_array(actual_labels["short_exit"].values,  [3])


def test_create_critical2_labels():
    df_critical = pd.DataFrame({
        "values":               [1.,   2.,   3.,    0.,    2.,   1.,   5.,    3.,    1.],
        "next_critical_values": [3,    3,    0,     5,     5,    5,    1,     1,     np.nan],
        "uptrends":             [True, True, False, True,  True, True, False, False, True],
        "pre_uptrends":         [True, True, True,  False, True, True, True,  False, False],
    })
    actual_labels = utils.create_critical2_labels(df_critical, thresh_entry=2.5)
    assert_bool_array(actual_labels["long_entry"].values,  [5])
    assert_bool_array(actual_labels["short_entry"].values, [])
    assert_bool_array(actual_labels["long_exit"].values,   [2, 6, 7])
    assert_bool_array(actual_labels["short_exit"].values,  [0, 1, 3, 4, 5, 8])

    df_critical = pd.DataFrame({
        "values":               [3.,    2.,    1.,    0.,    1.,    2.,     1.,     0.9,    0.8],
        "next_critical_values": [0,     0,     0,     2,     2,     np.nan, np.nan, np.nan, np.nan],
        "uptrends":             [False, False, False, True,  True,  False,  False,  False,  False],
        "pre_uptrends":         [True,  True,  False, False, False, True,   True,   True,   True],
    })
    actual_labels = utils.create_critical2_labels(df_critical, thresh_entry=2.5)
    assert_bool_array(actual_labels["long_entry"].values,  [])
    assert_bool_array(actual_labels["short_entry"].values, [])
    assert_bool_array(actual_labels["long_exit"].values,   [0, 1, 2, 5, 6, 7, 8])
    assert_bool_array(actual_labels["short_exit"].values,  [3, 4])


def test_create_smadiff_labels():
    values = np.array([1., 2., 3., 0., 2., 1., -1, 0., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_smadiff_labels(df, window_size_before=2, window_size_after=3, thresh_entry=1.0, thresh_hold=0.)
    # sma_before: [np.nan, 3/2,    5/2,  3/2,  1,  3/2,  0,      -1/2,   1/2]
    # sma_after:  [5/3,    5/3,    1,    2/3,  0,   0,   np.nan, np.nan, np.nan]
    # sma_diff:   [np.nan, np.nan, -3/2, -5/6, -1, -3/2, np.nan, np.nan, np.nan]
    assert_bool_array(actual_labels["long_entry"].values,  [])
    assert_bool_array(actual_labels["short_entry"].values, [2, 5])
    assert_bool_array(actual_labels["long_exit"].values,   [2, 3, 4, 5])
    assert_bool_array(actual_labels["short_exit"].values,  [])

    values = -np.array([1., 2., 3., 0., 2., 1., -1, 0., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_smadiff_labels(df, window_size_before=2, window_size_after=3, thresh_entry=1.0, thresh_hold=0.)
    assert_bool_array(actual_labels["long_entry"].values,  [2, 5])
    assert_bool_array(actual_labels["short_entry"].values, [])
    assert_bool_array(actual_labels["long_exit"].values,   [])
    assert_bool_array(actual_labels["short_exit"].values,  [2, 3, 4, 5])


def test_create_future_labels():
    values = np.array([1., 2., 3., 0., 2., 1., -1, 0., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_future_labels(df, future_step_min=3, future_step_max=5, thresh_entry=1.5, thresh_hold=0.)
    # future_sma: [1, 2/3,  0,  0, np.nan, np.nan, np.nan, np.nan, np.nan]
    # diff:       [0, -4/3, -3, 0. np.nan, np.nan, np.nan, np.nan, np.nan]
    assert_bool_array(actual_labels["long_entry"].values,  [])
    assert_bool_array(actual_labels["short_entry"].values, [2])
    assert_bool_array(actual_labels["long_exit"].values,   [1, 2])
    assert_bool_array(actual_labels["short_exit"].values,  [])

    values = -np.array([1., 2., 3., 0., 2., 1., -1, 0., 1.])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_future_labels(df, future_step_min=3, future_step_max=5, thresh_entry=1.5, thresh_hold=0.)
    assert_bool_array(actual_labels["long_entry"].values,  [2])
    assert_bool_array(actual_labels["short_entry"].values, [])
    assert_bool_array(actual_labels["long_exit"].values,   [])
    assert_bool_array(actual_labels["short_exit"].values,  [1, 2])


def test_create_smatrend_labels():
    values = np.array([0,      1, 2, 3, 7, 8, 6,  16, 14, 3, 10, 8, -6, 7, 5])
    # sma:            [np.nan, 1, 2, 4, 6, 7, 10, 12, 11, 9, 7,  4, 3,  2, np.nan]
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_smatrend_labels(df, window_size=3, step_before=2, step_after=2, thresh_entry=4)
    assert_bool_array(actual_labels["long_entry"].values,  [4])
    assert_bool_array(actual_labels["short_entry"].values, [10])
    assert_bool_array(actual_labels["long_exit"].values,   [7, 8, 9, 10, 11, 12])
    assert_bool_array(actual_labels["short_exit"].values,  [1, 2, 3, 4, 5, 6])

    values = -np.array([0,      1, 2, 3, 7, 8, 6,  16, 14, 3, 10, 8, -6, 7, 5])
    df = pd.DataFrame({"high": values + 1, "low": values - 1})
    actual_labels = utils.create_smatrend_labels(df, window_size=3, step_before=2, step_after=2, thresh_entry=4)
    assert_bool_array(actual_labels["long_entry"].values,  [10])
    assert_bool_array(actual_labels["short_entry"].values, [4])
    assert_bool_array(actual_labels["long_exit"].values,   [1, 2, 3, 4, 5, 6])
    assert_bool_array(actual_labels["short_exit"].values,  [7, 8, 9, 10, 11, 12])


def test_create_gain_labels():
    values = np.array([0, 2, 5, 8, 10, 7, 2, 3])
    future_sma = np.array([25/3, 19/3, 12/3, np.nan, np.nan, np.nan, np.nan, np.nan])
    diff = future_sma - values
    df = pd.DataFrame({"high": values + 1, "low": values - 1})

    entry_bias = -1.
    exit_bias = 1.
    expected_labels = pd.DataFrame({
        "long_entry": diff + entry_bias,
        "short_entry": -diff + entry_bias,
        "long_exit": diff + exit_bias,
        "short_exit": -diff + exit_bias,
    })
    actual_labels = utils.create_gain_labels(
        df,
        future_step_min=3, future_step_max=5,
        entry_bias=entry_bias, exit_bias=exit_bias,
    )
    pd.testing.assert_frame_equal(expected_labels, actual_labels)


def test_create_dummy1_labels():
    index = pd.date_range("2022-01-01 00:00:00", "2022-01-02 23:59:59", freq="4h")
    actual_labels = utils.create_dummy1_labels(index)
    assert_bool_array(actual_labels["long_entry"].values,  [0, 1, 6, 7])
    assert_bool_array(actual_labels["short_entry"].values, [2, 8])
    assert_bool_array(actual_labels["long_exit"].values,   [3, 4, 9, 10])
    assert_bool_array(actual_labels["short_exit"].values,  [5, 11])


def test_create_dummy2_labels():
    df_x_dict = {
        "continuous": {
            "1min": pd.DataFrame({
                "sma10_frac_lag1": np.arange(0, 100, 10)
            })
        }
    }
    actual_labels = utils.create_dummy2_labels(df_x_dict)
    assert_bool_array(actual_labels["long_entry"].values,  [0, 1, 2])
    assert_bool_array(actual_labels["short_entry"].values, [3, 4])
    assert_bool_array(actual_labels["long_exit"].values,   [5, 6, 7])
    assert_bool_array(actual_labels["short_exit"].values,  [8, 9])


def test_create_dummy3_labels():
    df_x_dict = {
        "sequential": {
            "1min": pd.DataFrame({
                "close": [0, 10, 20, 30, 30, 25, 15, 20, 0]
            })
        }
    }
    actual_labels = utils.create_dummy3_labels(df_x_dict)
    assert_bool_array(actual_labels["long_entry"].values,  [3, 4])
    assert_bool_array(actual_labels["short_entry"].values, [8])
    assert_bool_array(actual_labels["long_exit"].values,   [5])
    assert_bool_array(actual_labels["short_exit"].values,  [6, 7])


def test_create_labels():
    # TODO: テスト追加
    pass


def test_calc_specificity():
    label = np.array([True, True, False, False, False])
    pred = np.array([True, True, False, False, True])
    assert utils.calc_specificity(label, pred) == approx(2 / 3)


def assert_df_dict_equal(df_dict1, df_dict2, **kwargs):
    if isinstance(df_dict1, pd.DataFrame):
        pd.testing.assert_frame_equal(df_dict1, df_dict2, **kwargs)
    else:
        assert df_dict1.keys() == df_dict2.keys()
        for key in df_dict1:
            assert_df_dict_equal(df_dict1[key], df_dict2[key], **kwargs)


def assert_bool_array(actual_bool, expected_idx):
        expected_bool = np.zeros(len(actual_bool), dtype=bool)
        expected_bool[expected_idx] = True
        np.testing.assert_equal(actual_bool, expected_bool)
