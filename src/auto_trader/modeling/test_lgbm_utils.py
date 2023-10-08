import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.misc import derivative

from auto_trader.modeling import lgbm_utils


class TestLGBMDataset:
    def prepare_dataset(self):
        base_index = pd.date_range(
            "2022-01-01 00:10:00", "2022-01-01 00:11:00", freq="1min"
        )
        x = {
            "sequential": {
                "center": {
                    "1min": pd.DataFrame(
                        {
                            "open": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                            "low": [
                                0,
                                -10,
                                -20,
                                -30,
                                -40,
                                -50,
                                -60,
                                -70,
                                -80,
                                -90,
                                -100,
                                -110,
                            ],
                            "sma2": [
                                np.nan,
                                0.5,
                                1.5,
                                2.5,
                                3.5,
                                4.5,
                                5.5,
                                6.5,
                                7.5,
                                8.5,
                                9.5,
                                10.5,
                            ],
                            "sma4": [
                                np.nan,
                                np.nan,
                                np.nan,
                                1.5,
                                2.5,
                                3.5,
                                4.5,
                                5.5,
                                6.5,
                                7.5,
                                8.5,
                                9.5,
                            ],
                        },
                        index=pd.date_range(
                            "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
                        ),
                    ),
                    "2min": pd.DataFrame(
                        {
                            "open": [0, 2, 4, 6, 8, 10],
                            "low": [-10, -30, -50, -70, -90, -110],
                            "sma2": [np.nan, 1, 3, 5, 7, 9],
                            "sma4": [np.nan, np.nan, np.nan, 3, 5, 7],
                        },
                        index=pd.date_range(
                            "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min"
                        ),
                    ),
                },
                "nocenter": {
                    "1min": pd.DataFrame(
                        {
                            "macd": [
                                0.0,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                                1.0,
                                1.1,
                            ],
                            "macd_signal": [
                                -0.0,
                                -0.1,
                                -0.2,
                                -0.3,
                                -0.4,
                                -0.5,
                                -0.6,
                                -0.7,
                                -0.8,
                                -0.9,
                                -1.0,
                                -1.1,
                            ],
                        },
                        index=pd.date_range(
                            "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
                        ),
                    ),
                    "2min": pd.DataFrame(
                        {
                            "macd": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            "macd_signal": [-0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        },
                        index=pd.date_range(
                            "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min"
                        ),
                    ),
                },
            },
            "continuous": {
                "1min": pd.DataFrame(
                    {
                        "sma2_frac_lag1": [
                            np.nan,
                            np.nan,
                            50,
                            50,
                            50,
                            50,
                            50,
                            50,
                            50,
                            50,
                            50,
                            50,
                        ],
                        "hour": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "day_of_week": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                        "month": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    },
                    index=pd.date_range(
                        "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
                    ),
                ),
                "2min": pd.DataFrame(
                    {
                        "sma2_frac_lag1": [np.nan, np.nan, 0, 0, 0, 0],
                    },
                    index=pd.date_range(
                        "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="2min"
                    ),
                ),
            },
        }
        y = pd.DataFrame(
            {
                "long_entry": [
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                "short_entry": [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                ],
                "long_exit": [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                ],
                "short_exit": [
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                ],
            },
            index=pd.date_range(
                "2022-01-01 00:00:00", "2022-01-01 00:11:00", freq="1min"
            ),
        )

        return lgbm_utils.LGBMDataset(
            base_index, x, y, lag_max=2, sma_window_size_center=2
        )

    def test_bundle_features(self):
        ds = self.prepare_dataset()
        actual_result = ds.bundle_features()
        expected_result = pd.DataFrame(
            {
                "open_lag1_1min": [0.5, 0.5],
                "open_lag2_1min": [-0.5, -0.5],
                "low_lag1_1min": [-98.5, -109.5],
                "low_lag2_1min": [-88.5, -99.5],
                # "sma2_lag1_1min": [8.5, 9.5],
                "sma2_lag2_1min": [-1, -1],
                "sma4_lag1_1min": [-1, -1],
                "sma4_lag2_1min": [-2, -2],
                "macd_lag1_1min": [0.9, 1.0],
                "macd_lag2_1min": [0.8, 0.9],
                "macd_signal_lag1_1min": [-0.9, -1.0],
                "macd_signal_lag2_1min": [-0.8, -0.9],
                "open_lag1_2min": [1, 1],
                "open_lag2_2min": [-1, -1],
                "low_lag1_2min": [-97, -97],
                "low_lag2_2min": [-77, -77],
                # "sma2_lag1_2min": [7, 7],
                "sma2_lag2_2min": [-2, -2],
                "sma4_lag1_2min": [-2, -2],
                "sma4_lag2_2min": [-4, -4],
                "macd_lag1_2min": [0.8, 0.8],
                "macd_lag2_2min": [0.6, 0.6],
                "macd_signal_lag1_2min": [-0.8, -0.8],
                "macd_signal_lag2_2min": [-0.6, -0.6],
                "sma2_frac_lag1_1min": [50, 50],
                "hour_1min": [0, 0],
                "day_of_week_1min": [5, 5],
                "month_1min": [1, 1],
                "sma2_frac_lag1_2min": [0, 0],
            },
            index=ds.base_index,
        )
        pd.testing.assert_frame_equal(expected_result, actual_result, check_dtype=False)

    def test_train_test_split(self):
        ds = self.prepare_dataset()
        ds_train, ds_test = ds.train_test_split(test_proportion=0.5)
        assert ds_train.base_index == pd.DatetimeIndex(
            ["2022-01-01 00:10:00"], freq="1min"
        )
        assert ds_test.base_index == pd.DatetimeIndex(
            ["2022-01-01 00:11:00"], freq="1min"
        )
        assert id(ds_train.x) == id(ds.x)
        assert id(ds_train.y) == id(ds.y)
        assert id(ds_test.x) == id(ds.x)
        assert id(ds_test.y) == id(ds.y)


def test_focal_objective():
    N = 100
    label = np.random.choice([0, 1], size=N, replace=True)
    # HACK: sigmoid 後が 0,1 に近いと誤差が大きくなるため範囲を狭める
    pred = np.random.uniform(-0.5, 0.5, N)
    lds = lgb.Dataset(data=None, label=label)
    gamma = 1.0

    actual_grad, actual_hess = lgbm_utils.focal_objective(pred, lds, gamma)

    def focal_loss(y):
        eps = 1e-6
        ys = 1 / (1 + np.exp(-y))
        return -label * (1 - ys) ** gamma * np.log(ys + eps) - (
            1 - label
        ) * ys**gamma * np.log(1 - ys + eps)

    expected_grad = derivative(focal_loss, pred, n=1, dx=1e-6)
    expected_hess = derivative(focal_loss, pred, n=2, dx=1e-6)

    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual_hess, expected_hess, rtol=1e-3, atol=1e-3)
