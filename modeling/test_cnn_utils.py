import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import cnn_utils


class TestCNNDataset:
    def prepare_dataset(self):
        base_index = pd.date_range("2022-01-01 00:10:00", "2022-01-01 00:15:00", freq="1min")
        x = {
            "sequential": {
                "1min": pd.DataFrame({
                    "open": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    "low": [0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -130, -140, -150],
                    "sma2": [np.nan, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
                    "sma4": [np.nan, np.nan, np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5],
                }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:15:00", freq="1min")),
                "2min": pd.DataFrame({
                    "open": [0, 2, 4, 6, 8, 10, 12, 14],
                    "low": [-10, -30, -50, -70, -90, -110, -130, -150],
                    "sma2": [np.nan, 1, 3, 5, 7, 9, 11, 13],
                    "sma4": [np.nan, np.nan, np.nan, 3, 5, 7, 9, 11],
                }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:15:00", freq="2min")),
            },
            "continuous": {
                "1min": pd.DataFrame({
                    "sma2_frac_lag1": [np.nan, np.nan, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                    "hour": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "day_of_week": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                    "month": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:15:00", freq="1min")),
                "2min": pd.DataFrame({
                    "sma2_frac_lag1": [np.nan, np.nan, 0, 0, 0, 0, 0, 0],
                }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:15:00", freq="2min")),
            }
        }
        y = pd.DataFrame({
            "long_entry": [True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False],
            "short_entry": [False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, True],
            "long_exit": [False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True],
            "short_exit": [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False],
        }, index=pd.date_range("2022-01-01 00:00:00", "2022-01-01 00:15:00", freq="1min"))

        return cnn_utils.CNNDataset(base_index, x, y, lag_max=2, sma_window_size_center=2)

    def test_continuous_dim(self):
        ds = self.prepare_dataset()
        assert ds.continuous_dim() == 5

    def test_train_test_split(self):
        ds = self.prepare_dataset()
        ds_train, ds_test = ds.train_test_split(test_proportion=0.5)
        assert (ds_train.base_index == pd.date_range("2022-01-01 00:10:00", "2022-01-01 00:12:00", freq="1min")).all()
        assert (ds_test.base_index == pd.date_range("2022-01-01 00:13:00", "2022-01-01 00:15:00", freq="1min")).all()
        assert id(ds_train.x) == id(ds.x)
        assert id(ds_train.y) == id(ds.y)
        assert id(ds_test.x) == id(ds.x)
        assert id(ds_test.y) == id(ds.y)

    def test_create_loader(self):
        ds = self.prepare_dataset()
        loader = ds.create_loader(batch_size=5, randomize=False)

        actual_data, actual_labels = next(loader)
        expected_data = {
            "sequential": {
                "1min": np.array([
                    [
                        # lag1 (open, low, sma2, sma4)
                        [9 - 8.5, -90 - 8.5, 8.5 - 8.5, 7.5 - 8.5],
                        # lag2 (open, low, sma2, sma4)
                        [8 - 8.5, -80 - 8.5, 7.5 - 8.5, 6.5 - 8.5],
                    ], [
                        [10 - 9.5, -100 - 9.5, 9.5 - 9.5, 8.5 - 9.5],
                        [9 - 9.5, -90 - 9.5, 8.5 - 9.5, 7.5 - 9.5],
                    ],[
                        [11 - 10.5, -110 - 10.5, 10.5 - 10.5, 9.5 - 10.5],
                        [10 - 10.5, -100 - 10.5, 9.5 - 10.5, 8.5 - 10.5],
                    ], [
                        [12 - 11.5, -120 - 11.5, 11.5 - 11.5, 10.5 - 11.5],
                        [11 - 11.5, -110 - 11.5, 10.5 - 11.5, 9.5 - 11.5],
                    ], [
                        [13 - 12.5, -130 - 12.5, 12.5 - 12.5, 11.5 - 12.5],
                        [12 - 12.5, -120 - 12.5, 11.5 - 12.5, 10.5 - 12.5],
                    ]
                ]),
                "2min": np.array([
                    [
                        # lag1 (open, low, sma2, sma4)
                        [8 - 7, -90 - 7, 7 - 7, 5 - 7],
                        # lag2 (open, low, sma2, sma4)
                        [6 - 7, -70 - 7, 5 - 7, 3 - 7],
                    ], [
                        [8 - 7, -90 - 7, 7 - 7, 5 - 7],
                        [6 - 7, -70 - 7, 5 - 7, 3 - 7],
                    ], [
                        [10 - 9, -110 - 9, 9 - 9, 7 - 9],
                        [8 - 9, -90 - 9, 7 - 9, 5 - 9],
                    ], [
                        [10 - 9, -110 - 9, 9 - 9, 7 - 9],
                        [8 - 9, -90 - 9, 7 - 9, 5 - 9],
                    ], [
                        [12 - 11, -130 - 11, 11 - 11, 9 - 11],
                        [10 - 11, -110 - 11, 9 - 11, 7 - 11],
                    ]
                ]),
            },
            "continuous": {
                "1min": np.array([
                    # sma2_frac_lag1, hour, day_of_week, month
                    [50, 0, 5, 1],
                    [50, 0, 5, 1],
                    [50, 0, 5, 1],
                    [50, 0, 5, 1],
                    [50, 0, 5, 1],
                ]),
                "2min": np.array([
                    # sma2frac_lag1
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ])
            }
        }
        expected_labels = np.array([
            [False, False, True, False],
            [False, True, True, False],
            [True, False, False, True],
            [False, False, False, True],
            [False, False, True, False],
        ])
        assert_np_dict_close(actual_data, expected_data)
        np.testing.assert_equal(actual_labels, expected_labels)

        actual_data, actual_labels = next(loader)
        expected_data = {
            "sequential": {
                "1min": np.array([
                    [
                        [14 - 13.5, -140 - 13.5, 13.5 - 13.5, 12.5 - 13.5],
                        [13 - 13.5, -130 - 13.5, 12.5 - 13.5, 11.5 - 13.5],
                    ]
                ]),
                "2min": np.array([
                    [
                        [12 - 11, -130 - 11, 11 - 11, 9 - 11],
                        [10 - 11, -110 - 11, 9 - 11, 7 - 11],
                    ]
                ]),
            },
            "continuous": {
                "1min": np.array([
                    [50, 0, 5, 1],
                ]),
                "2min": np.array([
                    [0],
                ])
            }
        }
        expected_labels = np.array([
            [False, True, True, False],
        ])
        assert_np_dict_close(actual_data, expected_data)
        assert (actual_labels == expected_labels).all()


class MockedCNNDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def freqs(self):
        return ["1min", "4h"]

    def batch_num(self, batch_size):
        return 10

    def create_loader(self, batch_size, randomize=True):
        for x_b, y_b in zip(self.x, self.y):
            yield x_b, y_b


class TestCNNModel:
    def prepare_model(self):
        return cnn_utils.CNNModel(
            model_params={
                "batch_size": 2,
            },
            init_params=None,
            model=None,
            stats_mean=None,
            stats_var=None,
            run=None
        )

    def test__calc_stats(self):
        # data_size = 3
        # lag_max = 2
        x = {
            "sequential": {
                "1min": np.array([
                    [
                        [0, 1, 2, 3],
                        [1, 3, 5, 7],
                    ],
                    [
                        [2, 5, 8, 11],
                        [3, 7, 11, 15],
                    ],
                    [
                        [4, 9, 14, 19],
                        [5, 11, 17, 23],
                    ],
                ]),
                "4h": np.array([
                    [
                        [0, 10, 20, 30],
                        [10, 30, 50, 70],
                    ],
                    [
                        [20, 50, 80, 110],
                        [30, 70, 110, 150],
                    ],
                    [
                        [40, 90, 140, 190],
                        [50, 110, 170, 230],
                    ],
                ])
            },
            "continuous": {
                "1min": np.array([
                    [-1, -2, -3, -4],
                    [-2, -4, -6, -8],
                    [-3, -6, -9, -12],
                ]),
                "4h": np.array([
                    [-10, -20, -30, -40],
                    [-20, -40, -60, -80],
                    [-30, -60, -90, -120],
                ])
            }
        }
        y = np.array([
            [False, False, True, True],
            [True, True, False, False],
            [False, True, True, False],
        ])

        # batch_size = 2 (data_size = 3 なので2個目のバッチはサイズ1)
        ds = MockedCNNDataset(
            x = [{
                "sequential": {
                    "1min": x["sequential"]["1min"][[0, 1]],
                    "4h": x["sequential"]["4h"][[0, 1]],
                },
                "continuous": {
                    "1min": x["continuous"]["1min"][[0, 1]],
                    "4h": x["continuous"]["4h"][[0, 1]],
                }
            }, {
                "sequential": {
                    "1min": x["sequential"]["1min"][[2]],
                    "4h": x["sequential"]["4h"][[2]],
                },
                "continuous": {
                    "1min": x["continuous"]["1min"][[2]],
                    "4h": x["continuous"]["4h"][[2]],
                }
            }],
            y = [
                y[[0, 1]],
                y[[2]],
            ]
        )
        model = self.prepare_model()
        model._calc_stats(ds)

        actual_mean = model.stats_mean
        actual_var = model.stats_var
        expected_mean = {
            "sequential": {
                "1min": x["sequential"]["1min"].mean(axis=(0, 1)),
                "4h": x["sequential"]["4h"].mean(axis=(0, 1)),
            },
            "continuous": {
                "1min": x["continuous"]["1min"].mean(axis=0),
                "4h": x["continuous"]["4h"].mean(axis=0),
            }
        }
        expected_var = {
            "sequential": {
                "1min": x["sequential"]["1min"].var(axis=(0, 1)),
                "4h": x["sequential"]["4h"].var(axis=(0, 1)),
            },
            "continuous": {
                "1min": x["continuous"]["1min"].var(axis=0),
                "4h": x["continuous"]["4h"].var(axis=0),
            }
        }
        assert_np_dict_close(expected_mean, actual_mean)
        assert_np_dict_close(expected_var, actual_var)


class TestCNNBase:
    def test_num_params(self):
        in_channels = 3
        window_size = 32
        out_channels_list = [20, 40, 20]
        kernel_size_list = [5, 5, 5]
        max_pool_list = [True, False, True]
        out_dim = 128
        batch_norm = False
        dropout = 0.
        base = cnn_utils.CNNBase(
            in_channels=in_channels,
            window_size=window_size,
            out_channels_list=out_channels_list,
            kernel_size_list=kernel_size_list,
            max_pool_list=max_pool_list,
            out_dim=out_dim,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        assert len(base.convs) == 3 + 2 + 3

        print("CNNBase num params")

        assert isinstance(base.convs[0], nn.Conv1d)
        expected_count = (in_channels * kernel_size_list[0] + 1) * out_channels_list[0]
        print(f"convs_0: {expected_count}")
        assert count_params(base.convs[0]) == expected_count
        assert isinstance(base.convs[1], nn.ReLU)
        assert isinstance(base.convs[2], nn.MaxPool1d)

        assert isinstance(base.convs[3], nn.Conv1d)
        expected_count = (out_channels_list[0] * kernel_size_list[1] + 1) * out_channels_list[1]
        print(f"convs_1: {expected_count}")
        assert count_params(base.convs[3]) == expected_count
        assert isinstance(base.convs[4], nn.ReLU)

        assert isinstance(base.convs[5], nn.Conv1d)
        expected_count = (out_channels_list[1] * kernel_size_list[2] + 1) * out_channels_list[2]
        print(f"convs_2: {expected_count}")
        assert count_params(base.convs[5]) == expected_count
        assert isinstance(base.convs[6], nn.ReLU)
        assert isinstance(base.convs[7], nn.MaxPool1d)

        assert isinstance(base.fc_out, nn.Linear)
        expected_count = (out_channels_list[-1] * window_size // (2 ** 2) + 1) * out_dim
        print(f"fc_out: {expected_count}")
        assert count_params(base.fc_out) == expected_count


        in_channels = 3
        window_size = 16
        out_channels_list = [5, 20, 5]
        kernel_size_list = [7, 7, 7]
        max_pool_list = [False, True, False]
        out_dim = 64
        batch_norm = True
        dropout = 0.5
        base = cnn_utils.CNNBase(
            in_channels=in_channels,
            window_size=window_size,
            out_channels_list=out_channels_list,
            kernel_size_list=kernel_size_list,
            max_pool_list=max_pool_list,
            out_dim=out_dim,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        assert len(base.convs) == 4 + 5 + 4

        print("CNNBase num params")

        assert isinstance(base.convs[0], nn.Conv1d)
        expected_count = (in_channels * kernel_size_list[0] + 1) * out_channels_list[0]
        print(f"convs_0: {expected_count}")
        assert count_params(base.convs[0]) == expected_count
        assert isinstance(base.convs[1], nn.BatchNorm1d)
        assert count_params(base.convs[1]) == out_channels_list[0] * 2
        assert isinstance(base.convs[2], nn.ReLU)
        assert isinstance(base.convs[3], nn.Dropout)

        assert isinstance(base.convs[4], nn.Conv1d)
        expected_count = (out_channels_list[0] * kernel_size_list[1] + 1) * out_channels_list[1]
        print(f"convs_1: {expected_count}")
        assert count_params(base.convs[4]) == expected_count
        assert isinstance(base.convs[5], nn.BatchNorm1d)
        assert count_params(base.convs[5]) == out_channels_list[1] * 2
        assert isinstance(base.convs[6], nn.ReLU)
        assert isinstance(base.convs[7], nn.Dropout)
        assert isinstance(base.convs[8], nn.MaxPool1d)

        assert isinstance(base.convs[9], nn.Conv1d)
        expected_count = (out_channels_list[1] * kernel_size_list[2] + 1) * out_channels_list[2]
        print(f"convs_2: {expected_count}")
        assert count_params(base.convs[9]) == expected_count
        assert isinstance(base.convs[10], nn.BatchNorm1d)
        assert count_params(base.convs[10]) == out_channels_list[2] * 2
        assert isinstance(base.convs[11], nn.ReLU)
        assert isinstance(base.convs[12], nn.Dropout)

        assert isinstance(base.fc_out, nn.Linear)
        expected_count = (out_channels_list[-1] * window_size // (2 ** 1) + 1) * out_dim
        print(f"fc_out: {expected_count}")
        assert count_params(base.fc_out) == expected_count


class TestCNNNet:
    def test_num_params(self):
        continuous_dim = 8
        sequential_channels = 3
        freqs = ["high", "low", "close"]
        window_size = 32
        out_channels_list = [20, 40, 20]
        kernel_size_list = [5, 5, 5]
        max_pool_list = [True, False, True]
        base_out_dim = 128
        hidden_dim_list = [256, 128]
        out_dim = 4
        cnn_batch_norm = True
        fc_batch_norm = True
        cnn_dropout = 0.5
        fc_dropout = 0.5

        model = cnn_utils.CNNNet(
            continuous_dim=continuous_dim,
            sequential_channels=sequential_channels,
            freqs=freqs,
            window_size=window_size,
            out_channels_list=out_channels_list,
            kernel_size_list=kernel_size_list,
            max_pool_list=max_pool_list,
            base_out_dim=base_out_dim,
            hidden_dim_list=hidden_dim_list,
            out_dim=out_dim,
            cnn_batch_norm=cnn_batch_norm,
            fc_batch_norm=fc_batch_norm,
            cnn_dropout=cnn_dropout,
            fc_dropout=fc_dropout,
        )

        print("CNNNet num params")

        # +3 の内訳: Conv1d の bias, BatchNorm1d の weight, bias でそれぞれ +1
        expected_count_convs = (
            (sequential_channels * kernel_size_list[0] + 3) * out_channels_list[0]
            + (out_channels_list[0] * kernel_size_list[1] + 3) * out_channels_list[1]
            + (out_channels_list[1] * kernel_size_list[2] + 3) * out_channels_list[2]
            + (out_channels_list[-1] * window_size // (2 ** 2) + 1) * base_out_dim
        )
        print(f"convs (each): expected_count = {expected_count_convs}")
        for conv in model.convs.values():
            assert count_params(conv) == expected_count_convs

        expected_count_fc = (
            (base_out_dim * len(freqs) + continuous_dim + 3) * hidden_dim_list[0]
            + (hidden_dim_list[0] + 3) * hidden_dim_list[1]
            + (hidden_dim_list[1] + 1) * out_dim
        )
        print(f"fc_out: expected_count = {expected_count_fc}")
        assert count_params(model.fc_out) == expected_count_fc

        expected_count_total = expected_count_convs * len(freqs) + expected_count_fc
        print(f"total: expected_count = {expected_count_total}")
        assert count_params(model) == expected_count_total


def assert_np_dict_close(np_dict1, np_dict2, **kwargs):
    if isinstance(np_dict1, np.ndarray):
        np.testing.assert_allclose(np_dict1, np_dict2, **kwargs)
    else:
        assert np_dict1.keys() == np_dict2.keys()
        for key in np_dict1:
            assert_np_dict_close(np_dict1[key], np_dict2[key], **kwargs)


def count_params(model: nn.Module, only_trainable=False) -> int:
    return sum(p.numel() for p in model.parameters() if not only_trainable or p.requires_grad)
