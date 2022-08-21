import numpy as np
import pandas as pd

import utils


def test_download_preprocessed_data_range(mocker):
    read_preprocessed_data = mocker.patch('utils.download_preprocessed_data', return_value=None)

    gcs = "GCS"
    symbol = "symbol"
    data_directory = "./dir"
    utils.download_preprocessed_data_range(
        gcs, symbol, 2019, 10, 2020, 2, data_directory
    )

    assert read_preprocessed_data.call_args_list == [
        mocker.call(gcs, symbol, 2019, 10, data_directory),
        mocker.call(gcs, symbol, 2019, 11, data_directory),
        mocker.call(gcs, symbol, 2019, 12, data_directory),
        mocker.call(gcs, symbol, 2020, 1, data_directory),
        mocker.call(gcs, symbol, 2020, 2, data_directory),
    ]


def test_read_preprocessed_data_range(mocker):
    read_preprocessed_data = mocker.patch('utils.read_preprocessed_data', return_value=pd.DataFrame())

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


# def test_aggregate_time():
#     index = pd.date_range("2020-1-1 19:45:00", "2020-1-2 05:00:00", freq="1min")
#     data = [i * 1.0 for i in range(len(index))]
#     s_in = pd.Series(data, index)
#     freq = "4h"

#     s_out_first = utils.aggregate_time(s_in, freq, how="first")
#     s_out_last = utils.aggregate_time(s_in, freq, how="last")
#     s_out_max = utils.aggregate_time(s_in, freq, how="max")
#     s_out_min = utils.aggregate_time(s_in, freq, how="min")

#     assert (s_out_first.index == s_in.index).all()
#     assert (s_out_last.index == s_in.index).all()
#     assert (s_out_max.index == s_in.index).all()
#     assert (s_out_min.index == s_in.index).all()

#     # 2020-1-1 19:45 ~ 19:59
#     assert np.isnan(s_out_first.values[:15]).all()
#     assert np.isnan(s_out_last.values[:15]).all()
#     assert np.isnan(s_out_max.values[:15]).all()
#     assert np.isnan(s_out_min.values[:15]).all()

#     # 2020-1-2 04:00 ~ 05:00
#     assert np.isnan(s_out_first.values[-61:]).all()
#     assert np.isnan(s_out_last.values[-61:]).all()
#     assert np.isnan(s_out_max.values[-61:]).all()
#     assert np.isnan(s_out_min.values[-61:]).all()

#     assert s_out_first.tolist()[15:-61] == (
#         [15] * (60 * 4) +  # 20:00 ~ 23:59
#         [15 + 60 * 4] * (60 * 4)  # 00:00 ~ 03:59
#     )
#     assert s_out_last.tolist()[15:-61] == (
#         [14 + 60 * 4] * (60 * 4) +  # 20:00 ~ 23:59
#         [14 + 60 * 8] * (60 * 4)  # 00:00 ~ 03:59
#     )
#     assert s_out_max.tolist()[15:-61] == (
#         [14 + 60 * 4] * (60 * 4) +  # 20:00 ~ 23:59
#         [14 + 60 * 8] * (60 * 4)  # 00:00 ~ 03:59
#     )
#     assert s_out_min.tolist()[15:-61] == (
#         [15] * (60 * 4) +  # 20:00 ~ 23:59
#         [15 + 60 * 4] * (60 * 4)  # 00:00 ~ 03:59
#     )


def test_calc_slope():
    np.testing.assert_allclose(
        utils.calc_slope(np.array([0., 1., 2., 3.]), lookahead=2),
        np.array([1., 1., np.nan, np.nan])
    )

    np.testing.assert_allclose(
        utils.calc_slope(np.array([0., 1., 4., 6.]), lookahead=2),
        np.array([9 / 5, 13 / 5, np.nan, np.nan])
    )


def test_polyline_approx():
    x = np.array([1., 2.099, 3.])
    approx, slope = utils.polyline_approx(x, tol=0.1, lookahead_max=10)
    np.testing.assert_allclose(
        approx,
        np.array([1., 2., 3.])
    )
    # assert (joint == np.array([True, False, True])).all()
    # np.testing.assert_allclose(
    #     joint,
    #     np.array([0, 2])
    # )
    np.testing.assert_allclose(
        slope,
        np.array([1., 1., np.nan])
    )
    # np.testing.assert_allclose(
    #     start,
    #     np.array([1., 1., 1.])
    # )
    # np.testing.assert_allclose(
    #     end,
    #     np.array([3., 3., 3.])
    # )

    x = np.array([1., 2.101, 3.])
    approx, slope = utils.polyline_approx(x, tol=0.1, lookahead_max=10)
    np.testing.assert_allclose(
        approx,
        np.array([1., 2.101, 3.])
    )
    # assert (joint == np.array([True, True, True])).all()
    # np.testing.assert_allclose(
    #     joint,
    #     np.array([0, 1, 2])
    # )
    np.testing.assert_allclose(
        slope,
        np.array([1.101, 0.899, np.nan])
    )
    # np.testing.assert_allclose(
    #     start,
    #     np.array([1., 1., 2.101])
    # )
    # np.testing.assert_allclose(
    #     end,
    #     np.array([2.101, 3., 3.])
    # )

    x = np.array([1., 2., 3., 4., 5])
    approx, slope = utils.polyline_approx(x, tol=0.1, lookahead_max=3)
    np.testing.assert_allclose(
        approx,
        np.array([1., 2., 3., 4., 5.])
    )
    # import pdb; pdb.set_trace()
    # assert (joint == np.array([True, False, True, False, True])).all()
    # np.testing.assert_allclose(
    #     joint,
    #     np.array([0, 2, 4])
    # )
    np.testing.assert_allclose(
        slope,
        np.array([1., 1., 1., 1., np.nan])
    )
    # np.testing.assert_allclose(
    #     start,
    #     np.array([1., 1., 1., 3., 3.])
    # )
    # np.testing.assert_allclose(
    #     end,
    #     np.array([3., 3., 5., 5., 5.])
    # )


# def test_calc_critical_idxs():
#     values = np.array([0., 0.5, 2., 1., -1.])
#     joint_idxs = np.array([0, 1, 2, 3, 4])
#     critical_idxs = utils.calc_critical_idxs(values, joint_idxs)
#     np.testing.assert_array_equal(
#         critical_idxs,
#         np.array([0, 2, 4])
#     )


# def test_calc_next_critical_values():
#     values = np.array([0., 0.5, 2., 1., -1.])
#     critical_idxs = np.array([0, 2, 4])
#     next_critical_values = utils.calc_next_critical_values(values, critical_idxs)
#     np.testing.assert_allclose(
#         next_critical_values,
#         np.array([2., 2., -1., -1., np.nan])
#     )


# # def test

