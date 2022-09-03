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


def test_compute_ctirical_idxs():
    actual_critical_idxs = utils.compute_critical_idxs(
        values = np.array([1., 2., 3., 0., 2., 1., 4., 3., 1.]),
        thresh_hold = 1.5
    )
    expected_critical_idxs = np.array([2, 3, 6, 8])
    np.testing.assert_equal(actual_critical_idxs, expected_critical_idxs)

    actual_critical_idxs = utils.compute_critical_idxs(
        values = np.array([3., 2., 1., 0., 1., 2., 1., 0.9, 0.8]),
        thresh_hold = 1.5
    )
    expected_critical_idxs = np.array([0, 3, 5])
    np.testing.assert_equal(actual_critical_idxs, expected_critical_idxs)
