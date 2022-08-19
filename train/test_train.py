import numpy as np
import pandas as pd

import train


def test_compute_ctirical_idxs():
    actual_critical_idxs = train.compute_critical_idxs(
        values = np.array([1., 2., 3., 0., 2., 1., 4., 3., 1.]),
        thresh_hold = 1.5
    )
    expected_critical_idxs = np.array([2, 3, 6, 8])
    np.testing.assert_equal(actual_critical_idxs, expected_critical_idxs)

    actual_critical_idxs = train.compute_critical_idxs(
        values = np.array([3., 2., 1., 0., 1., 2., 1., 0.9, 0.8]),
        thresh_hold = 1.5
    )
    expected_critical_idxs = np.array([0, 3, 5])
    np.testing.assert_equal(actual_critical_idxs, expected_critical_idxs)



if __name__ == "__main__":
    test_compute_ctirical_idxs()
#     test_compute_entry_idxs()
