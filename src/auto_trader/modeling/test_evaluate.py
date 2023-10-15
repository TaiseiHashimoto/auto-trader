import numpy as np
from pytest import approx

from auto_trader.modeling import evaluate


def test_calc_specificity():
    label = np.array([True, True, False, False, False])
    pred = np.array([True, True, False, False, True])
    assert evaluate.calc_specificity(label, pred) == approx(2 / 3)
