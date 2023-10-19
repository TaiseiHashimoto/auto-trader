from datetime import date

from auto_trader.common import utils


def test_parse_yyyymm():
    assert utils.parse_yyyymm(202301) == date(2023, 1, 1)


def test_calc_yyyymm():
    assert utils.calc_yyyymm(202301, month_delta=2) == 202303
    assert utils.calc_yyyymm(202301, month_delta=11) == 202312
    assert utils.calc_yyyymm(202301, month_delta=12) == 202401
    assert utils.calc_yyyymm(202301, month_delta=-1) == 202212
    assert utils.calc_yyyymm(202302, month_delta=-1) == 202301
    assert utils.calc_yyyymm(202302, month_delta=0) == 202302


def test_get_pip_scale():
    assert utils.get_pip_scale("usdjpy") == 0.01
    assert utils.get_pip_scale("eurusd") == 0.0001
