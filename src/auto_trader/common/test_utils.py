from dataclasses import dataclass, field
from datetime import date

import pytest
from omegaconf import MISSING, MissingMandatoryValue, ValidationError

from auto_trader.common import utils


def test_get_config() -> None:
    @dataclass
    class DummyConfigChild:
        val: int = MISSING

        def __post_init__(self) -> None:
            if self.val != MISSING and self.val <= 0:
                raise ValueError(f"x must be positive: {self.val}")

    @dataclass
    class DummyConfig:
        child: DummyConfigChild = field(default_factory=DummyConfigChild)

    config = utils.get_config(DummyConfig, ["child.val=1"])
    assert type(config) is DummyConfig

    with pytest.raises(MissingMandatoryValue):
        config = utils.get_config(DummyConfig, [])

    with pytest.raises(ValidationError):
        config = utils.get_config(DummyConfig, ["child.val=0.5"])

    with pytest.raises(ValueError):
        config = utils.get_config(DummyConfig, ["child.val=-1"])


def test_parse_yyyymm() -> None:
    assert utils.parse_yyyymm(202301) == date(2023, 1, 1)


def test_calc_yyyymm() -> None:
    assert utils.calc_yyyymm(202301, month_delta=2) == 202303
    assert utils.calc_yyyymm(202301, month_delta=11) == 202312
    assert utils.calc_yyyymm(202301, month_delta=12) == 202401
    assert utils.calc_yyyymm(202301, month_delta=-1) == 202212
    assert utils.calc_yyyymm(202302, month_delta=-1) == 202301
    assert utils.calc_yyyymm(202302, month_delta=0) == 202302


def test_get_pip_scale() -> None:
    assert utils.get_pip_scale("usdjpy") == 0.01
    assert utils.get_pip_scale("eurusd") == 0.0001
