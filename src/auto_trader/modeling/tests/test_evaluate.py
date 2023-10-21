from pathlib import Path
from typing import cast

import numpy as np
from omegaconf import OmegaConf

from auto_trader.modeling import evaluate
from auto_trader.modeling.config import EvalConfig
from auto_trader.modeling.tests import test_train


def test_get_binary_pred() -> None:
    actual = evaluate.get_binary_pred(np.arange(10, dtype=np.float32), 30)
    expected = np.arange(10) > np.percentile(np.arange(10), 30)
    np.testing.assert_array_equal(actual, expected)


def test_main(tmp_path: Path) -> None:
    test_train.test_main(tmp_path)

    config = cast(
        EvalConfig,
        OmegaConf.merge(
            OmegaConf.structured(EvalConfig),
            OmegaConf.create(
                {
                    "output_dir": str(tmp_path / "output"),
                    "params_file": (tmp_path / "output" / "params.pt"),
                    "neptune": {"mode": "debug"},
                    "data": {
                        "symbol": "usdjpy",
                        "cleansed_data_dir": str(tmp_path / "cleansed"),
                        "yyyymm_begin": 202301,
                        "yyyymm_end": 202301,
                    },
                }
            ),
        ),
    )

    evaluate.main(config)
