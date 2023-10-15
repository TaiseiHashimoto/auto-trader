from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from auto_trader.data import collect


@patch("auto_trader.data.collect.execute_command")
def test_main(execute_command_mock: MagicMock, tmp_path: Path):
    config = OmegaConf.create(
        {
            "symbol": "usdjpy",
            "raw_data_dir": str(tmp_path),
            "yyyymm_begin": 202301,
            "yyyymm_end": 202302,
            "recreate_latest": True,
        }
    )
    # 202301 は bid が存在するので ask だけ作成される
    (tmp_path / "usdjpy-bid-20230101-20230131.csv").touch()
    # 202302 は 削除されて再作成される
    (tmp_path / "usdjpy-bid-20230201-20230228.csv").touch()
    (tmp_path / "usdjpy-ask-20230201-20230228.csv").touch()

    collect.main(config)

    commands = [call.args[0] for call in execute_command_mock.call_args_list]
    assert len(commands) == 3

    # 202301 ask
    assert "--instrument usdjpy" in commands[0]
    assert "--date-from 2023-01-01" in commands[0]
    assert "--date-to 2023-01-31" in commands[0]
    assert "--price-type ask" in commands[0]

    # 202302 bid
    assert "--instrument usdjpy" in commands[1]
    assert "--date-from 2023-02-01" in commands[1]
    assert "--date-to 2023-02-28" in commands[1]
    assert "--price-type bid" in commands[1]

    # 202302 ask
    assert "--instrument usdjpy" in commands[2]
    assert "--date-from 2023-02-01" in commands[2]
    assert "--date-to 2023-02-28" in commands[2]
    assert "--price-type ask" in commands[2]
