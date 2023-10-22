# dukascopy-node (https://github.com/Leo4815162342/dukascopy-node) を使用する

import calendar
import glob
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from auto_trader.common import utils
from auto_trader.data.config import CollectConfig


def execute_command(cmd: str) -> None:
    print(f"Execute command `{cmd}`")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def main(config: CollectConfig) -> None:
    # 古いデータは粒度が粗いので使わない
    if config.yyyymm_begin < 201008:
        raise RuntimeError("Trying to get too old data")

    os.makedirs(config.raw_data_dir, exist_ok=True)

    PRICE_TYPES = ["bid", "ask"]

    if config.recreate_latest:
        # 最新ファイルを削除して作り直す
        for price_type in PRICE_TYPES:
            raw_data_files = sorted(
                glob.glob(f"{config.raw_data_dir}/{config.symbol}-{price_type}-*.csv")
            )
            if len(raw_data_files) > 0:
                latest_file_path = raw_data_files[-1]
                print(f"Delete {latest_file_path}")
                os.remove(latest_file_path)

    # データをダウンロード
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    yyyymm = config.yyyymm_begin
    while yyyymm <= config.yyyymm_end:
        date_first = utils.parse_yyyymm(yyyymm)
        num_days = calendar.monthrange(date_first.year, date_first.month)[1]
        date_last = min(date_first + timedelta(days=num_days - 1), yesterday)

        date_first_str = date_first.strftime("%Y%m%d")
        date_last_str = date_last.strftime("%Y%m%d")
        print(f"{date_first_str} to {date_last_str}")

        for price_type in PRICE_TYPES:
            print(price_type)

            raw_data_file = os.path.join(
                config.raw_data_dir,
                f"{config.symbol}-{price_type}-{date_first_str}-{date_last_str}.csv",
            )
            if os.path.exists(raw_data_file):
                print("Skip")
            else:
                execute_command(
                    (
                        f"npx -y dukascopy-node "
                        f"--instrument {config.symbol} "
                        f"--date-from {date_first.isoformat()} "
                        f"--date-to {date_last.isoformat()} "
                        f"--timeframe m1 "
                        f"--format csv "
                        f"--price-type {price_type} "
                        f"--directory {config.raw_data_dir} "
                        f"--file-name {Path(raw_data_file).stem} "
                        "--cache"
                    )
                )

        yyyymm = utils.calc_yyyymm(yyyymm, month_delta=1)


if __name__ == "__main__":
    config = cast(CollectConfig, utils.get_config(CollectConfig))
    print(OmegaConf.to_yaml(config))

    main(config)
