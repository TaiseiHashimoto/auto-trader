import calendar
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from omegaconf import OmegaConf

from auto_trader.common import utils
from auto_trader.data.config import CollectConfig


def execute_command(cmd: str) -> None:
    print(f"Execute command `{cmd}`")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def main(config: CollectConfig) -> None:
    os.makedirs(config.raw_data_dir, exist_ok=True)
    output_dir = Path(config.raw_data_dir, config.symbol)

    PRICE_TYPES = ["bid", "ask"]

    if config.recreate_latest:
        # 最新ファイルを削除して作り直す
        for price_type in PRICE_TYPES:
            raw_data_files = sorted(output_dir.glob(f"{price_type}-*.csv"))
            if len(raw_data_files) > 0:
                latest_file_path = raw_data_files[-1]
                print(f"Delete {latest_file_path}")
                latest_file_path.unlink()

    # データをダウンロード
    yesterday = (datetime.now() - timedelta(days=1)).date()
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

            raw_data_file = (
                output_dir / f"{price_type}-{date_first_str}-{date_last_str}.csv"
            )
            if raw_data_file.exists():
                print("Skip")
            else:
                execute_command(
                    (
                        # dukascopy-node を使用する
                        f"npx -y dukascopy-node "
                        f"--instrument {config.symbol} "
                        f"--date-from {date_first.isoformat()} "
                        f"--date-to {date_last.isoformat()} "
                        f"--timeframe m1 "
                        f"--format csv "
                        f"--price-type {price_type} "
                        f"--directory {raw_data_file.parent} "
                        f"--file-name {raw_data_file.stem} "
                        "--cache"
                    )
                )

        yyyymm = utils.calc_yyyymm(yyyymm, month_delta=1)


if __name__ == "__main__":
    config = utils.get_config(CollectConfig)
    print(OmegaConf.to_yaml(config))

    main(config)
