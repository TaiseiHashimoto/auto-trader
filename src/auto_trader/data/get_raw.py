# dukascopy-node (https://github.com/Leo4815162342/dukascopy-node) を使用する

import argparse
import calendar
import glob
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from auto_trader.common import common_utils


def execute_command(cmd):
    print(f"Execute command `{cmd}`")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol", type=str, choices=["usdjpy", "eurusd"], required=True
    )
    parser.add_argument("--yyyymm-begin", type=int, required=True)
    parser.add_argument("--yyyymm-end", type=int, required=True)
    parser.add_argument("--raw-data-dir", type=str, default="./raw")
    parser.add_argument("--recreate-latest", action="store_true")
    args = parser.parse_args()

    # 古いデータは粒度が粗いので使わない
    if args.yyyymm_begin < 201008:
        raise RuntimeError("Trying to get too old data")

    args.raw_data_dir = "./raw"
    os.makedirs(args.raw_data_dir, exist_ok=True)

    PRICE_TYPES = ["bid", "ask"]

    if args.recreate_latest:
        # 最新ファイルを削除して作り直す
        for price_type in PRICE_TYPES:
            raw_data_files = sorted(
                glob.glob(f"{args.cleansed_data_dir}/{args.symbol}-{price_type}-*.csv")
            )
            if len(raw_data_files) > 0:
                latest_file_path = raw_data_files[-1]
                print(f"Delete {latest_file_path}")
                os.remove(latest_file_path)

    # 既存の最新ファイルを削除
    # 最新ファイルは不完全なデータで作成された可能性が高いため、削除して作り直す
    for price_type in PRICE_TYPES:
        data_file_paths = glob.glob(
            f"{args.raw_data_dir}/{args.symbol}-m1-{price_type}-*.csv"
        )
        if len(data_file_paths) > 0:
            latest_file_path = max(data_file_paths)
            print(f"Delete {latest_file_path}")
            os.remove(latest_file_path)

    # データをダウンロード
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    yyyymm = args.yyyymm_begin
    while yyyymm <= args.yyyymm_end:
        date_first = common_utils.parse_yyyymm(yyyymm)
        num_days = calendar.monthrange(date_first.year, date_first.month)[1]
        date_last = min(date_first + timedelta(days=num_days - 1), yesterday)

        date_first_str = date_first.strftime("%Y%m%d")
        date_last_str = date_last.strftime("%Y%m%d")
        print(f"{date_first_str} to {date_last_str}")

        for price_type in PRICE_TYPES:
            print(price_type)

            raw_data_file = os.path.join(
                args.raw_data_dir,
                f"{args.symbol}-{price_type}-{date_first_str}-{date_last_str}.csv",
            )
            if os.path.exists(raw_data_file):
                print("Skip")
            else:
                execute_command(
                    (
                        f"npx -y dukascopy-node "
                        f"--instrument {args.symbol} "
                        f"--date-from {date_first.isoformat()} "
                        f"--date-to {date_last.isoformat()} "
                        f"--timeframe m1 "
                        f"--format csv "
                        f"--price-type {price_type} "
                        f"--directory {args.raw_data_dir} "
                        f"--file-name {Path(raw_data_file).stem} "
                        "--cache"
                    )
                )

        yyyymm = common_utils.calc_yyyymm(yyyymm, month_delta=1)


if __name__ == "__main__":
    main()
