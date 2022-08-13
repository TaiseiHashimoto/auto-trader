# dukascopy-node (https://github.com/Leo4815162342/dukascopy-node) のインストールが必要
# npm install  -g dukascopy-node

import datetime
import subprocess
import os
import glob
from omegaconf import OmegaConf

from config import RawConfig


def execute_command(cmd):
    # print(f"Execute command [{cmd}]")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode == 0:
        # print(result.stdout)
        pass
    else:
        raise RuntimeError(result.stderr)


def main(config):
    DATA_DIRECTORY = "./raw"
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    PRICE_TYPES = ["bid", "ask"]

    # 既存の最新ファイルを削除
    # 最新ファイルは不完全なデータで作成された可能性が高いため、削除して作り直す
    for symbol in config.symbols:
        for price_type in PRICE_TYPES:
            data_file_paths = glob.glob(f"{DATA_DIRECTORY}/{symbol}-m1-{price_type}-*.csv")
            if len(data_file_paths) > 0:
                latest_file_path = max(data_file_paths)
                print(f"Delete {latest_file_path}")
                os.remove(latest_file_path)

    #####
    # データをダウンロード
    #####

    # Dukascopy が提供しているデータの開始日時
    # DUKAS_FIRST_DATETIME = {
    #     "usdjpy": "2003-05-04T21:00:00+00:00",
    #     "eurusd": "2003-05-04T21:00:00+00:00",
    # }

    for symbol in config.symbols:
        data_collect_start = datetime.datetime.fromisoformat(config.first_datetime)
        end = datetime.datetime.now(tz=datetime.timezone.utc)

        # 新しいデータから1ヶ月ごとに取得していく
        while True:
            month_start = datetime.datetime(end.year, end.month, day=1, hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc)
            start = max(month_start, data_collect_start)

            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')
            print(f"{symbol}: {start_date} to {end_date}")

            for price_type in PRICE_TYPES:
                file_path = os.path.join(DATA_DIRECTORY, f"{symbol}-m1-{price_type}-{start_date}-{end_date}.csv")
                if not os.path.exists(file_path):
                    execute_command((
                        f"npx dukascopy-node -i {symbol} "
                        f"-from {start.isoformat()} -to {end.isoformat()} "
                        f"-t m1 -f csv -p {price_type} -dir {DATA_DIRECTORY}"
                    ))

            if start == data_collect_start:
                break

            end = start - datetime.timedelta(seconds=1)


if __name__ == "__main__":
    base_config = OmegaConf.structured(RawConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    print(OmegaConf.to_yaml(config))

    main(config)
