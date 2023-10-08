import datetime
import glob
import os
import pathlib

import numpy as np
import pandas as pd
import utils
from config import PreprocessConfig
from omegaconf import OmegaConf

from auto_trader.common import common_utils


def validate_data(df, symbol):
    """
    データの妥当性を検証する
    """

    FLAT_RATIO_TOLERANCE = 0.1
    NO_MOVE_RATIO_TOLERANCE = 0.1
    BID_HIGHER_RATIO_TOLERANCE = 0.0

    # フラット期間が一定割合以下
    flat_idxs = np.nonzero(np.all(df.iloc[1:].values == df.iloc[:-1].values, axis=1))[0]
    flat_ratio = len(flat_idxs) / len(df)
    assert (
        flat_ratio <= FLAT_RATIO_TOLERANCE
    ), f"flat_ratio = {flat_ratio} > {FLAT_RATIO_TOLERANCE}"

    # 4値同一が一定割合以下
    no_move_mask = (df["bid_high"] == df["bid_low"]) | (df["ask_high"] == df["ask_low"])
    no_move_ratio = no_move_mask.mean()
    assert (
        no_move_ratio <= NO_MOVE_RATIO_TOLERANCE
    ), f"no_move_ratio = {no_move_ratio} > {NO_MOVE_RATIO_TOLERANCE}"

    # bid > ask が一定割合以下
    bid_higher_mask = (
        (df["bid_open"] > df["ask_open"])
        | (df["bid_high"] > df["ask_high"])
        | (df["bid_low"] > df["ask_low"])
        | (df["bid_close"] > df["ask_close"])
    )
    bid_higher_ratio = bid_higher_mask.mean()
    assert (
        bid_higher_ratio <= BID_HIGHER_RATIO_TOLERANCE
    ), f"bid_higer_ratio = {bid_higher_ratio} > {BID_HIGHER_RATIO_TOLERANCE}"

    # low < open, close < high の順になっている
    invalid_order_mask = (
        (df["bid_open"] < df["bid_low"])
        | (df["bid_close"] < df["bid_low"])
        | (df["bid_open"] > df["bid_high"])
        | (df["bid_close"] > df["bid_high"])
        | (df["ask_open"] < df["ask_low"])
        | (df["ask_close"] < df["ask_low"])
        | (df["ask_open"] > df["ask_high"])
        | (df["ask_close"] > df["ask_high"])
    )
    assert invalid_order_mask.sum() == 0

    if symbol == "usdjpy":
        extreme_value_mask = (df < 70) | (df > 150)
        assert (extreme_value_mask.sum() == 0).all()
    elif symbol == "eurusd":
        extreme_value_mask = (df < 0.8) | (df > 1.6)
        assert (extreme_value_mask.sum() == 0).all()


def main(config):
    IN_DIRECTORY = "./raw"
    OUT_DIRECTORY = "./preprocessed"

    os.makedirs(OUT_DIRECTORY, exist_ok=True)

    # 既存の最新ファイルを削除
    # 最新ファイルは不完全なデータで作成された可能性が高いため、削除して作り直す
    for symbol in config.symbols:
        data_file_paths = glob.glob(f"{OUT_DIRECTORY}/{symbol}-*.pkl")
        if len(data_file_paths) > 0:
            latest_file_path = max(data_file_paths)
            print(f"Delete {latest_file_path}")
            os.remove(latest_file_path)

    #####
    # データを整形して保存
    #####

    for symbol in config.symbols:
        year = config.first_year
        month = config.first_month

        while (year, month) <= (config.last_year, config.last_month):
            year_month_str = f"{year}-{month:02d}"
            file_path = f"{OUT_DIRECTORY}/{symbol}-{year_month_str}.pkl"

            if not os.path.exists(file_path):
                print(f"{symbol}: {year_month_str}")

                # 元データファイルは UTC+0 基準で保存されているので, UTC+2/+3 に合わせるために前月のデータが2/3時間分だけ必要
                prev_year, prev_month = common_utils.calc_year_month_offset(
                    year, month, month_offset=-1
                )
                df_source = pd.concat(
                    [
                        utils.read_raw_data(
                            symbol,
                            prev_year,
                            prev_month,
                            convert_timezone=True,
                            data_directory=IN_DIRECTORY,
                        ),
                        utils.read_raw_data(
                            symbol,
                            year,
                            month,
                            convert_timezone=True,
                            data_directory=IN_DIRECTORY,
                        ),
                    ]
                ).astype(np.float32)

                # 当月データを抽出
                df = df_source.loc[year_month_str]
                df = utils.remove_flat_data(df)

                validate_data(df, symbol)

                df.to_pickle(file_path)

            year, month = common_utils.calc_year_month_offset(
                year, month, month_offset=1
            )

    #####
    # データを GCS に送信
    #####

    gcs = common_utils.GCSWrapper(config.gcp.project_id, config.gcp.bucket_name)
    file_names = gcs.list_file_names()

    for symbol in config.symbols:
        file_names_symbol = sorted(
            [name for name in file_names if name.startswith(symbol)]
        )
        if len(file_names_symbol) == 0:
            existing_file_names = []
        else:
            # 最新ファイルは不完全なデータで作成された可能性が高いため、アップロードし直す。
            existing_file_names = file_names_symbol[:-1]
            print(f"Ignore existing {file_names_symbol[-1]}")

        preprocessed_file_names = sorted(glob.glob(f"{OUT_DIRECTORY}/{symbol}-*.pkl"))
        for src_path in preprocessed_file_names:
            src_base = os.path.basename(src_path)
            if src_base not in existing_file_names:
                print(f"Upload {src_base}")
                gcs.upload_file(local_path=src_path, gcs_path=src_base)


if __name__ == "__main__":
    credential_path = (
        pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    base_config = OmegaConf.structured(PreprocessConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    print(OmegaConf.to_yaml(config))

    main(config)
