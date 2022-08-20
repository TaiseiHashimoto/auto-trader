import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pickle
from omegaconf import OmegaConf
from typing import Optional
import itertools
import neptune.new as neptune
from neptune.new.types import File

import utils
from config import EvalConfig

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import common_utils


def get_latest_model_version_id(model_id: str):
    model = neptune.init_model(model=model_id)
    model_versions_df = model.fetch_model_versions_table().to_pandas()
    # 削除された model version は除く
    model_versions_df = model_versions_df.loc[~model_versions_df["sys/trashed"]]
    return model_versions_df.iloc[0]["sys/id"]


def main(eval_config):
    if ON_COLAB:
        DATA_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "preprocessed")
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        # GCS からデータ取得
        print("Download data from GCS")
        gcs = common_utils.GCSWrapper(eval_config.gcp.project_id, eval_config.gcp.bucket_name)
        utils.download_preprocessed_data_range(
            gcs,
            eval_config.data.symbol,
            eval_config.data.first_year, eval_config.data.first_month,
            eval_config.data.last_year, eval_config.data.last_month,
            DATA_DIRECTORY
        )
    else:
        DATA_DIRECTORY = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "preprocessed")

    OUTPUT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "output")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # モデル取得
    print("Fetch model")
    model_version_id = eval_config.neptune.model_version_id
    if model_version_id == "":
        model_version_id = get_latest_model_version_id(eval_config.neptune.model_id)
    print(f"model_version_id = {model_version_id}")

    model_version = neptune.init_model_version(version=model_version_id)
    model_path = f"{OUTPUT_DIRECTORY}/model.pkl"
    model_version["binary"].download(model_path)
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # 過去の結果が残っている場合は削除
    if model_version.exists("eval"):
        del model_version["eval"]

    train_config = OmegaConf.create(model_version["config"].fetch())

    # データ読み込み
    print("Load data")
    df = utils.read_preprocessed_data_range(
        eval_config.data.symbol,
        eval_config.data.first_year, eval_config.data.first_month,
        eval_config.data.last_year, eval_config.data.last_month,
        DATA_DIRECTORY
    )

    # 評価データを準備
    print("Create features")
    df_x = utils.create_featurs(
        df,
        train_config.data.symbol,
        train_config.feature.timings,
        train_config.feature.freqs,
        train_config.feature.sma_timing,
        train_config.feature.sma_window_size,
        train_config.feature.lag_max
    )
    print("Create labels")
    df_y = utils.create_labels(
        df,
        train_config.label.thresh_entry,
        train_config.label.thresh_hold
    )

    # データが足りない行を削除
    nan_mask = df_x.isnull().any(axis=1)
    df_x = df_x.loc[~nan_mask]
    df_y = df_y.loc[~nan_mask]

    # 学習に使われた行を削除
    train_last_timestamp = model_version["train/last_timestamp"].fetch()
    time_mask = df_x.index > train_last_timestamp
    df_x = df_x.loc[time_mask]
    df_y = df_y.loc[time_mask]
    df_merged = utils.merge_bid_ask(df.loc[df_x.index])

    eval_first_timestamp = df_x.index[0]
    eval_last_timestamp = df_x.index[-1]
    print(f"Data range : {str(eval_first_timestamp)} ~ {str(eval_last_timestamp)}")
    model_version["eval/first_timestamp"] = str(eval_first_timestamp)
    model_version["eval/last_timestamp"] = str(eval_last_timestamp)
    days = (eval_last_timestamp - eval_first_timestamp).days * (5/7)
    months = (eval_last_timestamp - eval_first_timestamp).days / 30

    # 予測
    preds = {}
    for label_name in df_y.columns:
        label = df_y[label_name].values
        model = models[label_name]
        pred = model.predict(df_x).astype(np.float32)
        model_version["eval/auc"] = roc_auc_score(label, pred)
        preds[label_name] = pred

    # シミュレーション
    params_list = list(itertools.product(eval_config.prob_entry_list, eval_config.prob_exit_list))
    for prob_entry, prob_exit in tqdm(params_list):
        simulator = common_utils.OrderSimulator(
            eval_config.start_hour,
            eval_config.end_hour,
            eval_config.thresh_loss_cut
        )

        long_entry = preds["long_entry"] > prob_entry
        short_entry = preds["short_entry"] > prob_entry
        long_exit = preds["long_exit"] > prob_exit
        short_exit = preds["short_exit"] > prob_exit
        for i in range(len(df_merged)):
            timestamp = df_merged.index[i]
            rate = df_merged[eval_config.simulate_timing][i]
            simulator.step(timestamp, rate, long_entry[i], short_entry[i], long_exit[i], short_exit[i])

        profits = np.array([order.gain for order in simulator.order_history]) - eval_config.spread
        timedeltas = np.array([
            (order.exit_timestamp - order.entry_timestamp).total_seconds() / 60
            for order in simulator.order_history
        ])

        param_str = f"{prob_entry:.3f},{prob_exit:.3f}"
        model_version[f"eval/simulation/{param_str}/num_order"] = len(profits)
        if len(profits) > 0:
            model_version[f"eval/simulation/{param_str}/profit_per_trade"] = profits.mean()
            model_version[f"eval/simulation/{param_str}/profit_per_day"] = profits.sum() / days
            model_version[f"eval/simulation/{param_str}/profit_per_month"] = profits.sum() / months
            model_version[f"eval/simulation/{param_str}/timedelta/min"] = timedeltas.min()
            model_version[f"eval/simulation/{param_str}/timedelta/max"] = timedeltas.max()
            model_version[f"eval/simulation/{param_str}/timedelta/mean"] = timedeltas.mean()
            model_version[f"eval/simulation/{param_str}/timedelta/median"] = np.median(timedeltas)

        results = simulator.export_results()
        results_path = f"{OUTPUT_DIRECTORY}/results_{param_str}.csv"
        results.to_csv(results_path, index=False)
        model_version[f"eval/simulation/{param_str}/results"].upload(results_path)


if __name__ == "__main__":
    eval_config = OmegaConf.merge(OmegaConf.structured(EvalConfig), OmegaConf.from_cli())
    print(OmegaConf.to_yaml(eval_config))

    ON_COLAB = os.environ.get("ON_COLAB", False)
    if not ON_COLAB:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    # neptune 設定
    os.environ["NEPTUNE_PROJECT"] = eval_config.neptune.project
    secretmanager = common_utils.SecretManagerWrapper(eval_config.gcp.project_id)
    neptune_api_token = secretmanager.fetch_secret(eval_config.gcp.secret_id)
    os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token

    main(eval_config)
