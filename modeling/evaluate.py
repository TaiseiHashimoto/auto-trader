import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf
import itertools
import neptune.new as neptune

import utils
import lgbm_utils
import cnn_utils
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
    OUTPUT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "output_eval")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

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

    # モデル取得
    print("Fetch model")
    model_version_id = eval_config.neptune.model_version_id
    if model_version_id == "":
        model_id = common_utils.get_neptune_model_id(eval_config.neptune.project_key, model_type)
        model_version_id = get_latest_model_version_id(model_id)
    print(f"model_version_id = {model_version_id}")

    model_version = neptune.init_model_version(version=model_version_id)
    model_path = f"{OUTPUT_DIRECTORY}/model.bin"
    model_version["binary"].download(model_path)

    # with open(model_path, "rb") as f:
    #     models = pickle.load(f)
    if model_type == "lgbm":
        model = lgbm_utils.LGBMModel.from_file(model_path)
    elif model_type == "cnn":
        model = cnn_utils.CNNModel.from_file(model_path)

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
    df = utils.merge_bid_ask(df)

    # 評価データを準備
    print("Create features")
    feature_params = common_utils.conf2dict(train_config.feature)
    base_index, df_x_dict = utils.create_features(df, train_config.data.symbol, **feature_params)
    # if model_type == "lgbm":
    #     df_x_dict = lgbm_utils.create_lgbm_featurs(df, train_config.data.symbol, **feature_params)
    # elif model_type == "cnn":
    #     df_x_dict = cnn_utils.create_cnn_features(df, train_config.data.symbol, **feature_params)

    print("Create labels")
    df_y = utils.create_labels(
        df,
        train_config.label.thresh_entry,
        train_config.label.thresh_hold
    )

    # 学習に使われた行を削除
    train_last_timestamp = model_version["train/last_timestamp"].fetch()
    # df_x_dict = utils.apply_df_dict(df_x_dict, lambda df: df.loc[df.index > train_last_timestamp])
    base_index = base_index[base_index > train_last_timestamp]

    if model_type == "lgbm":
        # df_x = lgbm_utils.merge_features(df_x_dict)
        ds = lgbm_utils.LGBMDataset(base_index, df_x_dict, df_y, train_config.feature.lag_max, train_config.feature.sma_window_size_center)
    elif model_type == "cnn":
        ds = cnn_utils.CNNDataset(base_index, df_x_dict, df_y, train_config.feature.lag_max, train_config.feature.sma_window_size_center)

    eval_first_timestamp = base_index[0]
    eval_last_timestamp = base_index[-1]
    print(f"Evaluation period: {eval_first_timestamp} ~ {eval_last_timestamp}")
    model_version["eval/first_timestamp"] = str(eval_first_timestamp)
    model_version["eval/last_timestamp"] = str(eval_last_timestamp)
    days = (eval_last_timestamp - eval_first_timestamp).days * (5/7)
    months = (eval_last_timestamp - eval_first_timestamp).days / 30

    # 予測
    preds = model.predict_score(ds)

    PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for label_name in preds.columns:
        pred = preds[label_name].values
        label = df_y.loc[base_index, label_name].values
        model_version["eval/auc"] = roc_auc_score(label, pred)
        model_version["eval/pred"] = {
            p: np.percentile(pred, p) for p in PERCENTILES
        }

    # シミュレーション
    params_list = list(itertools.product(eval_config.prob_entry_list, eval_config.prob_exit_list))
    for prob_entry, prob_exit in tqdm(params_list):
        simulator = common_utils.OrderSimulator(
            eval_config.start_hour,
            eval_config.end_hour,
            eval_config.thresh_loss_cut
        )

        long_entry = preds["long_entry"].values > prob_entry
        short_entry = preds["short_entry"].values > prob_entry
        long_exit = preds["long_exit"].values > prob_exit
        short_exit = preds["short_exit"].values > prob_exit
        for i, timestamp in enumerate(base_index):
            rate = df.loc[timestamp, eval_config.simulate_timing]
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
    model_type = sys.argv[1]
    assert model_type in ("lgbm", "cnn")

    base_config = OmegaConf.structured(EvalConfig)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    eval_config = OmegaConf.merge(base_config, cli_config)
    print(OmegaConf.to_yaml(eval_config))

    ON_COLAB = os.environ.get("ON_COLAB", False)
    if not ON_COLAB:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    # neptune 設定
    common_utils.setup_neptune(eval_config.neptune.project, eval_config.gcp.project_id, eval_config.gcp.secret_id)

    main(eval_config)
