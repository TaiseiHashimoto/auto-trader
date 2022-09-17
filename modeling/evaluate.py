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
from config import get_eval_config

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


def main(config):
    run = neptune.init_run(tags=["eval", config.model_type])
    run["config"] = OmegaConf.to_yaml(config)

    OUTPUT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "output_eval")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    if ON_COLAB:
        DATA_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "preprocessed")
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        # GCS からデータ取得
        print("Download data from GCS")
        gcs = common_utils.GCSWrapper(config.gcp.project_id, config.gcp.bucket_name)
        utils.download_preprocessed_data_range(
            gcs,
            config.data.symbol,
            config.data.first_year, config.data.first_month,
            config.data.last_year, config.data.last_month,
            DATA_DIRECTORY
        )
    else:
        DATA_DIRECTORY = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "preprocessed")

    # モデル取得
    print("Fetch model")
    model_version_id = config.neptune.model_version_id
    if model_version_id == "":
        model_id = common_utils.get_neptune_model_id(config.neptune.project_key, config.model_type)
        model_version_id = get_latest_model_version_id(model_id)

    run["model_version_id"] = model_version_id
    print(f"model_version_id = {model_version_id}")

    model_version = neptune.init_model_version(version=model_version_id)
    model_path = f"{OUTPUT_DIRECTORY}/{model_version_id}_binary.bin"
    if not os.path.exists(model_path):
        model_version["binary"].download(model_path)

    if config.model_type == "lgbm":
        model = lgbm_utils.LGBMModel.from_file(model_path)
    elif config.model_type == "cnn":
        model = cnn_utils.CNNModel.from_file(model_path)

    run["model_version_url"] = model_version.get_url()
    model_version["eval_run_url"] = run.get_url()

    train_config = OmegaConf.create(model_version["config"].fetch())

    # データ読み込み
    print("Load data")
    df = utils.read_preprocessed_data_range(
        config.data.symbol,
        config.data.first_year, config.data.first_month,
        config.data.last_year, config.data.last_month,
        DATA_DIRECTORY
    )
    df = utils.merge_bid_ask(df)

    # 評価データを準備
    print("Create features")
    feature_params = common_utils.conf2dict(train_config.feature)
    base_index, df_x_dict = utils.create_features(df, train_config.data.symbol, **feature_params)

    print("Create labels")
    label_params = common_utils.conf2dict(train_config.label, exclude_keys=["label_type"])
    df_y = utils.create_labels(train_config.label.label_type, df, df_x_dict, label_params)

    # 学習に使われた行を削除
    train_last_timestamp = model_version["train/last_timestamp"].fetch()
    base_index = base_index[base_index > train_last_timestamp]

    if config.model_type == "lgbm":
        ds = lgbm_utils.LGBMDataset(base_index, df_x_dict, df_y, train_config.feature.lag_max, train_config.feature.sma_window_size_center)
    elif config.model_type == "cnn":
        ds = cnn_utils.CNNDataset(base_index, df_x_dict, df_y, train_config.feature.lag_max, train_config.feature.sma_window_size_center)

    eval_first_timestamp = base_index[0]
    eval_last_timestamp = base_index[-1]
    print(f"Evaluation period: {eval_first_timestamp} ~ {eval_last_timestamp}")
    run["first_timestamp"] = str(eval_first_timestamp)
    run["last_timestamp"] = str(eval_last_timestamp)
    days = (eval_last_timestamp - eval_first_timestamp).days * (5/7)
    months = (eval_last_timestamp - eval_first_timestamp).days / 30

    # 予測
    preds = model.predict_score(ds)

    # 予測スコアの AUC とパーセンタイルを計算
    PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for label_name in preds.columns:
        pred = preds[label_name].values
        label = df_y.loc[base_index, label_name].values
        run[f"auc/{label_name}"] = roc_auc_score(label, pred)
        run[f"score_percentile/{label_name}"] = {
            p: np.percentile(pred, p) for p in PERCENTILES
        }

    # ラベルとスコアの CSV をアップロード（分析用）
    labels_path = f"{OUTPUT_DIRECTORY}/labels.csv"
    scores_path = f"{OUTPUT_DIRECTORY}/scores.csv"
    df_y.to_csv(labels_path)
    preds.to_csv(scores_path)
    run["dump/scores"].upload(scores_path)
    run["dump/labels"].upload(labels_path)


    # パーセンタイルとリフトを計算
    rates = df.loc[base_index, config.simulate_timing]
    FUTURE_STEPS = [5, 10, 20, 30, 60]
    preds_binary = {}
    for label_name in preds:
        if "entry" in label_name:
            percentile_list = config.percentile_entry_list
        else:
            percentile_list = config.percentile_exit_list

        preds_binary[label_name] = {}
        for percentile in percentile_list:
            pred_binary = preds[label_name].values >= np.percentile(preds[label_name].values,  percentile)
            preds_binary[label_name][percentile] = pred_binary

            tpr, fpr = utils.calc_tpr_fpr(df_y.loc[base_index, label_name].values, pred_binary)
            run[f"tpr/{label_name}/{percentile}"] = tpr
            run[f"fpr/{label_name}/{percentile}"] = fpr

            for fs in FUTURE_STEPS:
                rates_diff = rates.shift(-fs).values - rates.values
                lift = np.nanmean(rates_diff[pred_binary])
                run[f"lift/{label_name}/{percentile}/{fs}"] = lift

    # シミュレーション
    rates = df.loc[base_index, config.simulate_timing].values
    params_list = list(itertools.product(config.percentile_entry_list, config.percentile_exit_list))
    for percentile_entry, percentile_exit in tqdm(params_list):
        param_str = f"{percentile_entry},{percentile_exit}"

        simulator = common_utils.OrderSimulator(
            config.start_hour,
            config.end_hour,
            config.thresh_loss_cut
        )

        for i, timestamp in enumerate(base_index):
            simulator.step(timestamp,
            rates[i],
            preds_binary["long_entry"][percentile_entry][i],
            preds_binary["short_entry"][percentile_entry][i],
            preds_binary["long_exit"][percentile_exit][i],
            preds_binary["short_exit"][percentile_exit][i],
        )

        profits = np.array([order.gain for order in simulator.order_history]) - config.spread
        timedeltas = np.array([
            (order.exit_timestamp - order.entry_timestamp).total_seconds() / 60
            for order in simulator.order_history
        ])

        run[f"simulation/num_order/{param_str}"] = len(profits)
        if len(profits) > 0:
            run[f"simulation/profit_per_trade/{param_str}"] = profits.mean()
            run[f"simulation/profit_per_day/{param_str}"] = profits.sum() / days
            run[f"simulation/profit_per_month/{param_str}"] = profits.sum() / months
            run[f"simulation/timedelta/min/{param_str}"] = timedeltas.min()
            run[f"simulation/timedelta/max/{param_str}"] = timedeltas.max()
            run[f"simulation/timedelta/mean/{param_str}"] = timedeltas.mean()
            run[f"simulation/timedelta/median/{param_str}"] = np.median(timedeltas)

        results = simulator.export_results()
        results_path = f"{OUTPUT_DIRECTORY}/results_{param_str}.csv"
        results.to_csv(results_path, index=False)
        run[f"simulation/results/{param_str}"].upload(results_path)


if __name__ == "__main__":
    config = get_eval_config()
    print(OmegaConf.to_yaml(config))

    ON_COLAB = os.environ.get("ON_COLAB", False)
    if not ON_COLAB:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    # neptune 設定
    common_utils.setup_neptune(config.neptune.project, config.gcp.project_id, config.gcp.secret_id)

    main(config)
