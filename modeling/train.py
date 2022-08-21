import numpy as np
import pandas as pd
import os
import sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
from omegaconf import OmegaConf
import neptune.new as neptune

import utils
from config import TrainConfig

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import common_utils


def index2mask(index: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask


def main(config):
    run = neptune.init_run()
    run["config"] = OmegaConf.to_yaml(config)

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

    OUTPUT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent / "output")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # データ読み込み
    print("Load data")
    df = utils.read_preprocessed_data_range(
        config.data.symbol,
        config.data.first_year, config.data.first_month,
        config.data.last_year, config.data.last_month,
        DATA_DIRECTORY
    )

    # 学習データを準備
    print("Create features")
    df_x = utils.create_featurs(
        df,
        config.data.symbol,
        config.feature.timings,
        config.feature.freqs,
        config.feature.sma_timing,
        config.feature.sma_window_size,
        config.feature.lag_max
    )
    print("Create labels")
    df_y = utils.create_labels(
        df,
        config.label.thresh_entry,
        config.label.thresh_hold
    )

    # データが足りない行を削除
    nan_mask = df_x.isnull().any(axis=1)
    df_x = df_x.loc[~nan_mask]
    df_y = df_y.loc[~nan_mask]

    # 学習用パラメータを準備
    model_params = OmegaConf.to_container(config.model, resolve=True)
    model_params["random_state"] = config.random_seed

    # 学習データとテストデータに分けて学習・評価
    print("Train")
    valid_size = int(len(df_x) * config.valid_ratio)
    train_size = len(df_x) - valid_size

    df_x_train = df_x.iloc[:train_size]
    df_x_valid = df_x.iloc[train_size:]
    df_y_train = df_y.iloc[:train_size]
    df_y_valid = df_y.iloc[train_size:]

    run["label/positive_ratio/train"] = df_y_train.mean().to_dict()
    run["label/positive_ratio/valid"] = df_y_valid.mean().to_dict()

    for label_name in df_y.columns:
        train_set = lgb.Dataset(df_x_train, df_y_train[label_name])
        valid_set = lgb.Dataset(df_x_valid, df_y_valid[label_name])

        evals_result = {}
        model = lgb.train(
            model_params,
            train_set,
            config.num_iterations,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.callback.record_evaluation(evals_result),
                lgb.callback.log_evaluation()
            ]
        )

        for loss in evals_result["train"]["binary_logloss"]:
            run[f"train/loss/train/{label_name}"].log(loss)
        for loss in evals_result["valid"]["binary_logloss"]:
            run[f"train/loss/valid/{label_name}"].log(loss)

        train_pred = model.predict(df_x_train).astype(np.float32)
        train_label = df_y_train["long_entry"].values
        run[f"train/auc/train/{label_name}"] = roc_auc_score(train_label, train_pred)

        valid_pred = model.predict(df_x_valid).astype(np.float32)
        valid_label = df_y_valid["long_entry"].values
        run[f"train/auc/valid/{label_name}"] = roc_auc_score(valid_label, valid_pred)

    if not config.save_model:
        return


    # 全データで再学習
    print("Re-train")
    models = {}

    for label_name in df_y.columns:
        train_set = lgb.Dataset(df_x, df_y[label_name])

        evals_result = {}
        model = lgb.train(
            model_params,
            train_set,
            config.num_iterations,
            valid_sets=[train_set],
            valid_names=["train"],
            callbacks=[
                lgb.callback.record_evaluation(evals_result),
                lgb.callback.log_evaluation()
            ]
        )
        models[label_name] = model

        for loss in evals_result["train"]["binary_logloss"]:
            run[f"retrain/loss/train/{label_name}"].log(loss)

        # 特徴量の重要度を記録
        importance = pd.Series(model.feature_importance("gain"), index=df_x.columns)
        importance.sort_values(inplace=True, ascending=False)
        importance_path = f"{OUTPUT_DIRECTORY}/importance_{label_name}.csv"
        importance.to_csv(importance_path)
        run[f"retrain/importance/{label_name}"].upload(importance_path)

    # モデル保存
    print("Save model")
    model_path = f"{OUTPUT_DIRECTORY}/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    model_version = neptune.init_model_version(model=config.neptune.model_id)
    model_version["binary"].upload(model_path)

    # メタデータ保存
    model_version["config"] = OmegaConf.to_yaml(config)
    model_version["run_url"] = run.get_url()
    # config だけからでは学習データの最新の日時がわからないため、別途記録する
    model_version["train/first_timestamp"] = str(df.index[0])
    model_version["train/last_timestamp"] = str(df.index[-1])

    run["model_version_url"] = model_version.get_url()

    # 後片付け
    run.stop()
    model_version.stop()


if __name__ == "__main__":
    config = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.from_cli())
    print(OmegaConf.to_yaml(config))

    ON_COLAB = os.environ.get("ON_COLAB", False)
    if not ON_COLAB:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    # neptune 設定
    os.environ["NEPTUNE_PROJECT"] = config.neptune.project
    secretmanager = common_utils.SecretManagerWrapper(config.gcp.project_id)
    neptune_api_token = secretmanager.fetch_secret(config.gcp.secret_id)
    os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token

    main(config)
