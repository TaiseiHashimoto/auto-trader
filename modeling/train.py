import numpy as np
import os
import sys
from omegaconf import OmegaConf
import neptune.new as neptune

import utils
import lgbm_utils
import cnn_utils
from config import get_train_config, LGBMModelConfig, CNNModelConfig

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
    run["sys/tags"].add(config.model.model_type)
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
    df = utils.merge_bid_ask(df)

    # 学習データを準備
    print("Create features")
    feature_params = common_utils.conf2dict(config.feature)
    base_index, df_x_dict = utils.create_features(df, config.data.symbol, **feature_params)
    print(f"Train period: {base_index[0]} ~ {base_index[-1]}")

    print("Create labels")
    if config.label.label_type == "critical":
        df_y = utils.create_critical_labels(df, thresh_entry=config.label.thresh_entry, thresh_hold=config.label.thresh_hold)
    elif config.label.label_type == "dummy1":
        df_y = utils.create_dummy1_labels(df)
    elif config.label.label_type == "dummy2":
        df_y = utils.create_dummy2_labels(df_x_dict)
    elif config.label.label_type == "dummy3":
        df_y = utils.create_dummy3_labels(df_x_dict)

    if config.model.model_type == "lgbm":
        ds = lgbm_utils.LGBMDataset(base_index, df_x_dict, df_y, config.feature.lag_max, config.feature.sma_window_size_center)
    elif config.model.model_type == "cnn":
        ds = cnn_utils.CNNDataset(base_index, df_x_dict, df_y, config.feature.lag_max, config.feature.sma_window_size_center)

    # 学習用パラメータを準備
    model_params = {k: v for k, v in common_utils.conf2dict(config.model).items() if k != "model_type"}
    if config.model.model_type == "lgbm":
        model = lgbm_utils.LGBMModel.from_scratch(model_params, run)
    elif config.model.model_type == "cnn":
        model = cnn_utils.CNNModel.from_scratch(model_params, run)

    # 学習データとテストデータに分けて学習・評価
    print("Train")
    ds_train, ds_valid = ds.train_test_split(config.valid_ratio)
    run["label/positive_ratio/train"] = ds_train.get_labels().mean().to_dict()
    run["label/positive_ratio/valid"] = ds_valid.get_labels().mean().to_dict()
    model.train_with_validation(ds_train, ds_valid)

    if not config.save_model:
        return

    # 全データで再学習
    print("Re-train")
    importance_dict = model.train_without_validation(ds)
    if config.model.model_type == "lgbm":
        for label_name in importance_dict:
            importance_path = f"{OUTPUT_DIRECTORY}/importance_{label_name}.csv"
            importance_dict[label_name].to_csv(importance_path)
            run[f"train_wo_valid/importance/{label_name}"].upload(importance_path)

    # モデル保存
    print("Save model")
    model_path = f"{OUTPUT_DIRECTORY}/model.bin"
    model.save(model_path)

    model_id = common_utils.get_neptune_model_id(config.neptune.project_key, config.model.model_type)
    model_version = neptune.init_model_version(model=model_id)
    model_version["binary"].upload(model_path)

    # メタデータ保存
    model_version["config"] = OmegaConf.to_yaml(config)
    model_version["run_url"] = run.get_url()
    # config だけからでは学習データの期間が正確にはわからないため、別途記録する
    model_version["train/first_timestamp"] = str(base_index[0])
    model_version["train/last_timestamp"] = str(base_index[-1])

    run["model_version_url"] = model_version.get_url()


if __name__ == "__main__":
    config = get_train_config()
    print(OmegaConf.to_yaml(config))

    common_utils.set_random_seed(config.random_seed)

    ON_COLAB = os.environ.get("ON_COLAB", False)
    if not ON_COLAB:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    # neptune 設定
    common_utils.setup_neptune(config.neptune.project, config.gcp.project_id, config.gcp.secret_id)

    main(config)
