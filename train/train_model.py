import numpy as np
import pandas as pd
import os
import sys
from collections import deque
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
# import yaml
# from dotmap import DotMap
from omegaconf import OmegaConf
import neptune.new as neptune

import utils

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import common_utils
from config import LGBMConfig


def compute_entry_idxs(values: np.ndarray, thresh_entry: float, thresh_hold: float):
    # それ以降で走査した中で最大のインデックス
    max_idxs = deque()
    checked_count = 0
    entry_idxs = []
    gains = []
    for i in range(len(values)):
        # より大きい値が後で見つかった場合には、古いインデックスを削除する
        while len(max_idxs) > 0 and values[max_idxs[-1]] < values[i]:
            max_idxs.pop()
        max_idxs.append(i)

        max_value = values[max_idxs[0]]
        # 最大値からしきい値を超えて下落したときに買い時を探す
        if max_value - values[i] > thresh_hold:
            # それ以降で走査した中で最小のインデックス
            min_idxs = deque()
            for j in range(checked_count, max_idxs[0]):
                while len(min_idxs) > 0 and values[min_idxs[-1]] > values[j]:
                    min_idxs.pop()
                min_idxs.append(j)

                gains.append(max_value - values[j])

            while len(min_idxs) > 0:
                min_value = values[min_idxs[0]]
                if max_value - min_value >= thresh_entry:
                    entry_idxs.append(min_idxs[0])
                min_idxs.popleft()

            checked_count = max_idxs[0]
            max_idxs.popleft()

    for j in range(checked_count, len(values)):
        gains.append(np.nan)

    return np.array(entry_idxs), np.array(gains)


def index2mask(index: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask


def merge_bid_ask(df):
    # bid と ask の平均値を計算
    return pd.DataFrame({
        "open": (df["bid_open"] + df["ask_open"]) / 2,
        "high": (df["bid_high"] + df["ask_high"]) / 2,
        "low": (df["bid_low"] + df["ask_low"]) / 2,
        "close": (df["bid_close"] + df["ask_close"]) / 2,
    }, index=df.index)


def create_labels(
    df: pd.DataFrame, config
    # thresh_entry: float, thresh_hold: float, thresh_exit: float
) -> pd.DataFrame:
    """
    ラベルを作成する
    df: (ask|bid)_(open|high|low|close)
    config
        label.thresh_entry
        label.thresh_hold
        label.thresh_exit
    """

    df = merge_bid_ask(df)
    df = df.add_suffix("_1min")

    long_entry_idxs_high, long_gains_high = compute_entry_idxs(df["high_1min"].values, config.label.thresh_entry, config.label.thresh_hold)
    long_entry_idxs_low, long_gains_low = compute_entry_idxs(df["low_1min"].values, config.label.thresh_entry, config.label.thresh_hold)
    short_entry_idxs_high, short_gains_high = compute_entry_idxs(-df["high_1min"].values, config.label.thresh_entry, config.label.thresh_hold)
    short_entry_idxs_low, short_gains_low = compute_entry_idxs(-df["low_1min"].values, config.label.thresh_entry, config.label.thresh_hold)

    df_labels = pd.DataFrame({
        "long_entry": index2mask(long_entry_idxs_high, len(df)) & index2mask(long_entry_idxs_low, len(df)),
        "short_entry": index2mask(short_entry_idxs_high, len(df)) & index2mask(short_entry_idxs_low, len(df)),
        "long_exit": (long_gains_high < config.label.thresh_exit) | (long_gains_low < config.label.thresh_exit),
        "short_exit": (short_gains_high < config.label.thresh_exit) | (short_gains_low < config.label.thresh_exit),
    }, index=df.index)

    assert not (df_labels["long_entry"] & df_labels["long_exit"]).any()
    assert not (df_labels["short_entry"] & df_labels["short_exit"]).any()

    return df_labels


def create_featurs(
    df: pd.DataFrame, config
    # symbol: str,
    # thresh_entry: float, thresh_hold: float, thresh_exit: float
):
    """
    特徴量を作成する
    df: (ask|bid)_(open|high|low|close)
    config
        data.symbol
        feature.timings
        feature.freqs
        feature.sma_timing
        feature.sma_window_size
    """

    df = merge_bid_ask(df)
    df = df.add_suffix("_1min")

    PIP_SCALE = 0.01 if config.data.symbol == "usdjpy" else 0.0001
    TIMING2OP = {"open": "first", "high": "max", "low": "min", "close": "last"}

    # 複数タイムスケールのデータ作成
    df_dict = {
        "1min": df[[f"{timing}_1min" for timing in config.feature.timings]]
    }
    for freq in config.feature.freqs[1:]:
        df_dict[freq] = pd.concat({
            f"{timing}_{freq}": utils.aggregate_time(df["open_1min"], freq, how=TIMING2OP[timing])
            for timing in config.feature.timings
        }, axis=1)

    # 各タイムスケールで特徴量作成
    for freq, df_agg in df_dict.items():
        # 単純移動平均
        sma = (
            df_agg[f"{config.feature.sma_timing}_{freq}"]
                .shift(1)
                .rolling(config.feature.sma_window_size)
                .mean()
                .astype(np.float32)
        )
        # SettingWithCopyWarning が出るが、問題なさそうなので無視する。
        df_agg[f"{config.feature.sma_timing}_{freq}_sma{config.feature.sma_window_size}_frac"] = (sma / PIP_SCALE) % 100

        for lag_i in range(1, config.feature.lag_max + 1):
            for timing in config.feature.timings:
                df_agg[f"{timing}_{freq}_lag{lag_i}_cent"] = df_agg[f"{timing}_{freq}"].shift(lag_i) - sma

    # 全タイムスケールのデータをまとめる
    df_merged = df_dict["1min"].copy()
    for freq in config.feature.freqs[1:]:
        df_agg = df_dict[freq]
        df_merged = pd.concat([df_merged, df_agg.reindex(df_merged.index, method="ffill")], axis=1)

    assert (df_merged.dtypes == np.float32).all()

    # 時間関連の特徴量
    df_merged["hour"] = df_merged.index.hour
    df_merged["day_of_week"] = df_merged.index.day_of_week
    df_merged["month"] = df_merged.index.month

    # 未来のデータを入力に使わないよう除外 (close_1min など)
    non_feature_names = []
    for freq in config.feature.freqs:
        for timing in config.feature.timings:
            non_feature_names.append(f"{timing}_{freq}")

    feature_names = list(set(df_merged.columns) - set(non_feature_names))

    return df_merged[feature_names]


def main(config):
    # neptune 設定
    os.environ["NEPTUNE_PROJECT"] = config.neptune.project
    secretmanager = common_utils.SecretManagerWrapper(config.gcp.project_id)
    neptune_api_token = secretmanager.fetch_secret(config.gcp.secret_id)
    os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token

    run = neptune.init_run()
    run["config"] = config

    if config.on_colab:
        DATA_DIRECTORY = "./preprocessed"
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        # GCS からデータ取得
        gcs = common_utils.GCSWrapper(config.gcp.project_id, config.gcp.bucket_name)
        utils.download_preprocessed_data_range(
            gcs,
            config.data.symbol,
            config.data.first_year, config.data.first_month,
            config.data.last_year, config.data.last_month,
            DATA_DIRECTORY
        )
    else:
        DATA_DIRECTORY = "../data/preprocessed"

    OUTPUT_DIRECTORY = "./output"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # データ読み込み
    df = utils.read_preprocessed_data_range(
        config.data.symbol,
        config.data.first_year, config.data.first_month,
        config.data.last_year, config.data.last_month,
        DATA_DIRECTORY
    )
    assert (df.index[0].hour, df.index[0].minute) == (0, 0)
    assert (df.index[-1].hour, df.index[-1].minute) == (23, 59)

    # 学習データを準備
    df_y = create_labels(df, config)
    df_x = create_featurs(df, config)

    # データが足りない行を削除
    nan_mask = df_x.isnull().any(axis=1)
    df_x = df_x.loc[~nan_mask]
    df_y = df_y.loc[~nan_mask]

    # 学習用パラメータを準備
    model_params = config.model.toDict()
    model_params["random_state"] = config.random_seed

    # 学習データとテストデータに分けて学習・評価
    valid_size = int(len(df_x) * config.train.valid_ratio)
    train_size = len(df_x) - valid_size

    df_x_train = df_x.iloc[:train_size]
    df_x_valid = df_x.iloc[train_size:]
    df_y_train = df_y.iloc[:train_size]
    df_y_valid = df_y.iloc[train_size:]

    for label_name in df_y.columns:
        train_set = lgb.Dataset(df_x_train, df_y_train[label_name])
        valid_set = lgb.Dataset(df_x_valid, df_y_valid[label_name])

        evals_result = {}
        model = lgb.train(
            model_params,
            train_set,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[lgb.callback.record_evaluation(evals_result)]
        )

        for loss in evals_result["train"]["binary_logloss"]:
            run[f"train/{label_name}/loss/train"].log(loss)
        for loss in evals_result["valid"]["binary_logloss"]:
            run[f"train/{label_name}/loss/valid"].log(loss)

        valid_pred = model.predict(df_x_valid).astype(np.float32)
        valid_label = df_y_valid["long_entry"].values
        run[f"train/{label_name}/auc"] = roc_auc_score(valid_label, valid_pred)

    if not config.train.save_model:
        return


    # 全データで再学習
    models = {}

    for label_name in df_y.columns:
        train_set = lgb.Dataset(df_x, df_y[label_name])

        evals_result = {}
        model = lgb.train(
            model_params,
            train_set,
            valid_sets=[train_set],
            valid_names=["train"],
            callbacks=[lgb.callback.record_evaluation(evals_result)]
        )
        models[label_name] = model

        for loss in evals_result["train"]["binary_logloss"]:
            run[f"retrain/{label_name}/loss/train"].log(loss)

        # 特徴量の重要度を記録
        importance = model.feature_importance("gain")
        run[f"retrain/{label_name}/importance"] = {
            k: v for k, v in zip(df_x.columns, importance)
        }

    # モデル保存
    MODEL_PATH = f"{OUTPUT_DIRECTORY}/model.pt"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    model_version = neptune.init_model_version(model=f"{config.neptune.project_key}-{config.neptune.model_key}")
    model_version["binary"].upload(MODEL_PATH)

    # メタデータ保存
    model_version["config"] = config
    model_version["run_url"] = run.get_url()
    # config だけからでは学習データの最新の日時がわからないため、別途記録する
    model_version["first_timestamp"] = str(df.index[0])
    model_version["last_timestamp"] = str(df.index[-1])

    run["model_version_url"] = model_version.get_url()

    # 後片付け
    run.stop()
    model_version.stop()


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    base_config = OmegaConf.structured(LGBMConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    # import pdb; pdb.set_trace()

    # GCP サービスアカウントキーの設定
    if config.on_colab:
        credential_path = "/content/drive/MyDrive/sa_key/auto-trader-sa.json"
    else:
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    main(config)
