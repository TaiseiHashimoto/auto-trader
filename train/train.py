import numpy as np
import pandas as pd
import os
import sys
from collections import deque
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
from omegaconf import OmegaConf
import neptune.new as neptune

import utils
from config import LGBMConfig

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import common_utils


def compute_critical_idxs(values: np.ndarray, thresh_hold: float) -> np.ndarray:
    """
    極大・極小をとるインデックスの配列を取得する。
    thresh_hold 以下の変動は無視する。
    """
    # TODO: uptrend から始めて大丈夫か？
    is_uptrend = True
    max_idx = 0
    max_value = values[0]
    min_idx = None
    min_value = np.inf
    critical_idxs = []
    for i in range(1, len(values)):
        v = values[i]
        # print(i, v, is_uptrend)
        if is_uptrend:
            if v > max_value:
                # 上昇中にさらに高値が更新された場合
                max_idx = i
                max_value = v
                # これまでの安値は無効化
                min_idx = None
                min_value = np.inf
            elif v < min_value:
                # 上昇中に安値が更新された場合
                min_idx = i
                min_value = v

                if max_value - min_value > thresh_hold:
                    # print("Add max_idx", max_idx)
                    # print("To downtrend")
                    # 下降に転換
                    critical_idxs.append(max_idx)
                    is_uptrend = False
                    max_idx = None
                    max_value = -np.inf
        else:
            if v < min_value:
                # 下降中にさらに安値が更新された場合
                min_idx = i
                min_value = v
                # これまでの高値は無効化
                max_idx = None
                max_value = -np.inf
            elif v > max_value:
                # 下降中に高値が更新された場合
                max_idx = i
                max_value = v

                if max_value - min_value > thresh_hold:
                    # print("Add min_idx", min_idx)
                    # print("To uptrend")
                    # 上昇に転換
                    critical_idxs.append(min_idx)
                    is_uptrend = True
                    min_idx = None
                    min_value = np.inf
        # print(max_value, min_value)

    if is_uptrend:
        critical_idxs.append(max_idx)
    else:
        critical_idxs.append(min_idx)

    return np.array(critical_idxs)


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


def create_labels_sub(values: np.ndarray, thresh_entry: float, thresh_hold: float):
    critical_idxs = compute_critical_idxs(values, thresh_hold)

    values_next_critical = np.empty(len(values))
    prev_cidx = 0
    for cidx in critical_idxs:
        values_next_critical[prev_cidx:cidx] = values[cidx]
        prev_cidx = cidx

    values_diff = values_next_critical - values

    # entry: 利益が出る and それ以降にさらに利益が出ることはない
    long_entry_labels_ = values_diff > thresh_entry
    short_entry_labels_ = values_diff < -thresh_entry
    # exit: 損失が出る
    long_exit_labels = values_diff < -thresh_hold
    short_exit_labels = values_diff > thresh_hold

    # "それ以降にさらに利益が出ることはない" の部分を考慮
    long_entry_labels = np.zeros(len(values), dtype=bool)
    short_entry_labels = np.zeros(len(values), dtype=bool)
    done_count = 0
    is_uptrend = True
    for cidx in critical_idxs:
        min_value = np.inf
        max_value = -np.inf
        # critical_idx までの index を逆順に走査して、エントリーのタイミングを探す
        for i in reversed(range(done_count, cidx)):
            if is_uptrend and values[i] < min_value:
                min_value = values[i]
                long_entry_labels[i] = long_entry_labels_[i]
            elif not is_uptrend and values[i] > max_value:
                max_value = values[i]
                short_entry_labels[i] = short_entry_labels_[i]

        done_count = cidx
        is_uptrend = not is_uptrend

    return long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels


def create_labels(df: pd.DataFrame, config: OmegaConf) -> pd.DataFrame:
    """
    ラベルを作成する
    df: (ask|bid)_(open|high|low|close)
    config
        label.thresh_entry
        label.thresh_hold
        label.thresh_exit
    """

    df = merge_bid_ask(df)

    mean_high_low = (df["high"].values + df["low"].values) / 2
    long_entry_labels, short_entry_labels, long_exit_labels, short_exit_labels = create_labels_sub(mean_high_low, config.label.thresh_entry, config.label.thresh_hold)

    df_labels = pd.DataFrame({
        "long_entry": long_entry_labels,
        "short_entry": short_entry_labels,
        "long_exit": long_exit_labels,
        "short_exit": short_exit_labels,
    }, index=df.index)

    assert not (df_labels["long_entry"] & df_labels["long_exit"]).any()
    assert not (df_labels["short_entry"] & df_labels["short_exit"]).any()

    return df_labels


def create_featurs(df: pd.DataFrame, config: OmegaConf) -> pd.DataFrame:
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

    df_merged = df_merged.astype(np.float32)

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
    assert (df.index[0].hour, df.index[0].minute) == (0, 0)
    assert (df.index[-1].hour, df.index[-1].minute) == (23, 59)

    # 学習データを準備
    print("Create features")
    df_x = create_featurs(df, config)
    print("Create labels")
    df_y = create_labels(df, config)

    # データが足りない行を削除
    nan_mask = df_x.isnull().any(axis=1)
    df_x = df_x.loc[~nan_mask]
    df_y = df_y.loc[~nan_mask]

    # 学習用パラメータを準備
    model_params = OmegaConf.to_container(config.model, resolve=True)
    model_params["random_state"] = config.random_seed

    # 学習データとテストデータに分けて学習・評価
    print("Train")
    valid_size = int(len(df_x) * config.train.valid_ratio)
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
            config.train.num_iterations,
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

    if not config.train.save_model:
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
            config.train.num_iterations,
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
        run[f"retrain/importance/{label_name}"] = importance.sort_values().to_string()

    # モデル保存
    print("Save model")
    MODEL_PATH = f"{OUTPUT_DIRECTORY}/model.pt"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    model_version = neptune.init_model_version(model=config.neptune.model_id)
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
    base_config = OmegaConf.structured(LGBMConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    print(OmegaConf.to_yaml(config))
    # import pdb; pdb.set_trace()

    if not config.on_colab:
        # GCP サービスアカウントキーの設定
        # colab ではユーザ認証するため不要
        credential_path = pathlib.Path(__file__).resolve().parents[1] / "auto-trader-sa.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)

    main(config)
