import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import neptune.new as neptune
import functools

import utils


class LGBMDataset:
    def __init__(
        self,
        base_index: pd.DatetimeIndex,
        x: Dict[str, Dict[str, pd.DataFrame]],
        y: pd.DataFrame,
        lag_max: int,
        sma_window_size_center: int,
    ):
        self.base_index = base_index
        self.x = x
        self.y = y
        self.lag_max = lag_max
        self.sma_window_size_center = sma_window_size_center

    def get_label_names(self) -> List[str]:
        return list(self.y.columns)

    def get_freqs(self) -> List[str]:
        return list(self.x["continuous"].keys())

    def get_base_index(self) -> pd.DatetimeIndex:
        return self.base_index

    def bundle_features(self) -> pd.DataFrame:
        df_seq_dict = {}
        for freq in self.get_freqs():
            df_center_lagged = utils.create_lagged_features(self.x["sequential"]["center"][freq], self.lag_max)
            df_nocenter_lagged = utils.create_lagged_features(self.x["sequential"]["nocenter"][freq], self.lag_max)

            # 中心化
            sma_colname = f"sma{self.sma_window_size_center}_lag1"
            sma = df_center_lagged[sma_colname]
            df_center_lagged = df_center_lagged - sma.values[:, np.newaxis]

            df_seq_dict[freq] = pd.concat([
                # 基準となる sma のカラムは常に 0 なので削除する
                df_center_lagged.drop(sma_colname, axis=1),
                df_nocenter_lagged,
            ], axis=1)

        df_seq = utils.align_frequency(self.base_index, df_seq_dict)
        df_cont = utils.align_frequency(self.base_index, self.x["continuous"])
        x = pd.concat([df_seq, df_cont], axis=1)
        assert not x.isnull().any(axis=None)
        return x

    def get_labels(self, label_name: str = None) -> Union[pd.DataFrame, pd.Series]:
        if label_name is None:
            return self.y.loc[self.base_index]
        else:
            return self.y.loc[self.base_index, label_name]

    def train_test_split(self, test_proportion: float) -> Tuple["LGBMDataset", "LGBMDataset"]:
        assert self.y is not None
        train_size = int(len(self.base_index) * (1 - test_proportion))
        ds_train = LGBMDataset(self.base_index[:train_size], self.x, self.y, self.lag_max, self.sma_window_size_center)
        ds_test = LGBMDataset(self.base_index[train_size:], self.x, self.y, self.lag_max, self.sma_window_size_center)
        return ds_train, ds_test



def gain_loss(preds_raw: np.ndarray, lds: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # HACK: preds_raw = 0 で始まると hessian が 0 になり学習が進まないため、意図的にずらしている
    preds_raw_adjusted = preds_raw - 1.
    preds = utils.sigmoid(preds_raw_adjusted)
    loss = -lds.label * preds

    mask = ((preds > 0.01) | (lds.label > 0)).astype(np.float32)
    grad = loss * (1 - preds) * mask
    hess = grad * (1 - preds * 2)
    return loss, grad, hess


def gain_metric(preds_raw: np.ndarray, lds: lgb.Dataset):
    loss, _, _ = gain_loss(preds_raw, lds)
    return "gain_loss", loss.mean(), False


def gain_objective(preds_raw: np.ndarray, lds: lgb.Dataset):
    _, grad, hess = gain_loss(preds_raw, lds)
    return grad, hess


def focal_metric(preds_raw: np.ndarray, lds: lgb.Dataset, gamma: float):
    eps = 1e-6
    preds = utils.sigmoid(preds_raw)
    loss = (
        -lds.label * (1 - preds) ** gamma * np.log(preds + eps)
        -(1 - lds.label) * preds ** gamma * np.log(1 - preds + eps)
    )
    return "focal_loss", loss.mean(), False


def focal_objective(preds_raw: np.ndarray, lds: lgb.Dataset, gamma: float):
    eps = 1e-6
    preds = utils.sigmoid(preds_raw)
    preds_log = np.log(preds + eps)
    preds_inv_log = np.log(1 - preds + eps)
    grad = (
        +lds.label * gamma * preds * (1 - preds) ** gamma * preds_log
        -lds.label * (1 - preds) ** (gamma + 1)
        -(1 - lds.label) * gamma * preds ** gamma * (1 - preds) * preds_inv_log
        +(1 - lds.label) * preds ** (gamma + 1)
    )
    hess = preds * (1 - preds) * (
        +lds.label * gamma * (
            +(1 - preds) ** gamma * preds_log
            -gamma * preds * (1 - preds) ** (gamma - 1) * preds_log
            +(1 - preds) ** gamma
        )
        +lds.label * (gamma + 1) * (1 - preds) ** gamma
        +(1 - lds.label) * gamma * (
            +preds ** gamma * preds_inv_log
            -gamma * preds ** (gamma - 1) * (1 - preds) * preds_inv_log
            +preds ** gamma
        )
        +(1 - lds.label) * (gamma + 1) * preds ** gamma
    )
    return grad, hess


def binary_metric(pred_raw, data):
    y_train = data.get_label()
    pred = utils.sigmoid(pred_raw)
    loss = -(y_train * np.log(pred) + (1-y_train)*np.log(1-pred))
    return 'original_binary_logloss', np.mean(loss), False


def binary_objective(pred_raw, data):
    y_true = data.get_label()
    pred = utils.sigmoid(pred_raw)
    grad = pred - y_true
    hess = pred * (1-pred)
    return grad, hess


class LGBMModel:
    def __init__(
        self,
        model_params: Dict,
        models: Dict[str, lgb.Booster],
        run: neptune.Run,
    ):
        self.model_params = {k: v for k, v in model_params.items() if k != "loss"}
        self.models = models
        self.run = run
        self.additional_params = {}
        self.log_auc = True
        self.sigmoid_after_predict = False

        loss_type = model_params["loss"]["loss_type"]
        if loss_type == "binary":
            self.model_params["objective"] = "binary"
            self.model_params["scale_pos_weight"] = model_params["loss"]["pos_weight"]
        elif loss_type == "gain":
            self.additional_params["fobj"] = gain_objective
            self.additional_params["feval"] = gain_metric
            self.log_auc = False
            self.sigmoid_after_predict = True
        elif loss_type == "focal":
            gamma = model_params["loss"]["gamma"]
            self.additional_params["fobj"] = functools.partial(focal_objective, gamma=gamma)
            self.additional_params["feval"] = functools.partial(focal_metric, gamma=gamma)
            self.sigmoid_after_predict = True

    @classmethod
    def from_scratch(cls, model_params: Dict, run: neptune.Run):
        return cls(model_params, models=None, run=run)

    @classmethod
    def from_file(cls, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        model_params = data["params"]
        models = data["models"]
        return cls(model_params, models=models, run=None)

    def train(
        self,
        ds_train: LGBMDataset,
        ds_valid: Optional[LGBMDataset] = None,
        log_prefix: str = "train",
    ):
        # TODO: log はできるだけ外 (呼び出し側) で行う
        def log_auc(model: lgb.Booster, df_x: pd.DataFrame, df_y: pd.Series, log_suffix: str):
            label = df_y.values
            pred = model.predict(df_x).astype(np.float32)
            self.run[f"{log_prefix}/auc/roc/{log_suffix}"] = roc_auc_score(label, pred)
            self.run[f"{log_prefix}/auc/pr/{log_suffix}"] = average_precision_score(label, pred)

        df_x_train = ds_train.bundle_features()
        if ds_valid is not None:
            df_x_valid = ds_valid.bundle_features()

        self.models = {}

        for label_name in ds_train.get_label_names():
            print(f"label_name = {label_name}")
            df_y_train = ds_train.get_labels(label_name)
            lds_train = lgb.Dataset(df_x_train, df_y_train)
            valid_sets = [lds_train]
            valid_names = ["train"]
            if ds_valid is not None:
                df_y_valid = ds_valid.get_labels(label_name)
                lds_valid = lgb.Dataset(df_x_valid, df_y_valid)
                valid_sets.append(lds_valid)
                valid_names.append("valid")

            evals_results = {}
            model = lgb.train(
                self.model_params,
                lds_train,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.callback.record_evaluation(evals_results),
                    lgb.callback.log_evaluation()
                ],
                **self.additional_params
            )
            self.models[label_name] = model

            for valid_name in valid_names:
                assert len(evals_results[valid_name]) == 1
                loss_name = list(evals_results[valid_name].keys())[0]
                for itr, loss in enumerate(evals_results[valid_name][loss_name]):
                    self.run[f"{log_prefix}/loss/{valid_name}/{label_name}"].log(loss, itr + 1)

            if self.log_auc:
                log_auc(model, df_x_train, df_y_train, log_suffix=f"train/{label_name}")
                if ds_valid is not None:
                    log_auc(model, df_x_valid, df_y_valid, log_suffix=f"valid/{label_name}")

    def get_importance(self) -> Dict[str, pd.Series]:
        importance_dict = {}
        for label_name, model in self.models.items():
            importance = pd.Series(model.feature_importance("gain"), index=model.feature_name())
            importance_dict[label_name] = importance.sort_values(ascending=False)

        return importance_dict

    def predict_score(self, ds: LGBMDataset) -> pd.DataFrame:
        df_x = ds.bundle_features()
        preds = {}
        for label_name, model in self.models.items():
            pred = model.predict(df_x).astype(np.float32)
            if self.sigmoid_after_predict:
                pred = utils.sigmoid(pred)
            preds[label_name] = pred
        return pd.DataFrame(preds, index=ds.get_base_index())

    def save(self, output_path: str):
        assert self.models is not None and len(self.models) > 0
        with open(output_path, "wb") as f:
            pickle.dump({
                "params": self.model_params,
                "models": self.models,
            }, f)
