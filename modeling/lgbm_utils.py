import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
import neptune.new as neptune

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

    def label_names(self) -> List[str]:
        return list(self.y.columns)

    def freqs(self) -> List[str]:
        return list(self.x["sequential"].keys())

    def bundle_features(self) -> pd.DataFrame:
        df_seq_dict = {}
        for freq in self.freqs():
            df_lagged = utils.create_lagged_features(self.x["sequential"][freq], self.lag_max)
            sma_colname = f"sma{self.sma_window_size_center}_lag1"
            sma = df_lagged[sma_colname]
            df_lagged_centered = df_lagged - sma.values[:, np.newaxis]
            # sma のカラムは常に 0 なので削除する
            df_seq_dict[freq] = df_lagged_centered.drop(sma_colname, axis=1)

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


class LGBMModel:
    def __init__(
        self,
        model_params: Dict,
        models: Dict[str, lgb.Booster],
        run: neptune.Run,
    ):
        self.model_params = model_params
        self.models = models
        self.run = run

    @classmethod
    def from_scratch(cls, model_params: Dict, run: neptune.Run):
        return cls(model_params, models=None, run=run)

    @classmethod
    def from_file(cls, model_path: str):
        with open(model_path, "rb") as f:
            models = pickle.load(f)
        return cls(model_params=None, models=models, run=None)

    def _train(
        self,
        ds_train: LGBMDataset,
        ds_valid: Optional[LGBMDataset] = None,
        log_prefix: str = "train",
    ):
        def evaluate_auc(model: lgb.Booster, df_x: pd.DataFrame, df_y: pd.Series):
            label = df_y.values
            pred = model.predict(df_x).astype(np.float32)
            return roc_auc_score(label, pred)

        df_x_train = ds_train.bundle_features()
        if ds_valid is not None:
            df_x_valid = ds_valid.bundle_features()

        self.models = {}

        for label_name in ds_train.label_names():
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
                ]
            )
            self.models[label_name] = model

            for valid_name in valid_names:
                for loss in evals_results[valid_name]["binary_logloss"]:
                    self.run[f"{log_prefix}/loss/{valid_name}/{label_name}"].log(loss)

            self.run[f"{log_prefix}/auc/train/{label_name}"] = evaluate_auc(model, df_x_train, df_y_train)
            if ds_valid is not None:
                self.run[f"{log_prefix}/auc/valid/{label_name}"] = evaluate_auc(model, df_x_valid, df_y_valid)

    def train_with_validation(self, ds_train: LGBMDataset, ds_valid: LGBMDataset):
        self._train(ds_train, ds_valid, log_prefix="train_w_valid")

    def train_without_validation(self, ds_train: LGBMDataset) -> Dict[str, pd.Series]:
        self._train(ds_train, log_prefix="train_wo_valid")

        importance_dict = {}
        for label_name, model in self.models.items():
            importance = pd.Series(model.feature_importance("gain"), index=model.feature_name())
            importance_dict[label_name] = importance.sort_values(ascending=False)

        return importance_dict

    def predict_score(self, ds: LGBMDataset) -> pd.DataFrame:
        df_x = ds.bundle_features()
        preds = {}
        for label_name, model in self.models.items():
            preds[label_name] = model.predict(df_x).astype(np.float32)
        return pd.DataFrame(preds, index=ds.base_index)

    def save(self, output_path: str):
        assert self.models is not None and len(self.models) > 0
        with open(output_path, "wb") as f:
            pickle.dump(self.models, f)
