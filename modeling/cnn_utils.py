import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union, Generator
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import neptune.new as neptune
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class CNNDataset:
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
        return list(self.x["sequential"].keys())

    def get_lag_max(self) -> int:
        return self.lag_max

    def get_continuous_dim(self) -> int:
        return sum([v.shape[1] for v in self.x["continuous"].values()])

    def get_sequential_channels(self) -> int:
        return self.x["sequential"]["1min"].shape[1]

    def get_base_index(self) -> pd.DatetimeIndex:
        return self.base_index

    def get_labels(self, label_name: str = None) -> Union[pd.DataFrame, pd.Series]:
        if label_name is None:
            return self.y.loc[self.base_index]
        else:
            return self.y.loc[self.base_index, label_name]

    def train_test_split(self, test_proportion: float) -> Tuple["CNNDataset", "CNNDataset"]:
        assert self.y is not None
        train_size = int(len(self.base_index) * (1 - test_proportion))
        ds_train = CNNDataset(self.base_index[:train_size], self.x, self.y, self.lag_max, self.sma_window_size_center)
        ds_test = CNNDataset(self.base_index[train_size:], self.x, self.y, self.lag_max, self.sma_window_size_center)
        return ds_train, ds_test

    def calc_batch_num(self, batch_size: int) -> int:
        return int(np.ceil(len(self.base_index) / batch_size))

    def create_loader(self, batch_size: int, randomize: bool = True) -> Generator[Tuple, None, None]:
        x_seq = self.x["sequential"]
        x_cont = self.x["continuous"]

        index = self.base_index
        if randomize:
            index = index[np.random.permutation(len(index))]

        # それぞれの freq に対応する idx を予め計算しておく
        idx_dict = {}
        for freq in x_seq:
            index_freq = index.floor(pd.Timedelta(freq))
            idx_dict[freq] = x_seq[freq].index.get_indexer(index_freq)

        done_count = 0
        while done_count < len(index):
            idx_batch_dict = {
                freq: idx_dict[freq][done_count:done_count+batch_size]
                for freq in idx_dict
            }

            values_x_seq = {}
            values_x_cont = {}
            for freq in x_seq:
                idx_batch_freq = idx_batch_dict[freq]
                assert (idx_batch_freq >= self.lag_max).all()

                idx_expanded = np.stack([idx_batch_freq - lag_i for lag_i in range(1, self.lag_max + 1)], axis=1)
                v = x_seq[freq].values[idx_expanded.flatten()].reshape((len(idx_batch_freq), self.lag_max, -1))
                sma = x_seq[freq][f"sma{self.sma_window_size_center}"].values[idx_batch_freq-1]
                # shape: (batch_size, lag_max, feature_dim)
                values_x_seq[freq] = v - sma[:, np.newaxis, np.newaxis]

                values_x_cont[freq] = x_cont[freq].values[idx_batch_freq]

            values_y = self.y.values[idx_batch_dict["1min"]] if self.y is not None else None

            yield {"sequential": values_x_seq, "continuous": values_x_cont}, values_y

            done_count += batch_size


class CNNBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: int,
        out_channels_list: List[int],
        kernel_size_list: List[int],
        max_pool_list: List[bool],
        out_dim: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()

        assert len(out_channels_list) == len(kernel_size_list)

        convs = []
        size = window_size
        for out_channels, kernel_size, max_pool in zip(out_channels_list, kernel_size_list, max_pool_list):
            convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"))
            if batch_norm:
                convs.append(nn.BatchNorm1d(out_channels))
            convs.append(nn.ReLU())
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            if max_pool:
                convs.append(nn.MaxPool1d(kernel_size=2))
                size = (size + 1) // 2

            in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.fc_out = nn.Linear(out_channels * size, out_dim)

    def forward(self, t_x: torch.Tensor):
        y = self.convs(t_x)
        y = y.reshape(y.shape[0], -1)
        return self.fc_out(y)


class CNNNet(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        sequential_channels: int,
        freqs: List[str],
        window_size: int,
        out_channels_list: List[int],
        kernel_size_list: List[int],
        max_pool_list: List[bool],
        base_out_dim: int,
        hidden_dim_list: List[int],
        out_dim: int,
        cnn_batch_norm: bool,
        fc_batch_norm: bool,
        cnn_dropout: float,
        fc_dropout: float,
        **kwargs
    ):
        # TODO: 可能であれば、使用しないパラメータは渡さないようにする
        unused_keys = kwargs.keys()
        if len(unused_keys) > 0:
            warnings.warn(f"[CNNNet] unused keywords: " + ", ".join(unused_keys), stacklevel=2)

        super().__init__()

        self.convs = nn.ModuleDict({
            freq: CNNBase(
                sequential_channels,
                window_size,
                out_channels_list,
                kernel_size_list,
                max_pool_list,
                base_out_dim,
                cnn_batch_norm,
                cnn_dropout,
            )
            for freq in freqs
        })

        fc_out = []
        in_dim = base_out_dim * len(freqs) + continuous_dim
        for hidden_dim in hidden_dim_list:
            fc_out.append(nn.Linear(in_dim, hidden_dim))
            if fc_batch_norm:
                fc_out.append(nn.BatchNorm1d(hidden_dim))
            fc_out.append(nn.ReLU())
            if fc_dropout > 0:
                fc_out.append(nn.Dropout(fc_dropout))
            in_dim = hidden_dim
        fc_out.append(nn.Linear(hidden_dim, out_dim))
        self.fc_out = nn.Sequential(*fc_out)

    def forward(self, t_x: Dict) -> torch.Tensor:
        fc_in = []
        for freq in t_x["sequential"]:
            conv = self.convs[freq]
            fc_in.append(conv(t_x["sequential"][freq]))
            fc_in.append(t_x["continuous"][freq])

        y = torch.cat(fc_in, dim=1)
        return self.fc_out(y)

    @torch.no_grad()
    def predict_score(self, t_x: Dict) -> torch.Tensor:
        return torch.sigmoid(self.forward(t_x))


class CNNModel:
    def __init__(
        self,
        model_params: Dict,
        model: nn.Module,
        stats_mean: Dict,
        stats_var: Dict,
        run: neptune.Run,
    ):
        self.model_params = model_params
        self.model = model
        self.stats_mean = stats_mean
        self.stats_var = stats_var
        self.run = run
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def from_scratch(cls, model_params: Dict, run: neptune.Run):
        return cls(
            model_params=model_params,
            model=None,
            stats_mean=None,
            stats_var=None,
            run=run
        )

    @classmethod
    def from_file(cls, model_path: str):
        with open(model_path, "rb") as f:
            model_data = torch.load(f)

        model = CNNNet(**model_data["model_params"])
        model.load_state_dict(model_data["state_dict"])

        return cls(
            model_params=model_data["model_params"],
            model=model,
            stats_mean=model_data["stats_mean"],
            stats_var=model_data["stats_var"],
            run=None
        )

    def _calc_stats(self, ds: CNNDataset, batch_size: int = 2048):
        data_types = ["sequential", "continuous"]
        freqs = ds.get_freqs()

        stats_count = {data_type: {freq: 0 for freq in freqs} for data_type in data_types}
        self.stats_mean = {data_type: {freq: None for freq in freqs} for data_type in data_types}
        self.stats_var = {data_type: {freq: None for freq in freqs} for data_type in data_types}

        loader = ds.create_loader(batch_size, randomize=False)
        batch_num = ds.calc_batch_num(batch_size)
        for d_x, _ in tqdm(loader, desc="[stats]", total=batch_num):
            for data_type in data_types:
                for freq in freqs:
                    # shape: (batch_size, dim) or (batch_size, lag_max, dim)
                    data = d_x[data_type][freq]
                    # shape: (batch_size, dim) or (batch_size * lag_max, dim)
                    data = data.reshape((-1, data.shape[-1]))

                    count_add = len(data)
                    mean_add = np.mean(data, axis=0)
                    var_add = np.var(data, axis=0)

                    count_old = stats_count[data_type][freq]
                    mean_old = self.stats_mean[data_type][freq]
                    var_old = self.stats_var[data_type][freq]

                    if count_old == 0:
                        count_new = count_add
                        mean_new = mean_add
                        var_new = var_add
                    else:
                        count_new = count_old + count_add
                        mean_new = (mean_old * count_old + mean_add * count_add) / (count_old + count_add)
                        var_new = (
                            (var_old * count_old + var_add * count_add) / (count_old + count_add)
                            + ((mean_old - mean_add) ** 2) * count_old * count_add / ((count_old + count_add) ** 2)
                        )

                    stats_count[data_type][freq] = count_new
                    self.stats_mean[data_type][freq] = mean_new
                    self.stats_var[data_type][freq] = var_new

    def _normalize(self, d_x: Dict, eps: Optional[float] = 1e-6):
        d_x_norm = {"sequential": {}, "continuous": {}}

        data_types = d_x.keys()
        freqs = d_x["sequential"].keys()

        for data_type in data_types:
            for freq in freqs:
                mean = self.stats_mean[data_type][freq]
                std = self.stats_var[data_type][freq] ** 0.5
                d_x_norm[data_type][freq] = (d_x[data_type][freq] - mean) / (std + eps)

        return d_x_norm

    def _to_torch_x(self, d_x: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, torch.Tensor]:
        d_x_norm = self._normalize(d_x)
        t_x_seq = {
            # (batch_size, channel, length) の順に並び替える
            freq: torch.from_numpy(d_x_norm["sequential"][freq]).float().to(self.device).permute(0, 2, 1)
            for freq in d_x["sequential"]
        }
        t_x_cont = {
            freq: torch.from_numpy(d_x_norm["continuous"][freq]).float().to(self.device)
            for freq in d_x["continuous"]
        }
        return {"sequential": t_x_seq, "continuous": t_x_cont}

    def _to_torch_y(self, v_y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(v_y).float().to(self.device)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, pos_weight: float):
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
            pos_weight=torch.tensor(pos_weight, device=self.device),
        )

    def train(
        self,
        ds_train: CNNDataset,
        ds_valid: Optional[CNNDataset] = None,
        log_prefix: str = "train",
    ):
        self._calc_stats(ds_train)

        self.model_params.update({
            "window_size": ds_train.get_lag_max(),
            "continuous_dim": ds_train.get_continuous_dim(),
            "sequential_channels": ds_train.get_sequential_channels(),
            "freqs": ds_train.get_freqs(),
            "out_dim": len(ds_train.get_label_names()),
        })

        self.model = CNNNet(**self.model_params).to(self.device)
        if self.model_params["weight_decay"] == 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_params["learning_rate"])
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_params["learning_rate"], weight_decay=self.model_params["weight_decay"])

        for epoch in range(self.model_params["num_epochs"]):
            self.model.train()

            loader_train = ds_train.create_loader(self.model_params["batch_size"])
            batch_num = ds_train.calc_batch_num(self.model_params["batch_size"])
            loss_train = []
            for d_x, v_y in tqdm(loader_train, desc=f"[train {epoch}]", total=batch_num):
                t_x = self._to_torch_x(d_x)
                t_y = self._to_torch_y(v_y)

                pred = self.model(t_x)
                # shape: (batch_size, label_num)
                loss = self._compute_loss(pred, t_y, self.model_params["pos_weight"])

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                loss_train.append(loss.detach().cpu().numpy())

            loss_train = np.concatenate(loss_train, axis=0)
            self.run[f"{log_prefix}/loss/train/mean"].log(loss_train.mean())
            for i, label_name in enumerate(ds_train.get_label_names()):
                self.run[f"{log_prefix}/loss/train/{label_name}"].log(loss_train[:, i].mean())

            if ds_valid is not None:
                if self.model_params["eval_on_valid"]:
                    self.model.eval()

                loader_valid = ds_valid.create_loader(self.model_params["batch_size"])
                batch_num = ds_valid.calc_batch_num(self.model_params["batch_size"])
                loss_valid = []
                for d_x, v_y in tqdm(loader_valid, desc=f"[valid {epoch}]", total=batch_num):
                    t_x = self._to_torch_x(d_x)
                    t_y = self._to_torch_y(v_y)

                    with torch.no_grad():
                        pred = self.model(t_x)
                        loss = self._compute_loss(pred, t_y, self.model_params["pos_weight"])

                    loss_valid.append(loss.cpu().numpy())

                loss_valid = np.concatenate(loss_valid, axis=0)
                self.run[f"{log_prefix}/loss/valid/mean"].log(loss_valid.mean())
                for i, label_name in enumerate(ds_valid.get_label_names()):
                    self.run[f"{log_prefix}/loss/valid/{label_name}"].log(loss_valid[:, i].mean())

        def log_auc(ds: CNNDataset, log_suffix: str):
            pred_df = self.predict_score(ds)
            for label_name in ds.get_label_names():
                label = ds.get_labels(label_name).values
                pred = pred_df[label_name].values
                self.run[f"{log_prefix}/auc/roc/{log_suffix}/{label_name}"] = roc_auc_score(label, pred)
                self.run[f"{log_prefix}/auc/pr/{log_suffix}/{label_name}"] = average_precision_score(label, pred)

        log_auc(ds_train, log_suffix="train")
        if ds_valid is not None:
            log_auc(ds_valid, log_suffix="valid")

    def predict_score(self, ds: CNNDataset, eval: bool = True) -> pd.DataFrame:
        self.model.train(not eval)

        loader = ds.create_loader(self.model_params["batch_size"], randomize=False)
        batch_num = ds.calc_batch_num(self.model_params["batch_size"])
        preds = []
        for d_x, _ in tqdm(loader, desc="[predict score]", total=batch_num):
            t_x = self._to_torch_x(d_x)
            preds.append(self.model.predict_score(t_x).cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        return pd.DataFrame(preds, index=ds.get_base_index(), columns=ds.get_label_names())

    def save(self, output_path: str):
        model_data = {
            "model_params": self.model_params,
            "state_dict": self.model.state_dict(),
            "stats_mean": self.stats_mean,
            "stats_var": self.stats_var,
        }
        torch.save(model_data, output_path)
