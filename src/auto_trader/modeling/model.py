from typing import Callable, cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from auto_trader.modeling import data


class PeriodicActivation(nn.Module):
    def __init__(self, num_coefs: int, sigma: float) -> None:
        super().__init__()
        self.params = nn.Parameter(torch.randn(num_coefs) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        # (...batch, num_coefs)
        x = x * self.params * 2 * np.pi
        # (...batch, num_coefs*2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def build_conv_layer(
    window_size: int,
    in_channel: int,
    out_channels: list[int],
    kernel_sizes: list[int],
    batchnorm: bool,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    if len(out_channels) != len(kernel_sizes):
        raise ValueError("Number of channels and kernel sizes must be the same.")

    layers: list[nn.Module] = []
    size = window_size
    for out_channel, kernel_size in zip(out_channels, kernel_sizes):
        layers.append(nn.Conv1d(in_channel, out_channel, kernel_size, padding="same"))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.MaxPool1d(kernel_size=2))

        size = size // 2
        in_channel = out_channel

    return nn.Sequential(*layers), size


def build_fc_layer(
    in_dim: int,
    hidden_dims: list[int],
    batchnorm: bool,
    dropout: float,
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class BaseNet(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.FeatureName, data.FeatureInfo],
        window_size: int,
        numerical_emb_dim: int,
        categorical_emb_dim: int,
        periodic_activation_sigma: float,
        cnn_out_channels: list[int],
        cnn_kernel_sizes: list[int],
        cnn_batchnorm: bool,
        cnn_dropout: float,
        fc_hidden_dims: list[int],
        fc_batchnorm: bool,
        fc_dropout: float,
        output_dim: int,
    ):
        super().__init__()

        self.normalize_funs: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self.embed_layers = nn.ModuleDict()
        emb_output_dim = 0
        for name, info in feature_info.items():
            if info.dtype == np.float32:
                mean = info.mean
                std = info.var**0.5
                self.normalize_funs[name] = lambda x: (x - mean) / (std + 1e-6)
                self.embed_layers[name] = PeriodicActivation(
                    numerical_emb_dim // 2, periodic_activation_sigma
                )
                emb_output_dim += numerical_emb_dim
            elif info.dtype == np.int64:
                self.normalize_funs[name] = lambda x: x
                self.embed_layers[name] = nn.Embedding(
                    num_embeddings=info.max + 1,
                    embedding_dim=categorical_emb_dim,
                )
                emb_output_dim += categorical_emb_dim

        self.conv_layer, conv_output_size = build_conv_layer(
            window_size=window_size,
            in_channel=emb_output_dim,
            out_channels=cnn_out_channels,
            kernel_sizes=cnn_kernel_sizes,
            batchnorm=cnn_batchnorm,
            dropout=cnn_dropout,
        )
        self.fc_layer = build_fc_layer(
            in_dim=cnn_out_channels[-1] * conv_output_size,
            hidden_dims=fc_hidden_dims,
            batchnorm=fc_batchnorm,
            dropout=fc_dropout,
            output_dim=output_dim,
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_tensors = []
        for name, value in features.items():
            value_norm = self.normalize_funs[name](value)
            embedded_tensors.append(self.embed_layers[name](value_norm))

        # (batch, length, channel)
        x = torch.cat(embedded_tensors, dim=2)
        # (batch, channel, length)
        x = x.permute(0, 2, 1)
        # (batch, channel, conv_output_size)
        x = self.conv_layer(x)
        # (batch, channel * conv_output_size)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(x)
        # (batch, output_dim)
        return cast(torch.Tensor, self.fc_layer(x))


class Net(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]],
        window_size: int,
        numerical_emb_dim: int,
        categorical_emb_dim: int,
        periodic_activation_sigma: float,
        base_cnn_out_channels: list[int],
        base_cnn_kernel_sizes: list[int],
        base_cnn_batchnorm: bool,
        base_cnn_dropout: float,
        base_fc_hidden_dims: list[int],
        base_fc_batchnorm: bool,
        base_fc_dropout: float,
        base_fc_output_dim: int,
        head_hidden_dims: list[int],
        head_batchnorm: bool,
        head_dropout: float,
    ):
        super().__init__()

        self.base_nets = nn.ModuleDict()
        for timeframe in feature_info:
            self.base_nets[timeframe] = BaseNet(
                feature_info[timeframe],
                window_size=window_size,
                numerical_emb_dim=numerical_emb_dim,
                categorical_emb_dim=categorical_emb_dim,
                periodic_activation_sigma=periodic_activation_sigma,
                cnn_out_channels=base_cnn_out_channels,
                cnn_kernel_sizes=base_cnn_kernel_sizes,
                cnn_batchnorm=base_cnn_batchnorm,
                cnn_dropout=base_cnn_dropout,
                fc_hidden_dims=base_fc_hidden_dims,
                fc_batchnorm=base_fc_batchnorm,
                fc_dropout=base_fc_dropout,
                output_dim=base_fc_output_dim,
            )

        self.head = build_fc_layer(
            in_dim=base_fc_output_dim * len(feature_info),
            hidden_dims=head_hidden_dims,
            batchnorm=head_batchnorm,
            dropout=head_dropout,
            output_dim=4,
        )

    def forward(self, features: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        base_outputs = []
        for timeframe, base_net in self.base_nets.items():
            base_outputs.append(base_net(features[timeframe]))

        x = torch.cat(base_outputs, dim=1)
        x = F.relu(x)
        return cast(torch.Tensor, self.head(x))


class Model(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        entropy_coef: float = 0.01,
        spread: float = 2.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        log_stdout: bool = False,
    ):
        super().__init__()
        self.net = net
        self.entropy_coef = entropy_coef
        self.spread = spread
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_stdout = log_stdout

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.weight_decay == 0:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

    def _to_torch_features(
        self,
        features_np: dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
    ) -> dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]]:
        features_torch: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]] = {}

        for timeframe in features_np:
            features_torch[timeframe] = {}
            for feature_name, value_np in features_np[timeframe].items():
                if value_np.dtype == np.float32:
                    # shape: (batch, length, feature=1)
                    features_torch[timeframe][feature_name] = torch.unsqueeze(
                        torch.from_numpy(features_np[timeframe][feature_name]),
                        dim=2,
                    ).to(self.device)
                elif value_np.dtype == np.int64:
                    # shape: (batch, length)
                    features_torch[timeframe][feature_name] = torch.from_numpy(
                        features_np[timeframe][feature_name]
                    ).to(self.device)
                else:
                    raise ValueError(
                        f"Data type of {timeframe} {feature_name} is not supported: "
                        f"{value_np.dtype}"
                    )

        return features_torch

    def _to_torch_lift(self, lift: NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(lift).to(self.device)

    def _predict_prob(
        self, features_torch: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.net(features_torch)
        prob_long_entry = torch.sigmoid(pred[:, 0])
        # prob_exit = 1 - prob_hold, prob_hold > prob_entry
        prob_long_exit = 1 - (torch.sigmoid(pred[:, 0] + F.softplus(pred[:, 1])))
        prob_short_entry = torch.sigmoid(pred[:, 2])
        prob_short_exit = 1 - (torch.sigmoid(pred[:, 2] + F.softplus(pred[:, 3])))
        return prob_long_entry, prob_long_exit, prob_short_entry, prob_short_exit

    def _calc_binary_entropy(self, prob: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            -(prob * torch.log(prob + 1e-9) + (1 - prob) * torch.log(1 - prob + 1e-9)),
        )

    def _calc_loss(
        self,
        prob_long_entry: torch.Tensor,
        prob_long_exit: torch.Tensor,
        prob_short_entry: torch.Tensor,
        prob_short_exit: torch.Tensor,
        lift_torch: torch.Tensor,
        log_prefix: str,
    ) -> torch.Tensor:
        gain_long_entry = (prob_long_entry * (lift_torch - self.spread)).mean()
        gain_long_exit = (prob_long_exit * -lift_torch).mean()
        gain_short_entry = (prob_short_entry * (-lift_torch - self.spread)).mean()
        gain_short_exit = (prob_short_exit * lift_torch).mean()
        entropy_long_entry = self._calc_binary_entropy(prob_long_entry).mean()
        entropy_long_exit = self._calc_binary_entropy(prob_long_exit).mean()
        entropy_short_entry = self._calc_binary_entropy(prob_short_entry).mean()
        entropy_short_exit = self._calc_binary_entropy(prob_short_exit).mean()
        gain = gain_long_entry + gain_long_exit + gain_short_entry + gain_short_exit
        entropy = self.entropy_coef * (
            entropy_long_entry
            + entropy_long_exit
            + entropy_short_entry
            + entropy_short_exit
        )
        loss = -(gain + entropy)

        self.log_dict(
            {
                f"{log_prefix}/prob_long_entry": prob_long_entry.mean(),
                f"{log_prefix}/prob_long_exit": prob_long_exit.mean(),
                f"{log_prefix}/prob_short_entry": prob_short_entry.mean(),
                f"{log_prefix}/prob_short_exit": prob_short_exit.mean(),
                f"{log_prefix}/gain_long_entry": gain_long_entry,
                f"{log_prefix}/gain_long_exit": gain_long_exit,
                f"{log_prefix}/gain_short_entry": gain_short_entry,
                f"{log_prefix}/gain_short_exit": gain_short_exit,
                f"{log_prefix}/entropy_long_entry": entropy_long_entry,
                f"{log_prefix}/entropy_long_exit": entropy_long_exit,
                f"{log_prefix}/entropy_short_entry": entropy_short_entry,
                f"{log_prefix}/entropy_short_exit": entropy_short_exit,
                f"{log_prefix}/gain": gain,
                f"{log_prefix}/entropy": entropy,
                f"{log_prefix}/loss": loss,
            },
            # Accumulate metrics on epoch level
            on_step=False,
            on_epoch=True,
            batch_size=prob_long_entry.shape[0],
        )

        return loss

    def training_step(
        self,
        batch: tuple[
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        features_np, lift_np = batch
        features_torch = self._to_torch_features(features_np)
        lift_torch = self._to_torch_lift(lift_np)
        return self._calc_loss(
            *self._predict_prob(features_torch), lift_torch, log_prefix="train"
        )

    def validation_step(
        self,
        batch: tuple[
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        batch_idx: int,
    ) -> None:
        features_np, lift_np = batch
        features_torch = self._to_torch_features(features_np)
        lift_torch = self._to_torch_lift(lift_np)
        # ロギング目的
        _ = self._calc_loss(
            *self._predict_prob(features_torch), lift_torch, log_prefix="valid"
        )

    def predict_step(
        self,
        batch: tuple[
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]], None
        ],
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features_np, _ = batch
        features_torch = self._to_torch_features(features_np)
        return self._predict_prob(features_torch)

    def on_train_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Training metrics: {metrics}")

    def on_validation_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Validation metrics: {metrics}")
