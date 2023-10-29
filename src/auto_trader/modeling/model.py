from typing import Callable, cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from auto_trader.modeling import data

Predictions = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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


class ConvExtractor(nn.Module):
    def __init__(
        self,
        hist_len: int,
        input_channel: int,
        out_channels: list[int],
        kernel_sizes: list[int],
        batchnorm: bool,
        dropout: float,
    ):
        super().__init__()

        assert len(out_channels) == len(kernel_sizes)

        self.layers = nn.Sequential()
        size = hist_len
        for out_channel, kernel_size in zip(out_channels, kernel_sizes):
            self.layers.append(
                nn.Conv1d(input_channel, out_channel, kernel_size, padding="same")
            )
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(out_channel))
            self.layers.append(nn.ReLU())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.MaxPool1d(kernel_size=2))

            size = size // 2
            input_channel = out_channel

        self._output_dim = out_channel * size

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, length, channel) -> (batch, channel, length)
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        return x


class AttentionExtractor(nn.Module):
    def __init__(
        self,
        hist_len: int,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        pe_sigma: float,
        dropout: float,
    ):
        super().__init__()

        assert emb_dim % num_heads == 0

        self.attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=emb_dim,
                    num_heads=num_heads,
                    batch_first=True,
                ),
            )
            self.linear_layers.append(nn.Linear(emb_dim, emb_dim))
            self.feedforward_layers.append(
                nn.Sequential(
                    nn.Linear(emb_dim, feedforward_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(feedforward_dim, emb_dim),
                )
            )

        self.positional_encoding = nn.Parameter(
            torch.randn(hist_len, emb_dim) * pe_sigma
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

        self._output_dim = emb_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, length, embed)
        x = x + self.positional_encoding

        for i in range(len(self.attention_layers)):
            if i < len(self.attention_layers) - 1:
                q, k, v = x, x, x
            else:
                # query last time only
                q = x[:, -1, :].unsqueeze(dim=1)
                k, v = x, x

            # (batch, length or 1, embed)
            x_ = self.attention_layers[i](q, k, v, need_weights=False)[0]
            x_ = self.linear_layers[i](x_)
            x = self.layer_norm(q + self.dropout(x_))

            x_ = self.feedforward_layers[i](x)
            x = self.layer_norm(x + self.dropout(x_))

        # (batch, embed)
        return x.squeeze(dim=1)


def build_fc_layer(
    input_dim: int,
    hidden_dims: list[int],
    batchnorm: bool,
    dropout: float,
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class BaseNet(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.FeatureName, data.FeatureInfo],
        hist_len: int,
        numerical_emb_dim: int,
        categorical_emb_dim: int,
        periodic_activation_sigma: float,
        emb_output_dim: int,
        net_type: str,
        attention_num_layers: int,
        attention_num_heads: int,
        attention_feedforward_dim: int,
        attention_pe_sigma: float,
        attention_dropout: float,
        conv_out_channels: list[int],
        conv_kernel_sizes: list[int],
        conv_batchnorm: bool,
        conv_dropout: float,
        fc_hidden_dims: list[int],
        fc_batchnorm: bool,
        fc_dropout: float,
        output_dim: int,
    ):
        super().__init__()

        self.normalize_funs: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self.embed_layers = nn.ModuleDict()
        emb_total_dim = 0
        for name, info in feature_info.items():
            if info.dtype == np.float32:
                mean = info.mean
                std = info.var**0.5
                self.normalize_funs[name] = lambda x: (x - mean) / (std + 1e-6)
                self.embed_layers[name] = PeriodicActivation(
                    numerical_emb_dim // 2, periodic_activation_sigma
                )
                emb_total_dim += numerical_emb_dim
            elif info.dtype == np.int64:
                # max + 1 を OOV token とする
                self.normalize_funs[name] = lambda x: torch.clamp(x, max=info.max + 1)
                self.embed_layers[name] = nn.Embedding(
                    num_embeddings=info.max + 2,
                    embedding_dim=categorical_emb_dim,
                )
                emb_total_dim += categorical_emb_dim

        self.fc_embed = nn.Linear(emb_total_dim, emb_output_dim)

        self.extractor: nn.Module
        if net_type == "attention":
            self.extractor = AttentionExtractor(
                hist_len=hist_len,
                emb_dim=emb_output_dim,
                num_layers=attention_num_layers,
                num_heads=attention_num_heads,
                feedforward_dim=attention_feedforward_dim,
                pe_sigma=attention_pe_sigma,
                dropout=attention_dropout,
            )
        elif net_type == "conv":
            self.extractor = ConvExtractor(
                hist_len=hist_len,
                input_channel=emb_output_dim,
                out_channels=conv_out_channels,
                kernel_sizes=conv_kernel_sizes,
                batchnorm=conv_batchnorm,
                dropout=conv_dropout,
            )
        else:
            assert False

        self.fc_layer = build_fc_layer(
            input_dim=self.extractor.get_output_dim(),
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

        # (batch, length, emb_output_dim)
        x = self.fc_embed(torch.cat(embedded_tensors, dim=2))
        # (batch, extractor_output_dim)
        x = self.extractor(x)
        # (batch, output_dim)
        x = self.fc_layer(F.relu(x))
        return cast(torch.Tensor, x)


class Net(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]],
        hist_len: int,
        numerical_emb_dim: int,
        categorical_emb_dim: int,
        periodic_activation_sigma: float,
        emb_output_dim: int,
        base_net_type: str,
        base_attention_num_layers: int,
        base_attention_num_heads: int,
        base_attention_feedforward_dim: int,
        base_attention_pe_sigma: float,
        base_attention_dropout: float,
        base_conv_out_channels: list[int],
        base_conv_kernel_sizes: list[int],
        base_conv_batchnorm: bool,
        base_conv_dropout: float,
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
                hist_len=hist_len,
                numerical_emb_dim=numerical_emb_dim,
                categorical_emb_dim=categorical_emb_dim,
                periodic_activation_sigma=periodic_activation_sigma,
                emb_output_dim=emb_output_dim,
                net_type=base_net_type,
                attention_num_layers=base_attention_num_layers,
                attention_num_heads=base_attention_num_heads,
                attention_feedforward_dim=base_attention_feedforward_dim,
                attention_pe_sigma=base_attention_pe_sigma,
                attention_dropout=base_attention_dropout,
                conv_out_channels=base_conv_out_channels,
                conv_kernel_sizes=base_conv_kernel_sizes,
                conv_batchnorm=base_conv_batchnorm,
                conv_dropout=base_conv_dropout,
                fc_hidden_dims=base_fc_hidden_dims,
                fc_batchnorm=base_fc_batchnorm,
                fc_dropout=base_fc_dropout,
                output_dim=base_fc_output_dim,
            )

        self.head = build_fc_layer(
            input_dim=base_fc_output_dim * len(feature_info),
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

    def _to_torch_gain(self, gain: NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(gain).to(self.device)

    def _predict_prob(
        self, features_torch: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]]
    ) -> Predictions:
        # (long_entry, long_exit, short_entry, short_exit) は
        # (1, 0, 0, 1), (0, 0, 0, 1), (0, 1, 0, 0),  (0, 1, 1, 0) のいずれか。
        # 各ケースの確率を p0, p1, p2, p3 とすると、
        # (prob_long_entry, prob_long_exit, prob_short_entry, prob_short_exit)
        # = (p0, p2 + p3, p3, p0 + p1)
        pred = self.net(features_torch)
        pred_norm = torch.softmax(pred, dim=1)
        prob_long_entry = pred_norm[:, 0]
        prob_long_exit = pred_norm[:, 2] + pred_norm[:, 3]
        prob_short_entry = pred_norm[:, 3]
        prob_short_exit = pred_norm[:, 0] + pred_norm[:, 1]
        return prob_long_entry, prob_long_exit, prob_short_entry, prob_short_exit

    def _calc_binary_entropy(self, prob: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            -(prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6)),
        )

    def _calc_loss(
        self,
        prob_long_entry: torch.Tensor,
        prob_long_exit: torch.Tensor,
        prob_short_entry: torch.Tensor,
        prob_short_exit: torch.Tensor,
        gain_long_torch: torch.Tensor,
        gain_short_torch: torch.Tensor,
        log_prefix: str,
    ) -> torch.Tensor:
        gain_long_entry = (prob_long_entry * (gain_long_torch - self.spread)).mean()
        gain_long_exit = (prob_long_exit * -gain_long_torch).mean()
        gain_short_entry = (prob_short_entry * (gain_short_torch - self.spread)).mean()
        gain_short_exit = (prob_short_exit * -gain_short_torch).mean()
        entropy_long_entry = self._calc_binary_entropy(prob_long_entry).mean()
        entropy_long_exit = self._calc_binary_entropy(prob_long_exit).mean()
        entropy_short_entry = self._calc_binary_entropy(prob_short_entry).mean()
        entropy_short_exit = self._calc_binary_entropy(prob_short_exit).mean()
        gain = gain_long_entry + gain_long_exit + gain_short_entry + gain_short_exit
        entropy = (
            entropy_long_entry
            + entropy_long_exit
            + entropy_short_entry
            + entropy_short_exit
        )
        loss = -(gain + self.entropy_coef * entropy)

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
            tuple[NDArray[np.float32], NDArray[np.float32]],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        features_np, (gain_long_np, gain_short_np) = batch
        features_torch = self._to_torch_features(features_np)
        gain_long_torch = self._to_torch_gain(gain_long_np)
        gain_short_torch = self._to_torch_gain(gain_short_np)
        return self._calc_loss(
            *self._predict_prob(features_torch),
            gain_long_torch,
            gain_short_torch,
            log_prefix="train",
        )

    def validation_step(
        self,
        batch: tuple[
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        batch_idx: int,
    ) -> None:
        features_np, (gain_long_np, gain_short_np) = batch
        features_torch = self._to_torch_features(features_np)
        gain_long_torch = self._to_torch_gain(gain_long_np)
        gain_short_torch = self._to_torch_gain(gain_short_np)
        # ロギング目的
        _ = self._calc_loss(
            *self._predict_prob(features_torch),
            gain_long_torch,
            gain_short_torch,
            log_prefix="valid",
        )

    def predict_step(
        self,
        batch: tuple[
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]], None
        ],
        batch_idx: int,
    ) -> Predictions:
        features_np, _ = batch
        features_torch = self._to_torch_features(features_np)
        return self._predict_prob(features_torch)

    def on_trainput_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Training metrics: {metrics}")

    def on_validation_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Validation metrics: {metrics}")
