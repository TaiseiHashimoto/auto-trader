from typing import Any, Optional, cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import LRSchedulerConfig
from numpy.typing import NDArray

from auto_trader.modeling import data


class PeriodicActivation(nn.Module):
    def __init__(self, num_coefs: int, sigma: float) -> None:
        super().__init__()
        self.params = nn.Parameter(cast(torch.Tensor, torch.randn(num_coefs) * sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        # (...batch, num_coefs)
        x = x * self.params * 2 * np.pi
        # (...batch, num_coefs*2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def build_fc_layer(
    input_dim: int,
    hidden_dims: list[int],
    batchnorm: bool,
    dropout: float,
    output_dim: Optional[int] = None,
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

    if output_dim is not None:
        layers.append(nn.Linear(input_dim, output_dim))

    return nn.Sequential(*layers)


class BlockNet(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]],
        qkv_kernel_size: int,
        ff_kernel_size: int,
        channels: int,
        ff_channels: int,
        dropout: float,
    ):
        super().__init__()

        self.conv_qkv = nn.ModuleDict(
            {
                timeframe: nn.Conv1d(
                    channels,
                    channels * 3,
                    qkv_kernel_size,
                    padding=qkv_kernel_size // 2,
                )
                for timeframe in feature_info
            }
        )
        self.linear_q = nn.Linear(channels, channels)
        self.conv_ff = nn.ModuleDict(
            {
                timeframe: nn.Sequential(
                    nn.Conv1d(
                        channels,
                        ff_channels,
                        ff_kernel_size,
                        padding=ff_kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        ff_channels,
                        channels,
                        ff_kernel_size,
                        padding=ff_kernel_size // 2,
                    ),
                )
                for timeframe in feature_info
            }
        )
        self.linear_ff = nn.Sequential(
            nn.Linear(channels, ff_channels),
            nn.ReLU(),
            nn.Linear(ff_channels, channels),
        )

        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        self.layernorm1_cls = nn.LayerNorm(channels)
        self.layernorm2_cls = nn.LayerNorm(channels)

    def forward(
        self,
        inp: dict[data.Timeframe, torch.Tensor],
        cls: torch.Tensor,
    ) -> tuple[dict[data.Timeframe, torch.Tensor], torch.Tensor]:
        query_dict = {}
        key_dict = {}
        value_dict = {}
        for timeframe in inp:
            q, k, v = torch.chunk(self.conv_qkv[timeframe](inp[timeframe]), 3, dim=1)
            # (batch, channel, hist_len)
            query_dict[timeframe] = q
            key_dict[timeframe] = k
            value_dict[timeframe] = v

        query_dict["cls"] = self.linear_q(cls).unsqueeze(dim=2)

        # (batch, channels, num_timeframes * hist_len + 1)
        query = torch.cat(list(query_dict.values()), dim=2)
        # (batch, channels, num_timeframes * hist_len)
        key = torch.cat(list(key_dict.values()), dim=2)
        value = torch.cat(list(value_dict.values()), dim=2)

        # (batch, num_timeframes * hist_len + 1, num_timeframes * hist_len)
        attn_logits = torch.matmul(query.transpose(1, 2), key) / np.sqrt(query.shape[1])
        attention = F.softmax(attn_logits, dim=2)
        # (batch, num_timeframes * hist_len + 1, channels)
        attn_out = torch.matmul(attention, value.transpose(1, 2))
        # (batch, channels, num_timeframes * hist_len + 1)
        attn_out = attn_out.transpose(1, 2)
        # num_timeframes * (batch, channels, hist_len)
        attn_out_dict = dict(
            zip(inp.keys(), torch.chunk(attn_out[:, :, :-1], len(inp), dim=2))
        )
        # (batch, channels)
        attn_out_cls = attn_out[:, :, -1]

        y = {}
        for timeframe in inp:
            x = inp[timeframe] + self.dropout(attn_out_dict[timeframe])
            x = self.layernorm1(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(self.conv_ff[timeframe](x))
            y[timeframe] = self.layernorm2(x.transpose(1, 2)).transpose(1, 2)

        x = cls + self.dropout(attn_out_cls)
        x = self.layernorm1_cls(x)
        x = x + self.dropout(self.linear_ff(x))
        y_cls = self.layernorm2_cls(x)

        return y, y_cls


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, length: int):
        super().__init__()
        assert dim % 2 == 0

        pe = torch.zeros(dim, length)
        position = torch.arange(0, length, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[0::2] = torch.sin(position * div_term.unsqueeze(dim=1))
        pe[1::2] = torch.cos(position * div_term.unsqueeze(dim=1))

        self.pe: torch.Tensor
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return x


class Net(nn.Module):
    def __init__(
        self,
        symbol_num: int,
        feature_info: dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]],
        hist_len: int,
        numerical_emb_dim: int,
        periodic_activation_num_coefs: int,
        periodic_activation_sigma: float,
        categorical_emb_dim: int,
        emb_kernel_size: int,
        num_blocks: int,
        block_qkv_kernel_size: int,
        block_ff_kernel_size: int,
        block_channels: int,
        block_ff_channels: int,
        block_dropout: float,
        head_hidden_dims: list[int],
        head_batchnorm: bool,
        head_dropout: float,
        head_output_dim: int,
    ):
        super().__init__()

        self.emb_featurewise = nn.ModuleDict()
        self.emb_conv = nn.ModuleDict()
        for timeframe in feature_info:
            emb_total_dim = 0
            for name, info in feature_info[timeframe].items():
                if info.dtype == np.float32:
                    self.emb_featurewise[f"{timeframe}_{name}"] = nn.Sequential(
                        PeriodicActivation(
                            periodic_activation_num_coefs, periodic_activation_sigma
                        ),
                        nn.Linear(periodic_activation_num_coefs * 2, numerical_emb_dim),
                        nn.ReLU(),
                    )
                    emb_total_dim += numerical_emb_dim
                elif info.dtype == np.int64:
                    self.emb_featurewise[f"{timeframe}_{name}"] = nn.Embedding(
                        # max + 1 を OOV token とする
                        num_embeddings=info.max + 2,
                        embedding_dim=categorical_emb_dim,
                    )
                    emb_total_dim += categorical_emb_dim

            self.emb_conv[timeframe] = nn.Conv1d(
                emb_total_dim,
                block_channels,
                emb_kernel_size,
                padding=emb_kernel_size // 2,
            )

        self.positional_encoding = PositionalEncoding(block_channels, hist_len)
        self.emb_cls = nn.Parameter(torch.randn(block_channels))
        self.block_nets = nn.ModuleList(
            [
                BlockNet(
                    feature_info=feature_info,
                    qkv_kernel_size=block_qkv_kernel_size,
                    ff_kernel_size=block_ff_kernel_size,
                    channels=block_channels,
                    ff_channels=block_ff_channels,
                    dropout=block_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.symbol_emb = nn.Embedding(symbol_num, block_channels)
        self.head = build_fc_layer(
            input_dim=block_channels,
            hidden_dims=head_hidden_dims,
            batchnorm=head_batchnorm,
            dropout=head_dropout,
            output_dim=head_output_dim,
        )

    def forward(
        self,
        symbol_idx: torch.Tensor,
        features: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]],
    ) -> torch.Tensor:
        emb_dict = {}
        batch_size = 0
        for timeframe in features:
            # (batch, hist_len, emb_total_dim)
            x = torch.cat(
                [
                    self.emb_featurewise[f"{timeframe}_{feature_name}"](
                        features[timeframe][feature_name]
                    )
                    for feature_name in features[timeframe]
                ],
                dim=2,
            )
            # (batch, block_channels, hist_len)
            x = self.emb_conv[timeframe](x.transpose(1, 2))
            emb_dict[timeframe] = self.positional_encoding(x)
            batch_size = x.shape[0]

        emb_cls = torch.tile(self.emb_cls, dims=(batch_size, 1))
        for net in self.block_nets:
            emb_dict, emb_cls = net(emb_dict, emb_cls)

        # (batch, block_channels)
        x = emb_cls + self.symbol_emb(symbol_idx)
        return cast(torch.Tensor, self.head(x))


class Model(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        boundary: float,
        temperature: float = 0.1,
        canonical_batch_size: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        cosine_decay_steps: int = 0,
        cosine_decay_min: float = 0.01,
        log_stdout: bool = False,
    ):
        super().__init__()
        self.net = net

        self.boundary = boundary
        self.temperature = temperature
        self.canonical_batch_size = canonical_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cosine_decay_steps = cosine_decay_steps
        self.cosine_decay_min = cosine_decay_min
        self.log_stdout = log_stdout

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerConfig]]:
        optimizer: torch.optim.Optimizer
        if self.weight_decay == 0:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        scheduler_config = cast(
            LRSchedulerConfig,
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.cosine_decay_steps if self.cosine_decay_steps > 0 else 1,
                    eta_min=self.learning_rate
                    * (self.cosine_decay_min if self.cosine_decay_steps > 0 else 1.0),
                ),
                "interval": "step",
                "frequency": 1,
            },
        )

        return [optimizer], [scheduler_config]

    def _to_torch_features(
        self,
        features_np: dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
    ) -> dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]]:
        features_torch: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]] = {}

        for timeframe in features_np:
            features_torch[timeframe] = {}
            for feature_name, value_np in features_np[timeframe].items():
                if value_np.dtype == np.float32:
                    # shape: (batch, hist_len, feature=1)
                    features_torch[timeframe][feature_name] = torch.unsqueeze(
                        torch.from_numpy(features_np[timeframe][feature_name]),
                        dim=2,
                    ).to(self.device)
                elif value_np.dtype == np.int64:
                    # shape: (batch, hist_len)
                    features_torch[timeframe][feature_name] = torch.from_numpy(
                        features_np[timeframe][feature_name]
                    ).to(self.device)
                else:
                    raise ValueError(
                        f"Data type of {timeframe} {feature_name} is not supported: "
                        f"{value_np.dtype}"
                    )

        return features_torch

    def _predict_logits(
        self,
        symbol_idx: torch.Tensor,
        features: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]],
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.net(symbol_idx, features))

    def _predict_score(
        self,
        symbol_idx: torch.Tensor,
        features: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]],
    ) -> torch.Tensor:
        logit = self._predict_logits(symbol_idx, features)
        prob = torch.softmax(logit, dim=1)
        return cast(
            torch.Tensor, prob[:, 0] * -self.boundary + prob[:, 2] * self.boundary
        )

    def _calc_loss(
        self,
        logit: torch.Tensor,
        lift: torch.Tensor,
        log_prefix: str,
    ) -> torch.Tensor:
        soft_label = F.softmax(
            torch.stack(
                [
                    cast(torch.Tensor, -lift - self.boundary),
                    torch.zeros_like(lift),
                    cast(torch.Tensor, lift - self.boundary),
                ],
                dim=1,
            ),
            dim=1,
        )
        loss = -(logit * soft_label).sum(dim=1).mean()

        with torch.no_grad():
            prob = torch.softmax(logit, dim=1)

        self.log_dict(
            {
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/prob_lowest": prob[:, 0].mean(),
                f"{log_prefix}/prob_highest": prob[:, -1].mean(),
            },
            # Accumulate metrics on epoch level
            on_step=False,
            on_epoch=True,
            batch_size=logit.shape[0],
        )

        # バッチサイズに合わせて gradient をスケーリングする
        batch_size_ratio = logit.shape[0] / self.canonical_batch_size
        return cast(torch.Tensor, loss * batch_size_ratio)

    def training_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        symbol_idx_np, features_np, lift_np = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        lift_torch = torch.from_numpy(lift_np).to(self.device)
        return self._calc_loss(
            self._predict_logits(symbol_idx_torch, features_torch),
            lift_torch,
            log_prefix="train",
        )

    def validation_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        symbol_idx_np, features_np, lift_np = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        lift_torch = torch.from_numpy(lift_np).to(self.device)
        # ロギング目的
        _ = self._calc_loss(
            self._predict_logits(symbol_idx_torch, features_torch),
            lift_torch,
            log_prefix="valid",
        )

    def predict_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            None,
        ],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        symbol_idx_np, features_np, _ = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        return self._predict_score(symbol_idx_torch, features_torch)

    def on_train_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Training metrics: {metrics}")

    def on_validation_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Validation metrics: {metrics}")
