from typing import Optional, cast

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
        self.params = nn.Parameter(cast(torch.Tensor, torch.randn(num_coefs) * sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        # (...batch, num_coefs)
        x = x * self.params * 2 * np.pi
        # (...batch, num_coefs*2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: list[int],
    ):
        super().__init__()

        self.conv_bottleneck = nn.Conv1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            padding="same",
        )

        self.convs_main = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.convs_main.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )

        self.conv_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
        )

        self.batchnorm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))

        self._output_channels = out_channels * (len(kernel_sizes) + 1)

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.conv_bottleneck(x)
        x_maxpool = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)

        x_list = []
        for conv in self.convs_main:
            x_list.append(conv(x_bottleneck))
        x_list.append(self.conv_maxpool(x_maxpool))

        return cast(torch.Tensor, self.batchnorm(torch.concat(x_list, dim=1)))


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels=out_channels, kernel_size=1, padding="same"
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.batchnorm(self.conv(x)))


class Extractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: list[int],
        num_blocks: int,
        residual: bool,
        lstm_hidden_size: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()
        channels = in_channels
        for i in range(num_blocks):
            block = InceptionBlock(
                in_channels=channels,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels,
                kernel_sizes=kernel_sizes,
            )
            self.blocks.append(block)
            channels = block.output_channels

            if residual and i > 0:
                self.shortcut_layers.append(
                    ShortcutLayer(in_channels=in_channels, out_channels=channels)
                )

        self.lstm = nn.LSTM(
            input_size=channels, hidden_size=lstm_hidden_size, batch_first=True
        )

        self._output_dim = lstm_hidden_size

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, length, channel) -> (batch, channel, length)
        x = x.permute(0, 2, 1)

        x_ = x
        for i in range(len(self.blocks)):
            x_ = F.relu(self.blocks[i](x_))
            if len(self.shortcut_layers) > 0 and i > 0:
                x_ = F.relu(x_ + self.shortcut_layers[i - 1](x))

        # (batch, channel, length) -> (batch, length, channel)
        x = x_.permute(0, 2, 1)
        # (batch, length, channel) -> (batch, hidden_size)
        x = self.lstm(x)[0][:, -1]
        return x


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


class BaseNet(nn.Module):
    def __init__(
        self,
        feature_info: dict[data.FeatureName, data.FeatureInfo],
        numerical_emb_dim: int,
        periodic_activation_num_coefs: int,
        periodic_activation_sigma: float,
        categorical_emb_dim: int,
        inception_out_channels: int,
        inception_bottleneck_channels: int,
        inception_kernel_sizes: list[int],
        inception_num_blocks: int,
        inception_residual: bool,
        lstm_hidden_size: int,
        fc_hidden_dims: list[int],
        fc_batchnorm: bool,
        fc_dropout: float,
    ):
        super().__init__()

        self.emb_layers = nn.ModuleDict()
        emb_total_dim = 0
        for name, info in feature_info.items():
            if info.dtype == np.float32:
                self.emb_layers[name] = nn.Sequential(
                    PeriodicActivation(
                        periodic_activation_num_coefs, periodic_activation_sigma
                    ),
                    nn.Linear(periodic_activation_num_coefs * 2, numerical_emb_dim),
                    nn.ReLU(),
                )
                emb_total_dim += numerical_emb_dim
            elif info.dtype == np.int64:
                self.emb_layers[name] = nn.Embedding(
                    # max + 1 を OOV token とする
                    num_embeddings=info.max + 2,
                    embedding_dim=categorical_emb_dim,
                )
                emb_total_dim += categorical_emb_dim

        self.extractor = Extractor(
            in_channels=emb_total_dim,
            out_channels=inception_out_channels,
            bottleneck_channels=inception_bottleneck_channels,
            kernel_sizes=inception_kernel_sizes,
            num_blocks=inception_num_blocks,
            residual=inception_residual,
            lstm_hidden_size=lstm_hidden_size,
        )
        self.fc_layer = build_fc_layer(
            input_dim=self.extractor.output_dim,
            hidden_dims=fc_hidden_dims,
            batchnorm=fc_batchnorm,
            dropout=fc_dropout,
        )

        if len(fc_hidden_dims) > 0:
            self._output_dim = fc_hidden_dims[-1]
        else:
            self._output_dim = self.extractor.output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, features: dict[data.FeatureName, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for name, value in features.items():
            embeddings.append(self.emb_layers[name](value))

        # (batch, length, emb_total_dim)
        x = torch.cat(embeddings, dim=2)
        # (batch, extractor.output_dim)
        x = self.extractor(x)
        # (batch, output_dim)
        x = self.fc_layer(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        symbol_num: int,
        feature_info: dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]],
        numerical_emb_dim: int,
        periodic_activation_num_coefs: int,
        periodic_activation_sigma: float,
        categorical_emb_dim: int,
        inception_out_channels: int,
        inception_bottleneck_channels: int,
        inception_kernel_sizes: list[int],
        inception_num_blocks: int,
        inception_residual: bool,
        lstm_hidden_size: int,
        base_fc_hidden_dims: list[int],
        base_fc_batchnorm: bool,
        base_fc_dropout: float,
        head_hidden_dims: list[int],
        head_batchnorm: bool,
        head_dropout: float,
        head_output_dim: int,
    ):
        super().__init__()

        self.base_nets = nn.ModuleDict()
        base_output_dim_total = 0
        for timeframe in feature_info:
            base_net = BaseNet(
                feature_info=feature_info[timeframe],
                numerical_emb_dim=numerical_emb_dim,
                periodic_activation_num_coefs=periodic_activation_num_coefs,
                periodic_activation_sigma=periodic_activation_sigma,
                categorical_emb_dim=categorical_emb_dim,
                inception_out_channels=inception_out_channels,
                inception_bottleneck_channels=inception_bottleneck_channels,
                inception_kernel_sizes=inception_kernel_sizes,
                inception_num_blocks=inception_num_blocks,
                inception_residual=inception_residual,
                lstm_hidden_size=lstm_hidden_size,
                fc_hidden_dims=base_fc_hidden_dims,
                fc_batchnorm=base_fc_batchnorm,
                fc_dropout=base_fc_dropout,
            )
            self.base_nets[timeframe] = base_net
            base_output_dim_total += base_net.output_dim

        self.symbol_emb = nn.Embedding(symbol_num, categorical_emb_dim)
        self.head = build_fc_layer(
            input_dim=base_output_dim_total + categorical_emb_dim,
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
        base_outputs = []
        for timeframe, base_net in self.base_nets.items():
            base_outputs.append(base_net(features[timeframe]))

        x = F.relu(torch.cat(base_outputs, dim=1))
        x = torch.cat([x, self.symbol_emb(symbol_idx)], dim=1)
        return cast(torch.Tensor, self.head(x))


class Model(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        bucket_boundaries: list[float],
        label_smoothing: float = 0.0,
        canonical_batch_size: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        log_stdout: bool = False,
    ):
        super().__init__()
        self.net = net

        self.bucket_boundaries = torch.tensor(bucket_boundaries, dtype=torch.float32)
        bucket_centers = [bucket_boundaries[0]]
        for left, right in zip(bucket_boundaries[:-1], bucket_boundaries[1:]):
            bucket_centers.append((left + right) / 2)
        bucket_centers.append(bucket_boundaries[-1])
        self.bucket_centers = torch.tensor(bucket_centers, dtype=torch.float32)

        self.label_smoothing = label_smoothing
        self.canonical_batch_size = canonical_batch_size
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

    def _predict_logits(
        self,
        symbol_idx: torch.Tensor,
        features: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.net(symbol_idx, features)
        return cast(tuple[torch.Tensor, torch.Tensor], torch.chunk(pred, 2, dim=1))

    def _predict_probs(
        self,
        symbol_idx: torch.Tensor,
        features: dict[data.Timeframe, dict[data.FeatureName, torch.Tensor]],
    ) -> Predictions:
        self.bucket_centers = self.bucket_centers.to(self.device)
        logit_long, logit_short = self._predict_logits(symbol_idx, features)
        prob_long = torch.softmax(logit_long, dim=1)
        prob_short = torch.softmax(logit_short, dim=1)
        pred_gain_long = (prob_long * self.bucket_centers).sum(dim=1)
        pred_gain_short = (prob_short * self.bucket_centers).sum(dim=1)
        return (
            # 順番が変わらなければ sigmoid でなくてもよい
            torch.sigmoid(pred_gain_long),
            torch.sigmoid(-pred_gain_long),
            torch.sigmoid(pred_gain_short),
            torch.sigmoid(-pred_gain_short),
        )

    def _calc_loss(
        self,
        logit_long: torch.Tensor,
        logit_short: torch.Tensor,
        gain_long: torch.Tensor,
        gain_short: torch.Tensor,
        log_prefix: str,
    ) -> torch.Tensor:
        self.bucket_boundaries = self.bucket_boundaries.to(self.device)
        loss_long = F.cross_entropy(
            input=logit_long,
            target=torch.bucketize(gain_long, self.bucket_boundaries),
            label_smoothing=self.label_smoothing,
        )
        loss_short = F.cross_entropy(
            input=logit_short,
            target=torch.bucketize(gain_short, self.bucket_boundaries),
            label_smoothing=self.label_smoothing,
        )
        loss = loss_long + loss_short

        with torch.no_grad():
            prob_long = torch.softmax(logit_long, dim=1)
            prob_short = torch.softmax(logit_short, dim=1)

        self.log_dict(
            {
                f"{log_prefix}/loss_long": loss_long,
                f"{log_prefix}/loss_short": loss_short,
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/prob_long_lowest": prob_long[:, 0].mean(),
                f"{log_prefix}/prob_long_highest": prob_long[:, -1].mean(),
                f"{log_prefix}/prob_short_lowest": prob_short[:, 0].mean(),
                f"{log_prefix}/prob_short_highest": prob_short[:, -1].mean(),
            },
            # Accumulate metrics on epoch level
            on_step=False,
            on_epoch=True,
            batch_size=logit_long.shape[0],
        )

        # バッチサイズに合わせて gradient をスケーリングする
        batch_size_ratio = logit_long.shape[0] / self.canonical_batch_size
        return cast(torch.Tensor, loss * batch_size_ratio)

    def training_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            tuple[NDArray[np.float32], NDArray[np.float32]],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        symbol_idx_np, features_np, (gain_long_np, gain_short_np) = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        gain_long_torch = torch.from_numpy(gain_long_np).to(self.device)
        gain_short_torch = torch.from_numpy(gain_short_np).to(self.device)
        return self._calc_loss(
            *self._predict_logits(symbol_idx_torch, features_torch),
            gain_long_torch,
            gain_short_torch,
            log_prefix="train",
        )

    def validation_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            NDArray[np.float32],
        ],
        batch_idx: int,
    ) -> None:
        symbol_idx_np, features_np, (gain_long_np, gain_short_np) = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        gain_long_torch = torch.from_numpy(gain_long_np).to(self.device)
        gain_short_torch = torch.from_numpy(gain_short_np).to(self.device)
        # ロギング目的
        _ = self._calc_loss(
            *self._predict_logits(symbol_idx_torch, features_torch),
            gain_long_torch,
            gain_short_torch,
            log_prefix="valid",
        )

    def predict_step(
        self,
        batch: tuple[
            NDArray[np.int64],
            dict[data.Timeframe, dict[data.FeatureName, data.FeatureValue]],
            None,
        ],
        batch_idx: int,
    ) -> Predictions:
        symbol_idx_np, features_np, _ = batch
        symbol_idx_torch = torch.from_numpy(symbol_idx_np).to(self.device)
        features_torch = self._to_torch_features(features_np)
        return self._predict_probs(symbol_idx_torch, features_torch)

    def on_train_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Training metrics: {metrics}")

    def on_validation_epoch_end(self) -> None:
        if self.log_stdout:
            metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
            print(f"Validation metrics: {metrics}")
