import os
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import NeptuneLogger
from omegaconf import OmegaConf

from auto_trader.common import utils
from auto_trader.modeling import data, model
from auto_trader.modeling.config import TrainConfig


def main(config: TrainConfig) -> None:
    logger = NeptuneLogger(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["train", "cnn"],
    )
    logger.experiment["config"] = OmegaConf.to_yaml(config)

    # データ読み込み
    df = data.read_cleansed_data(
        config.data.symbol,
        config.data.yyyymm_begin,
        config.data.yyyymm_end,
        config.data.cleansed_data_dir,
    )
    df_base = data.merge_bid_ask(df)

    # 学習データを準備
    features = {}
    for timeframe in config.feature.timeframes:
        df_resampled = data.resample(df_base, timeframe)
        features[timeframe] = data.create_features(
            df_resampled,
            base_timing=config.feature.base_timing,
            sma_window_sizes=config.feature.sma_window_sizes,
            sma_window_size_center=config.feature.sma_window_size_center,
            sigma_window_sizes=config.feature.sigma_window_sizes,
            sma_frac_unit=config.feature.sma_frac_unit,
        )

    lift = data.calc_lift(df_base["close"], config.lift.target_alpha)

    base_index = data.calc_available_index(
        features,
        lift,
        config.feature.hist_len,
        config.feature.start_hour,
        config.feature.end_hour,
    )
    print(f"Train period: {base_index[0]} ~ {base_index[-1]}")

    total_size = len(base_index)
    valid_size = int(total_size * config.valid_ratio)
    train_size = total_size - valid_size
    logger.experiment["data/size/total"] = total_size
    logger.experiment["data/size/train"] = train_size
    logger.experiment["data/size/valid"] = valid_size
    logger.experiment["data/first_timestamp"] = str(base_index[0])
    logger.experiment["data/last_timestamp"] = str(base_index[-1])

    idxs_permuted = np.random.permutation(total_size)
    base_index_train = base_index[idxs_permuted[:train_size]]
    base_index_valid = base_index[idxs_permuted[train_size:]]
    loader_train = data.DataLoader(
        base_index=base_index_train,
        features=features,
        lift=lift,
        hist_len=config.feature.hist_len,
        sma_window_size_center=config.feature.sma_window_size_center,
        batch_size=config.batch_size,
        shuffle=True,
    )
    loader_valid = data.DataLoader(
        base_index=base_index_valid,
        features=features,
        lift=lift,
        hist_len=config.feature.hist_len,
        sma_window_size_center=config.feature.sma_window_size_center,
        batch_size=config.batch_size,
    )

    feature_info = data.get_feature_info(loader_train)

    net = model.Net(
        feature_info=feature_info,
        window_size=config.feature.hist_len,
        numerical_emb_dim=config.net.numerical_emb_dim,
        categorical_emb_dim=config.net.categorical_emb_dim,
        base_cnn_out_channels=config.net.base_cnn_out_channels,
        base_cnn_kernel_sizes=config.net.base_cnn_kernel_sizes,
        base_cnn_batchnorm=config.net.base_cnn_batchnorm,
        base_cnn_dropout=config.net.base_cnn_dropout,
        base_fc_hidden_dims=config.net.base_fc_hidden_dims,
        base_fc_batchnorm=config.net.base_fc_batchnorm,
        base_fc_dropout=config.net.base_fc_dropout,
        base_fc_output_dim=config.net.base_fc_output_dim,
        head_hidden_dims=config.net.head_hidden_dims,
        head_batchnorm=config.net.head_batchnorm,
        head_dropout=config.net.head_dropout,
    )
    model_ = model.Model(
        net,
        entropy_coef=config.loss.entropy_coef,
        spread=config.loss.spread,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )

    print("Train")
    trainer = pl.Trainer(
        max_epochs=config.max_epochs, logger=logger, enable_checkpointing=False
    )
    trainer.fit(
        model=model_,
        train_dataloaders=loader_train,
        val_dataloaders=loader_valid,
    )

    print("Save model")
    os.makedirs(config.output_dir, exist_ok=True)
    params_file = os.path.join(config.output_dir, "params.pt")
    torch.save(
        {
            "config": OmegaConf.to_container(config),
            "feature_info": feature_info,
            "net_state": net.state_dict(),
        },
        params_file,
    )
    logger.experiment["params_file"].upload(params_file)


if __name__ == "__main__":
    base_config = OmegaConf.structured(TrainConfig)
    cli_config = OmegaConf.from_cli()
    config = cast(TrainConfig, OmegaConf.merge(base_config, cli_config))
    print(OmegaConf.to_yaml(config))

    utils.set_random_seed(config.random_seed)
    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
