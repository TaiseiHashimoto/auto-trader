import os
from dataclasses import asdict

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from omegaconf import OmegaConf

from auto_trader.common import utils
from auto_trader.modeling import data, model
from auto_trader.modeling.config import YYYYMM_BEGIN_MAP, TrainConfig


def main(config: TrainConfig) -> None:
    logger = NeptuneLogger(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["classification", "train", "conv-transformer"],
        prefix="",
    )
    logger.experiment["config"] = OmegaConf.to_yaml(config)

    print("Prepare data")
    feature_info_all: dict[
        data.Symbol, dict[data.Timeframe, dict[data.FeatureName, data.FeatureInfo]]
    ] = {}
    loaders_train: dict[data.Symbol, data.NormalizedLoader] = {}
    loaders_valid: dict[data.Symbol, data.NormalizedLoader] = {}
    for symbol in config.symbols:
        print(f"{symbol=}")
        yyyymm_begin = config.yyyymm_begin or YYYYMM_BEGIN_MAP[symbol]
        df = data.read_cleansed_data(
            symbol=symbol,
            yyyymm_begin=yyyymm_begin,
            yyyymm_end=config.yyyymm_end,
            cleansed_data_dir=config.cleansed_data_dir,
        )
        df_base = data.merge_bid_ask(df)

        features = {}
        for timeframe in config.feature.timeframes:
            df_resampled = data.resample(df_base, timeframe)
            features[timeframe] = data.create_features(
                df_resampled,
                base_timing=config.feature.base_timing,
                moving_window_sizes=config.feature.moving_window_sizes,
                moving_window_size_center=config.feature.moving_window_size_center,
                use_sma_frac=config.feature.use_sma_frac,
                sma_frac_unit=config.feature.sma_frac_unit,
                use_hour=config.feature.use_hour,
                use_dow=config.feature.use_dow,
            )

        lift = data.calc_lift(df_base["close"], config.lift.alpha)

        base_index = data.calc_available_index(
            features=features,
            hist_len=config.feature.hist_len,
        )
        print(f"Train period: {base_index[0]} ~ {base_index[-1]}")

        idxs_train, idxs_valid = data.split_block_idxs(
            len(base_index),
            block_size=config.valid_block_size,
            valid_ratio=config.valid_ratio,
        )
        base_index_train = base_index[idxs_train]
        base_index_valid = base_index[idxs_valid]

        raw_loader_train = data.RawLoader(
            base_index=base_index_train,
            features=features,
            lift=lift,
            hist_len=config.feature.hist_len,
            moving_window_size_center=config.feature.moving_window_size_center,
            batch_size=config.batch_size,
            shuffle=True,
        )
        raw_loader_valid = data.RawLoader(
            base_index=base_index_valid,
            features=features,
            lift=lift,
            hist_len=config.feature.hist_len,
            moving_window_size_center=config.feature.moving_window_size_center,
            batch_size=config.batch_size,
        )
        feature_info, lift_info = data.get_feature_info(raw_loader_train)
        feature_info_all[symbol] = feature_info
        normalized_loader_train = data.NormalizedLoader(
            loader=raw_loader_train,
            feature_info=feature_info,
        )
        normalized_loader_valid = data.NormalizedLoader(
            loader=raw_loader_valid,
            feature_info=feature_info,
        )
        loaders_train[symbol] = normalized_loader_train
        loaders_valid[symbol] = normalized_loader_valid

        logger.experiment[f"data/{symbol}/feature_info"] = yaml.dump(
            {
                t: {n: str(feature_info[t][n]) for n in feature_info[t]}
                for t in feature_info
            }
        )
        logger.experiment[f"data/{symbol}/lift_info"] = str(lift_info)

    size_train = sum(loader.size for loader in loaders_train.values())
    size_valid = sum(loader.size for loader in loaders_valid.values())
    logger.experiment["data/size/train"] = size_train
    logger.experiment["data/size/valid"] = size_valid
    logger.experiment["data/size/total"] = size_train + size_valid

    for symbol in config.symbols:
        loader_train = loaders_train[symbol]
        loader_valid = loaders_valid[symbol]
        loader_train.set_batch_size(
            int(config.batch_size * loader_train.size / size_train)
        )
        loader_valid.set_batch_size(
            int(config.batch_size * loader_valid.size / size_valid)
        )

    symbol_idxs = {symbol: i for i, symbol in enumerate(config.symbols)}
    combined_loader_train = data.CombinedLoader(
        loaders=loaders_train,
        key_map=symbol_idxs,
    )
    combined_loader_valid = data.CombinedLoader(
        loaders=loaders_valid,
        key_map=symbol_idxs,
    )

    net = model.Net(
        symbol_num=len(config.symbols),
        # symbol ごとに特徴量の型式は変わらないので、どれを渡しても良い
        feature_info=feature_info_all[config.symbols[0]],
        hist_len=config.feature.hist_len,
        numerical_emb_dim=config.net.numerical_emb_dim,
        periodic_activation_num_coefs=config.net.periodic_activation_num_coefs,
        periodic_activation_sigma=config.net.periodic_activation_sigma,
        categorical_emb_dim=config.net.categorical_emb_dim,
        emb_kernel_size=config.net.emb_kernel_size,
        num_blocks=config.net.num_blocks,
        block_qkv_kernel_size=config.net.block_qkv_kernel_size,
        block_ff_kernel_size=config.net.block_ff_kernel_size,
        block_channels=config.net.block_channels,
        block_ff_channels=config.net.block_ff_channels,
        block_dropout=config.net.block_dropout,
        head_hidden_dims=config.net.head_hidden_dims,
        head_batchnorm=config.net.head_batchnorm,
        head_dropout=config.net.head_dropout,
        head_output_dim=len(config.loss.bucket_boundaries) + 1,
    )
    model_ = model.Model(
        net,
        bucket_boundaries=config.loss.bucket_boundaries,
        label_smoothing=config.loss.label_smoothing,
        canonical_batch_size=config.batch_size,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        cosine_decay_steps=config.optim.cosine_decay_steps,
        cosine_decay_min=config.optim.cosine_decay_min,
        log_stdout=config.neptune.mode == "debug",
    )

    print("Train")
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss",
        mode="min",
        patience=config.early_stopping_patience,
        check_finite=True,
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/loss",
        mode="min",
        dirpath=config.output_dir,
        save_top_k=1,
        enable_version_counter=False,
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    trainer.fit(
        model=model_,
        train_dataloaders=combined_loader_train,
        val_dataloaders=combined_loader_valid,
    )

    print("Save model")
    # best epoch のパラメータを復元
    model.Model.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        net=net,
        bucket_boundaries=config.loss.bucket_boundaries,
    )
    os.makedirs(config.output_dir, exist_ok=True)
    params_file = os.path.join(config.output_dir, "params.pt")
    torch.save(
        {
            "config": asdict(config),
            "symbol_idxs": symbol_idxs,
            "feature_info_all": feature_info_all,
            "net_state": net.state_dict(),
        },
        params_file,
    )
    logger.experiment["params_file"].upload(params_file)


if __name__ == "__main__":
    config = utils.get_config(TrainConfig)
    print(OmegaConf.to_yaml(config))

    utils.set_random_seed(config.random_seed)
    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
