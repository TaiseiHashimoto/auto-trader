import os
from dataclasses import asdict
from pathlib import Path

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
    yyyymm_begin = config.yyyymm_begin or YYYYMM_BEGIN_MAP[config.symbol]
    df_cleansed = data.read_cleansed_data(
        cleansed_data_dir=Path(config.cleansed_data_dir),
        symbol=config.symbol,
        yyyymm_begin=yyyymm_begin,
        yyyymm_end=config.yyyymm_end,
    )
    df_rate = data.merge_bid_ask(df_cleansed)

    features = data.create_features(
        df_rate,
        base_timing=config.feature.base_timing,
        window_sizes=config.feature.window_sizes,
        use_fraction=config.feature.use_fraction,
        fraction_unit=config.feature.fraction_unit,
        use_hour=config.feature.use_hour,
        use_dow=config.feature.use_dow,
    )
    features = data.relativize_features(features, config.feature.base_timing)
    feature_stats = data.get_feature_stats(features)
    features = data.normalize_features(features, feature_stats)
    logger.experiment["data/feature_stats"] = yaml.dump(
        {n: str(feature_stats[n]) for n in feature_stats}
    )

    label = data.create_label(
        df_rate["close"], config.label.future_step, config.label.bin_boundary
    )
    label_stats = data.CategoricalFeatureStats(
        label.value_counts().sort_index().to_dict()
    )
    logger.experiment["data/label_stats"] = str(label_stats)

    index = data.calc_available_index(
        features=features,
        label=label,
        hist_len=config.feature.hist_len,
        hour_begin=config.feature.hour_begin,
        hour_end=config.feature.hour_end,
    )
    print(f"Train period: {index[0]} ~ {index[-1]}")

    idxs_train, idxs_valid = data.split_block_idxs(
        len(index),
        block_size=config.valid_block_size,
        valid_ratio=config.valid_ratio,
    )
    index_train = index[idxs_train]
    index_valid = index[idxs_valid]
    loader_train = data.SequentialLoader(
        available_index=index_train,
        features=features,
        label=label,
        hist_len=config.feature.hist_len,
        batch_size=config.batch_size,
        shuffle=True,
    )
    loader_valid = data.SequentialLoader(
        available_index=index_valid,
        features=features,
        label=label,
        hist_len=config.feature.hist_len,
        batch_size=config.batch_size,
    )
    logger.experiment["data/size/train"] = loader_train.size
    logger.experiment["data/size/valid"] = loader_valid.size
    logger.experiment["data/size/total"] = loader_train.size + loader_valid.size

    net = model.Net(
        feature_stats=feature_stats,
        hist_len=config.feature.hist_len,
        continuous_emb_dim=config.net.continuous_emb_dim,
        periodic_activation_num_coefs=config.net.periodic_activation_num_coefs,
        periodic_activation_sigma=config.net.periodic_activation_sigma,
        categorical_emb_dim=config.net.categorical_emb_dim,
        out_channels=config.net.out_channels,
        kernel_sizes=config.net.kernel_sizes,
        strides=config.net.strides,
        batchnorm=config.net.batchnorm,
        dropout=config.net.dropout,
        head_hidden_dims=config.net.head_hidden_dims,
        head_batchnorm=config.net.head_batchnorm,
        head_dropout=config.net.head_dropout,
        head_output_dim=label_stats.vocab_size,
    )
    model_ = model.Model(
        net,
        boundary=config.label.bin_boundary,
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
        profiler=config.lightning_profiler,
    )
    trainer.fit(
        model=model_,
        train_dataloaders=loader_train,
        val_dataloaders=loader_valid,
    )

    print("Save model")
    # best epoch のパラメータを復元
    model.Model.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        net=net,
        boundary=config.loss.boundary,
    )
    os.makedirs(config.output_dir, exist_ok=True)
    params_file = os.path.join(config.output_dir, "params.pt")
    torch.save(
        {
            "config": asdict(config),
            "feature_stats": feature_stats,
            "label_stats": label_stats,
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
