import os
from typing import cast

import lightning.pytorch as pl
import neptune
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from auto_trader.common import utils
from auto_trader.modeling import data, model, order, strategy
from auto_trader.modeling.config import EvalConfig, TrainConfig


def get_binary_pred(pred: NDArray[np.float32], percentile: float) -> NDArray[np.bool_]:
    return pred > np.percentile(pred, percentile)


def calc_stats(values: NDArray[np.float32]) -> dict[str, float]:
    PERCENTILES = np.arange(21) * 5
    return {
        "mean": cast(float, np.mean(values)),
        **{
            f"percentile_{p}": cast(float, np.percentile(values, p))
            for p in PERCENTILES
        },
    }


def log_metrics(
    config: EvalConfig,
    lift: "pd.Series[float]",
    score: "pd.Series[float]",
    run: neptune.Run,
) -> None:
    score_np = cast(NDArray[np.float32], score.values)
    run["stats/score"] = calc_stats(score_np)

    label_long_entry = (lift.loc[score.index] > config.simulation.spread).values
    label_short_entry = (lift.loc[score.index] < -config.simulation.spread).values
    run["stats/roc_auc/long_entry"] = roc_auc_score(label_long_entry, score)
    run["stats/roc_auc/short_entry"] = roc_auc_score(label_short_entry, -score)
    run["stats/pr_auc/long_entry"] = average_precision_score(label_long_entry, score)
    run["stats/pr_auc/short_entry"] = average_precision_score(label_short_entry, -score)

    pred_long_entry = score_np > np.percentile(
        score_np, config.strategy.percentile_entry
    )
    pred_short_entry = score_np < np.percentile(
        score_np, 100 - config.strategy.percentile_entry
    )
    run["stats/precision/long_entry"] = precision_score(
        label_long_entry, pred_long_entry
    )
    run["stats/precision/short_entry"] = precision_score(
        label_short_entry, pred_short_entry
    )
    run["stats/recall/long_entry"] = recall_score(label_long_entry, pred_long_entry)
    run["stats/recall/short_entry"] = recall_score(label_short_entry, pred_short_entry)

    lift_long_entry = cast(
        NDArray[np.float32], lift.loc[score.index].loc[pred_long_entry].values
    )
    lift_short_entry = cast(
        NDArray[np.float32], lift.loc[score.index].loc[pred_short_entry].values
    )
    if len(lift_long_entry) > 0:
        run["stats/lift/long_entry"] = calc_stats(lift_long_entry)
    if len(lift_short_entry) > 0:
        run["stats/lift/short_entry"] = calc_stats(lift_short_entry)


def run_simulations(
    config: EvalConfig,
    rates: "pd.Series[float]",
    score: "pd.Series[float]",
    run: neptune.Run,
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    strategy_ = strategy.TimeLimitStrategy(
        thresh_long_entry=cast(
            float, np.percentile(score, config.strategy.percentile_entry)
        ),
        thresh_short_entry=cast(
            float, np.percentile(score, 100 - config.strategy.percentile_entry)
        ),
        max_entry_time=config.strategy.max_entry_time,
    )
    simulator = order.OrderSimulator(
        config.simulation.start_hour,
        config.simulation.end_hour,
        config.simulation.thresh_losscut,
    )
    for i, timestamp in enumerate(rates.index):
        (
            pred_long_entry,
            pred_long_exit,
            pred_short_entry,
            pred_short_exit,
        ) = strategy_.make_decision(timestamp, score.values[i])

        # TODO: 戦略 (e.g. n 回連続で long 選択の場合に実際に long する) を導入
        simulator.step(
            timestamp,
            rates.iloc[i],
            pred_long_entry,
            pred_long_exit,
            pred_short_entry,
            pred_short_exit,
        )

    results = simulator.export_results()
    results_path = os.path.join(config.output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    run["simulation/results"].upload(results_path)

    if len(results) > 0:
        profit = pd.Series(
            cast(NDArray[np.float32], results["gain"].values)
            - config.simulation.spread,
            index=results["entry_time"],
        )
        profit_per_day = profit.resample("1d").sum()
        num_order_per_day = profit.resample("1d").count()
        run["simulation/num_order_per_day"] = calc_stats(
            cast(NDArray[np.float32], num_order_per_day.values)
        )
        run["simulation/profit_per_trade"] = calc_stats(
            cast(NDArray[np.float32], profit.values)
        )
        run["simulation/profit_per_day"] = calc_stats(
            cast(NDArray[np.float32], profit_per_day.values)
        )
        duration = (
            results["exit_time"] - results["entry_time"]
        ).dt.total_seconds() / 60
        run["simulation/duration"] = calc_stats(
            cast(NDArray[np.float32], duration.values)
        )


def main(config: EvalConfig) -> None:
    run = neptune.init_run(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["classification", "eval", "conv-transformer"],
    )
    run["config"] = OmegaConf.to_yaml(config)

    print("Fetch params")
    if config.params_file == "":
        train_run = neptune.init_run(
            project=config.neptune.project,
            with_id=config.train_run_id,
            mode="read-only",
        )
        os.makedirs(config.output_dir, exist_ok=True)
        params_file = os.path.join(config.output_dir, "params.pt")
        train_run["params_file"].download(params_file)
        train_run.stop()
    else:
        params_file = config.params_file

    params = torch.load(params_file)
    train_config = cast(TrainConfig, OmegaConf.create(params["config"]))
    symbol_idxs = params["symbol_idxs"]
    feature_info_all = params["feature_info_all"]
    net_state = params["net_state"]

    df = data.read_cleansed_data(
        symbol=config.symbol,
        yyyymm_begin=config.yyyymm_begin,
        yyyymm_end=config.yyyymm_end,
        cleansed_data_dir=config.cleansed_data_dir,
    )
    df_base = data.merge_bid_ask(df)

    features = {}
    for timeframe in train_config.feature.timeframes:
        df_resampled = data.resample(df_base, timeframe)
        features[timeframe] = data.create_features(
            df_resampled,
            base_timing=train_config.feature.base_timing,
            moving_window_sizes=train_config.feature.moving_window_sizes,
            moving_window_size_center=train_config.feature.moving_window_size_center,
            sma_frac_unit=train_config.feature.sma_frac_unit,
        )

    lift = data.calc_lift(df_base["close"], train_config.lift.alpha)

    base_index = data.calc_available_index(
        features=features,
        hist_len=train_config.feature.hist_len,
    )

    print(f"Evaluation period: {base_index[0]} ~ {base_index[-1]}")
    run["data/size"] = len(base_index)
    run["data/first_timestamp"] = str(base_index[0])
    run["data/last_timestamp"] = str(base_index[-1])

    raw_loader = data.RawLoader(
        base_index=base_index,
        features=features,
        lift=lift,
        hist_len=train_config.feature.hist_len,
        moving_window_size_center=train_config.feature.moving_window_size_center,
        batch_size=train_config.batch_size,
    )
    normalized_loader = data.NormalizedLoader(
        loader=raw_loader,
        feature_info=feature_info_all[config.symbol],
    )
    combined_loader = data.CombinedLoader(
        loaders={config.symbol: normalized_loader},
        key_map=symbol_idxs,
    )

    net = model.Net(
        symbol_num=len(train_config.symbols),
        feature_info=feature_info_all[config.symbol],
        hist_len=train_config.feature.hist_len,
        numerical_emb_dim=train_config.net.numerical_emb_dim,
        periodic_activation_num_coefs=train_config.net.periodic_activation_num_coefs,
        periodic_activation_sigma=train_config.net.periodic_activation_sigma,
        categorical_emb_dim=train_config.net.categorical_emb_dim,
        emb_kernel_size=train_config.net.emb_kernel_size,
        num_blocks=train_config.net.num_blocks,
        block_qkv_kernel_size=train_config.net.block_qkv_kernel_size,
        block_ff_kernel_size=train_config.net.block_ff_kernel_size,
        block_channels=train_config.net.block_channels,
        block_ff_channels=train_config.net.block_ff_channels,
        block_dropout=train_config.net.block_dropout,
        head_hidden_dims=train_config.net.head_hidden_dims,
        head_batchnorm=train_config.net.head_batchnorm,
        head_dropout=train_config.net.head_dropout,
        head_output_dim=len(train_config.loss.bucket_boundaries) + 1,
    )
    net.load_state_dict(net_state)
    model_ = model.Model(net, bucket_boundaries=train_config.loss.bucket_boundaries)
    trainer = pl.Trainer(logger=False)

    scores_torch = cast(list[torch.Tensor], trainer.predict(model_, combined_loader))
    score = pd.Series(
        np.concatenate([s.numpy() for s in scores_torch]),
        index=base_index,
        dtype=np.float32,
    )

    rates = df_base.loc[base_index, config.simulation.timing]
    log_metrics(
        config=config,
        lift=lift,
        score=score,
        run=run,
    )
    run_simulations(
        config=config,
        rates=rates,
        score=score,
        run=run,
    )


if __name__ == "__main__":
    config = utils.get_config(EvalConfig)
    print(OmegaConf.to_yaml(config))

    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
