import os
from pathlib import Path
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


def calc_stats(rates: NDArray[np.float32]) -> dict[str, float]:
    PERCENTILES = np.arange(21) * 5
    return {
        "mean": cast(float, np.mean(rates)),
        **{
            f"percentile_{p}": cast(float, np.percentile(rates, p)) for p in PERCENTILES
        },
    }


def log_metrics(
    config: EvalConfig,
    lift: "pd.Series[float]",
    label: "pd.Series[float]",
    score: "pd.Series[float]",
    run: neptune.Run,
) -> None:
    score_np = score.to_numpy()
    run["stats/score"] = calc_stats(score_np)

    label_long_entry = label.loc[score.index] == 2
    label_short_entry = label.loc[score.index] == 0
    run["stats/roc_auc/long_entry"] = roc_auc_score(label_long_entry, score)
    run["stats/roc_auc/short_entry"] = roc_auc_score(label_short_entry, -score)
    run["stats/pr_auc/long_entry"] = average_precision_score(label_long_entry, score)
    run["stats/pr_auc/short_entry"] = average_precision_score(label_short_entry, -score)

    thresh_entry = np.percentile(np.abs(score_np), config.strategy.percentile_entry)
    pred_long_entry = score_np > thresh_entry
    pred_short_entry = score_np < -thresh_entry
    run["stats/ratio/long_entry"] = pred_long_entry.mean()
    run["stats/ratio/short_entry"] = pred_short_entry.mean()
    run["stats/precision/long_entry"] = precision_score(
        label_long_entry, pred_long_entry
    )
    run["stats/precision/short_entry"] = precision_score(
        label_short_entry, pred_short_entry
    )
    run["stats/recall/long_entry"] = recall_score(label_long_entry, pred_long_entry)
    run["stats/recall/short_entry"] = recall_score(label_short_entry, pred_short_entry)

    lift_long_entry = lift.loc[score.index].loc[pred_long_entry].to_numpy()
    lift_short_entry = lift.loc[score.index].loc[pred_short_entry].to_numpy()
    if len(lift_long_entry) > 0:
        run["stats/lift/long_entry"] = calc_stats(lift_long_entry)
    if len(lift_short_entry) > 0:
        run["stats/lift/short_entry"] = calc_stats(lift_short_entry)


def run_simulations(
    config: EvalConfig,
    rate: "pd.Series[float]",
    score: "pd.Series[float]",
    run: neptune.Run,
) -> None:
    thresh_entry = cast(
        float, np.percentile(np.abs(score.to_numpy()), config.strategy.percentile_entry)
    )
    strategy_ = strategy.TimeLimitStrategy(
        thresh_long_entry=thresh_entry,
        thresh_short_entry=-thresh_entry,
        entry_time_max=config.strategy.entry_time_max,
    )
    simulator = order.OrderSimulator(
        config.simulation.start_hour,
        config.simulation.end_hour,
        config.simulation.thresh_losscut,
    )
    for i, timestamp in enumerate(rate.index):
        (
            pred_long_entry,
            pred_long_exit,
            pred_short_entry,
            pred_short_exit,
        ) = strategy_.make_decision(timestamp, score.values[i])

        # TODO: 戦略 (e.g. n 回連続で long 選択の場合に実際に long する) を導入
        simulator.step(
            timestamp,
            rate.iloc[i],
            pred_long_entry,
            pred_long_exit,
            pred_short_entry,
            pred_short_exit,
        )

    os.makedirs(config.output_dir, exist_ok=True)
    results = simulator.export_results()
    results_path = os.path.join(config.output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    run["simulation/results"].upload(results_path)

    if len(results) > 0:
        profit = pd.Series(
            results["gain"].to_numpy() - config.simulation.spread,
            index=results["entry_time"],
        )
        profit_per_day = profit.resample("1d").sum()
        num_order_per_day = profit.resample("1d").count()
        run["simulation/num_order_per_day"] = calc_stats(num_order_per_day.to_numpy())
        run["simulation/profit_per_trade"] = calc_stats(profit.to_numpy())
        run["simulation/profit_per_day"] = calc_stats(profit_per_day.to_numpy())
        duration = (
            results["exit_time"] - results["entry_time"]
        ).dt.total_seconds() / 60
        run["simulation/duration"] = calc_stats(duration.to_numpy())


def main(config: EvalConfig) -> None:
    run = neptune.init_run(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["classification", "eval"],
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
    feature_stats = cast(
        dict[data.FeatureName, data.FeatureStats], params["feature_stats"]
    )
    label_stats = cast(data.CategoricalFeatureStats, params["label_stats"])
    net_state = params["net_state"]

    df_cleansed = data.read_cleansed_data(
        cleansed_data_dir=Path(config.cleansed_data_dir),
        symbol=config.symbol,
        yyyymm_begin=config.yyyymm_begin,
        yyyymm_end=config.yyyymm_end,
    )
    df_rate = data.merge_bid_ask(df_cleansed)

    features = data.create_features(
        df_rate,
        base_timing=train_config.feature.base_timing,
        window_sizes=train_config.feature.window_sizes,
        use_fraction=train_config.feature.use_fraction,
        fraction_unit=train_config.feature.fraction_unit,
        use_hour=train_config.feature.use_hour,
        use_dow=train_config.feature.use_dow,
    )
    features = data.relativize_features(features, train_config.feature.base_timing)
    features = data.normalize_features(features, feature_stats)
    label = data.create_label(
        df_rate["close"],
        train_config.label.future_step,
        train_config.label.bin_boundary,
    )

    index = data.calc_available_index(
        features=features,
        label=label,
        hist_len=train_config.feature.hist_len,
        hour_begin=train_config.feature.hour_begin,
        hour_end=train_config.feature.hour_end,
    )
    print(f"Evaluation period: {index[0]} ~ {index[-1]}")
    run["data/size"] = len(index)
    run["data/first_timestamp"] = str(index[0])
    run["data/last_timestamp"] = str(index[-1])

    loader = data.SequentialLoader(
        available_index=index,
        features=features,
        label=label,
        hist_len=train_config.feature.hist_len,
        batch_size=train_config.batch_size,
    )

    net = model.Net(
        feature_stats=feature_stats,
        hist_len=train_config.feature.hist_len,
        continuous_emb_dim=train_config.net.continuous_emb_dim,
        periodic_activation_num_coefs=train_config.net.periodic_activation_num_coefs,
        periodic_activation_sigma=train_config.net.periodic_activation_sigma,
        categorical_emb_dim=train_config.net.categorical_emb_dim,
        out_channels=train_config.net.out_channels,
        kernel_sizes=train_config.net.kernel_sizes,
        pooling_sizes=train_config.net.pooling_sizes,
        batchnorm=train_config.net.batchnorm,
        layernorm=train_config.net.layernorm,
        dropout=train_config.net.dropout,
        head_hidden_dims=train_config.net.head_hidden_dims,
        head_batchnorm=train_config.net.head_batchnorm,
        head_dropout=train_config.net.head_dropout,
        head_output_dim=label_stats.vocab_size,
    )
    net.load_state_dict(net_state)
    model_ = model.Model(net)
    trainer = pl.Trainer(logger=False)

    scores_torch = cast(list[torch.Tensor], trainer.predict(model_, loader))
    score = pd.Series(
        np.concatenate([s.numpy() for s in scores_torch]),
        index=index,
        dtype=np.float32,
    )

    rate = df_rate.loc[index, config.simulation.timing]
    # NOTE: data.calc_lift の lift とは異なり、単純に future_step 後との比較
    lift = rate.shift(-train_config.label.future_step) - rate
    log_metrics(
        config=config,
        lift=lift,
        label=label,
        score=score,
        run=run,
    )
    run_simulations(
        config=config,
        rate=rate,
        score=score,
        run=run,
    )


if __name__ == "__main__":
    config = utils.get_config(EvalConfig)
    print(OmegaConf.to_yaml(config))

    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
