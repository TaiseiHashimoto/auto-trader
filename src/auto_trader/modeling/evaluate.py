import itertools
import os
from typing import Optional, cast

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
from tqdm import tqdm

from auto_trader.common import utils
from auto_trader.modeling import data, model, order
from auto_trader.modeling.config import EvalConfig, TrainConfig


def get_binary_pred(pred: NDArray[np.float32], percentile: float) -> NDArray[np.bool_]:
    return pred > np.percentile(pred, percentile)


def calc_metrics(
    config: EvalConfig,
    rates: "pd.Series[float]",
    lift: "pd.Series[float]",
    preds: pd.DataFrame,
    run: neptune.Run,
) -> None:
    PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for label_name in preds.columns:
        pred = cast(NDArray[np.float32], preds[label_name].values)
        run[f"stats/score_percentile/{label_name}"] = {
            str(p): np.percentile(pred, p) for p in PERCENTILES
        }

        if "entry" in label_name:
            percentile_list = config.percentile_entry_list
            lift_binary = (lift.loc[preds.index] > config.simulation.spread).values
        elif "exit" in label_name:
            percentile_list = config.percentile_exit_list
            lift_binary = (lift.loc[preds.index] < 0).values
        else:
            assert False

        run[f"stats/roc_auc/{label_name}"] = roc_auc_score(lift_binary, pred)
        run[f"stats/pr_auc/{label_name}"] = average_precision_score(lift_binary, pred)

        for percentile in percentile_list:
            pred_binary = get_binary_pred(pred, percentile)
            run[f"stats/precision/{label_name}/{percentile}"] = precision_score(
                lift_binary, pred_binary
            )
            run[f"stats/recall/{label_name}/{percentile}"] = recall_score(
                lift_binary, pred_binary
            )

            for future_step in [5, 10, 20, 30, 60]:
                rates_diff = rates.shift(-future_step) - rates
                run[f"lift/{label_name}/{percentile}/{future_step}"] = rates_diff[
                    pred_binary
                ].mean()


def run_simulations(
    config: EvalConfig, rates: "pd.Series[float]", preds: pd.DataFrame, run: neptune.Run
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    params_list = list(
        itertools.product(config.percentile_entry_list, config.percentile_exit_list)
    )
    for percentile_entry, percentile_exit in tqdm(params_list):
        pred_binary_long_entry = get_binary_pred(
            cast(NDArray[np.float32], preds["long_entry"].values), percentile_entry
        )
        pred_binary_long_exit = get_binary_pred(
            cast(NDArray[np.float32], preds["long_exit"].values), percentile_exit
        )
        pred_binary_short_entry = get_binary_pred(
            cast(NDArray[np.float32], preds["short_entry"].values), percentile_entry
        )
        pred_binary_short_exit = get_binary_pred(
            cast(NDArray[np.float32], preds["short_exit"].values), percentile_exit
        )

        simulator = order.OrderSimulator(
            config.simulation.start_hour,
            config.simulation.end_hour,
            config.simulation.thresh_loss_cut,
        )
        for i, timestamp in enumerate(rates.index):
            simulator.step(
                timestamp,
                rates.iloc[i],
                pred_binary_long_entry[i],
                pred_binary_short_entry[i],
                pred_binary_long_exit[i],
                pred_binary_short_exit[i],
            )

        profits = (
            np.array([order.gain for order in simulator.order_history])
            - config.simulation.spread
        )
        durations = np.array([order.duration for order in simulator.order_history])

        param_str = f"{percentile_entry},{percentile_exit}"
        run[f"simulation/num_order/{param_str}"] = len(profits)
        if len(profits) > 0:
            run[f"simulation/profit_per_trade/{param_str}"] = profits.mean()
            days = (rates.index[-1] - rates.index[0]).days * (5 / 7)
            run[f"simulation/profit_per_day/{param_str}"] = profits.sum() / days
            run[f"simulation/duration/min/{param_str}"] = durations.min()
            run[f"simulation/duration/max/{param_str}"] = durations.max()
            run[f"simulation/duration/mean/{param_str}"] = durations.mean()
            run[f"simulation/duration/median/{param_str}"] = np.median(durations)

        results = simulator.export_results()
        results_path = f"{config.output_dir}/results_{param_str}.csv"
        results.to_csv(results_path, index=False)
        run[f"simulation/results/{param_str}"].upload(results_path)


def main(config: EvalConfig) -> None:
    run = neptune.init_run(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["eval", "cnn"],
    )
    run["config"] = OmegaConf.to_yaml(config)

    print("Fetch params")
    train_run: Optional[neptune.Run] = None
    if config.params_file == "":
        params_file = os.path.join(config.output_dir, "params.pt")
        train_run = neptune.init_run(
            with_id=config.train_run_id,
            mode=config.neptune.mode,
        )
        train_run["params_file"].download(params_file)
    else:
        params_file = config.params_file

    params = torch.load(params_file)
    train_config = cast(TrainConfig, OmegaConf.create(params["config"]))
    feature_info = params["feature_info"]
    net_state = params["net_state"]

    df = data.read_cleansed_data(
        config.data.symbol,
        config.data.yyyymm_begin,
        config.data.yyyymm_end,
        config.data.cleansed_data_dir,
    )
    df_base = data.merge_bid_ask(df)

    features = {}
    for timeframe in train_config.feature.timeframes:
        df_resampled = data.resample(df_base, timeframe)
        features[timeframe] = data.create_features(
            df_resampled,
            base_timing=train_config.feature.base_timing,
            sma_window_sizes=train_config.feature.sma_window_sizes,
            sma_window_size_center=train_config.feature.sma_window_size_center,
            sigma_window_sizes=train_config.feature.sigma_window_sizes,
            sma_frac_unit=train_config.feature.sma_frac_unit,
        )

    lift = data.calc_lift(df_base["close"], train_config.lift.target_alpha)

    base_index = data.calc_available_index(
        features,
        lift,
        train_config.feature.hist_len,
        train_config.feature.start_hour,
        train_config.feature.end_hour,
    )

    print(f"Evaluation period: {base_index[0]} ~ {base_index[-1]}")
    run["data/size"] = len(base_index)
    run["data/first_timestamp"] = str(base_index[0])
    run["data/last_timestamp"] = str(base_index[-1])

    loader = data.DataLoader(
        base_index=base_index,
        features=features,
        lift=lift,
        hist_len=train_config.feature.hist_len,
        sma_window_size_center=train_config.feature.sma_window_size_center,
        batch_size=train_config.batch_size,
    )
    net = model.Net(
        feature_info=feature_info,
        window_size=train_config.feature.hist_len,
        numerical_emb_dim=train_config.net.numerical_emb_dim,
        categorical_emb_dim=train_config.net.categorical_emb_dim,
        periodic_activation_sigma=train_config.net.periodic_activation_sigma,
        base_cnn_out_channels=train_config.net.base_cnn_out_channels,
        base_cnn_kernel_sizes=train_config.net.base_cnn_kernel_sizes,
        base_cnn_batchnorm=train_config.net.base_cnn_batchnorm,
        base_cnn_dropout=train_config.net.base_cnn_dropout,
        base_fc_hidden_dims=train_config.net.base_fc_hidden_dims,
        base_fc_batchnorm=train_config.net.base_fc_batchnorm,
        base_fc_dropout=train_config.net.base_fc_dropout,
        base_fc_output_dim=train_config.net.base_fc_output_dim,
        head_hidden_dims=train_config.net.head_hidden_dims,
        head_batchnorm=train_config.net.head_batchnorm,
        head_dropout=train_config.net.head_dropout,
    )
    net.load_state_dict(net_state)
    model_ = model.Model(net)
    trainer = pl.Trainer(logger=False)

    preds_torch = cast(list[torch.Tensor], trainer.predict(model_, loader))
    preds = pd.DataFrame(
        {
            "long_entry": np.concatenate([p[0].numpy() for p in preds_torch]),
            "long_exit": np.concatenate([p[1].numpy() for p in preds_torch]),
            "short_entry": np.concatenate([p[2].numpy() for p in preds_torch]),
            "short_exit": np.concatenate([p[3].numpy() for p in preds_torch]),
        },
        index=base_index,
        dtype=np.float32,
    )

    rates = df_base.loc[base_index, config.simulation.timing]
    calc_metrics(
        config=config,
        rates=rates,
        lift=lift,
        preds=preds,
        run=run,
    )
    run_simulations(
        config=config,
        rates=rates,
        preds=preds,
        run=run,
    )


if __name__ == "__main__":
    config = cast(EvalConfig, utils.get_config(EvalConfig))
    print(OmegaConf.to_yaml(config))

    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
