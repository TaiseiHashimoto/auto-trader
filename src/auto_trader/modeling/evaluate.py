import itertools
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
from tqdm import tqdm

from auto_trader.common import utils
from auto_trader.modeling import data, model, order
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
    gain_long: "pd.Series[float]",
    gain_short: "pd.Series[float]",
    preds: pd.DataFrame,
    run: neptune.Run,
) -> None:
    for label_name in preds.columns:
        pred = cast(NDArray[np.float32], preds[label_name].values)
        run[f"stats/prob/{label_name}"] = calc_stats(pred)

        if label_name == "long_entry":
            label = (gain_long.loc[preds.index] > config.simulation.spread).values
        elif label_name == "long_exit":
            label = (gain_long.loc[preds.index] < 0).values
        elif label_name == "short_entry":
            label = (gain_short.loc[preds.index] > config.simulation.spread).values
        elif label_name == "short_exit":
            label = (gain_short.loc[preds.index] < 0).values
        else:
            assert False

        run[f"stats/roc_auc/{label_name}"] = roc_auc_score(label, pred)
        run[f"stats/pr_auc/{label_name}"] = average_precision_score(label, pred)

        if "entry" in label_name:
            percentile_list = config.percentile_entry_list
        elif "exit" in label_name:
            percentile_list = config.percentile_exit_list
        else:
            assert False

        for percentile in percentile_list:
            pred_binary = get_binary_pred(pred, percentile)
            run[f"stats/precision/{label_name}/{percentile}"] = precision_score(
                label, pred_binary
            )
            run[f"stats/recall/{label_name}/{percentile}"] = recall_score(
                label, pred_binary
            )


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
            config.simulation.thresh_losscut,
        )
        for i, timestamp in enumerate(rates.index):
            # TODO: 戦略 (e.g. n 回連続で long 選択の場合に実際に long する) を導入
            simulator.step(
                timestamp,
                rates.iloc[i],
                pred_binary_long_entry[i],
                pred_binary_long_exit[i],
                pred_binary_short_entry[i],
                pred_binary_short_exit[i],
            )

        PARAM_STR = f"{percentile_entry},{percentile_exit}"

        results = simulator.export_results()
        results_path = f"{config.output_dir}/results_{PARAM_STR}.csv"
        results.to_csv(results_path, index=False)
        run[f"simulation/{PARAM_STR}/results"].upload(results_path)

        if len(results) > 0:
            profit = pd.Series(
                cast(NDArray[np.float32], results["gain"].values)
                - config.simulation.spread,
                index=results["entry_time"],
            )
            profit_per_day = profit.resample("1d").sum()
            num_order_per_day = profit.resample("1d").count()
            run[f"simulation/{PARAM_STR}/num_order_per_day"] = calc_stats(
                cast(NDArray[np.float32], num_order_per_day.values)
            )
            run[f"simulation/{PARAM_STR}/profit_per_trade"] = calc_stats(
                cast(NDArray[np.float32], profit.values)
            )
            run[f"simulation/{PARAM_STR}/profit_per_day"] = calc_stats(
                cast(NDArray[np.float32], profit_per_day.values)
            )
            duration = (
                results["exit_time"] - results["entry_time"]
            ).dt.total_seconds() / 60
            run[f"simulation/{PARAM_STR}/duration"] = calc_stats(
                cast(NDArray[np.float32], duration.values)
            )


def main(config: EvalConfig) -> None:
    run = neptune.init_run(
        project=config.neptune.project,
        mode=config.neptune.mode,
        tags=["eval"],
    )
    run["config"] = OmegaConf.to_yaml(config)

    print("Fetch params")
    if config.params_file == "":
        train_run = neptune.init_run(
            project=config.neptune.project,
            with_id=config.train_run_id,
            mode=config.neptune.mode,
        )
        os.makedirs(config.output_dir, exist_ok=True)
        params_file = os.path.join(config.output_dir, "params.pt")
        train_run["params_file"].download(params_file)
        train_run.stop()
    else:
        params_file = config.params_file

    params = torch.load(params_file)
    train_config = cast(TrainConfig, OmegaConf.create(params["config"]))
    feature_info = params["feature_info"]
    net_state = params["net_state"]
    run["sys/tags"].add(train_config.net.base_net_type)

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

    gain_long, gain_short = data.calc_gains(
        df_base["close"], train_config.gain.alpha, train_config.gain.thresh_losscut
    )

    base_index = data.calc_available_index(
        features=features,
        hist_len=train_config.feature.hist_len,
        # 取引時間は OrderSimulator で絞る
        start_hour=0,
        end_hour=24,
    )

    print(f"Evaluation period: {base_index[0]} ~ {base_index[-1]}")
    run["data/size"] = len(base_index)
    run["data/first_timestamp"] = str(base_index[0])
    run["data/last_timestamp"] = str(base_index[-1])

    loader = data.DataLoader(
        base_index=base_index,
        features=features,
        gain_long=gain_long,
        gain_short=gain_short,
        hist_len=train_config.feature.hist_len,
        sma_window_size_center=train_config.feature.sma_window_size_center,
        batch_size=train_config.batch_size,
    )
    net = model.Net(
        feature_info=feature_info,
        hist_len=train_config.feature.hist_len,
        numerical_emb_dim=train_config.net.numerical_emb_dim,
        periodic_activation_num_coefs=train_config.net.periodic_activation_num_coefs,
        periodic_activation_sigma=train_config.net.periodic_activation_sigma,
        categorical_emb_dim=train_config.net.categorical_emb_dim,
        emb_output_dim=train_config.net.emb_output_dim,
        base_net_type=train_config.net.base_net_type,
        base_attention_num_layers=train_config.net.base_attention_num_layers,
        base_attention_num_heads=train_config.net.base_attention_num_heads,
        base_attention_feedforward_dim=train_config.net.base_attention_feedforward_dim,
        base_attention_dropout=train_config.net.base_attention_dropout,
        base_conv_out_channels=train_config.net.base_conv_out_channels,
        base_conv_kernel_sizes=train_config.net.base_conv_kernel_sizes,
        base_conv_batchnorm=train_config.net.base_conv_batchnorm,
        base_conv_dropout=train_config.net.base_conv_dropout,
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

    preds_torch = cast(list[model.Predictions], trainer.predict(model_, loader))
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
    log_metrics(
        config=config,
        gain_long=gain_long,
        gain_short=gain_short,
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
    config = utils.get_config(EvalConfig)
    print(OmegaConf.to_yaml(config))

    utils.validate_neptune_settings(config.neptune.mode)

    main(config)
