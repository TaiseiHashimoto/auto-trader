from datetime import datetime

import numpy as np
import pandas as pd
from pytest import approx

from auto_trader.modeling import order


class TestOrder:
    def test_gain(self) -> None:
        order_long = order.Order(
            position_type=order.PositionType.LONG,
            entry_time=datetime(2023, 1, 1, 10, 0, 0),
            entry_rate=100.1,
        )
        order_long.exit(
            exit_time=datetime(2023, 1, 1, 10, 5, 0),
            exit_rate=100.5,
        )
        assert order_long.gain == approx(0.4)

        order_short = order.Order(
            position_type=order.PositionType.SHORT,
            entry_time=datetime(2023, 1, 2, 10, 0, 0),
            entry_rate=100.1,
        )
        order_short.exit(
            exit_time=datetime(2023, 1, 2, 10, 5, 0),
            exit_rate=100.5,
        )
        assert order_short.gain == approx(-0.4)


class TestOrderSimulator:
    def test_step(self) -> None:
        index = pd.date_range("2023-01-03 00:00:00", "2023-01-03 23:59:59", freq="1min")

        # rate = [719, 718, ..., 1, 0, 0, 1, ..., 718, 719]
        half_range = range(0, len(index) // 2)
        rate = [*reversed(half_range), *half_range]

        order_df = pd.DataFrame(
            {
                "rate": rate,
                "long_entry": np.zeros(len(index), dtype=bool),
                "short_entry": np.zeros(len(index), dtype=bool),
                "long_exit": np.zeros(len(index), dtype=bool),
                "short_exit": np.zeros(len(index), dtype=bool),
            },
            index=index,
        )

        simulator = order.OrderSimulator(start_hour=2, end_hour=22, thresh_loss_cut=5.0)
        expected = []

        order_df.loc["2023-01-03 01:59:00", "long_entry"] = True  # 無効

        order_df.loc["2023-01-03 02:00:00", "long_entry"] = True
        order_df.loc["2023-01-03 02:00:00", "long_exit"] = True  # 無効
        order_df.loc["2023-01-03 02:01:00", "long_exit"] = True
        expected.append(
            {
                "position_type": "long",
                "entry_time": datetime(2023, 1, 3, 2, 00, 00),
                "exit_time": datetime(2023, 1, 3, 2, 1, 00),
                "entry_rate": 719 - 120,
                "exit_rate": 719 - 121,
                "gain": -1,
            }
        )

        order_df.loc["2023-01-03 02:01:00", "short_exit"] = True
        order_df.loc["2023-01-03 02:01:00", "short_entry"] = True
        order_df.loc["2023-01-03 02:59:00", "short_exit"] = True
        expected.append(
            {
                "position_type": "short",
                "entry_time": datetime(2023, 1, 3, 2, 1, 00),
                "exit_time": datetime(2023, 1, 3, 2, 59, 00),
                "entry_rate": 719 - 121,
                "exit_rate": 719 - 179,
                "gain": 58,
            }
        )

        order_df.loc["2023-01-03 03:00:00", "long_entry"] = True  # 損切り
        expected.append(
            {
                "position_type": "long",
                "entry_time": datetime(2023, 1, 3, 3, 00, 00),
                "exit_time": datetime(2023, 1, 3, 3, 5, 00),
                "entry_rate": 719 - 180,
                "exit_rate": 719 - 185,
                "gain": -5,
            }
        )

        order_df.loc["2023-01-03 13:00:00", "long_entry"] = True
        order_df.loc["2023-01-03 14:00:00", "short_exit"] = True  # 無効
        order_df.loc["2023-01-03 14:00:00", "long_exit"] = True
        expected.append(
            {
                "position_type": "long",
                "entry_time": datetime(2023, 1, 3, 13, 00, 00),
                "exit_time": datetime(2023, 1, 3, 14, 00, 00),
                "entry_rate": 60,
                "exit_rate": 120,
                "gain": 60,
            }
        )

        order_df.loc["2023-01-03 14:00:00", "short_entry"] = True  # 損切り
        expected.append(
            {
                "position_type": "short",
                "entry_time": datetime(2023, 1, 3, 14, 00, 00),
                "exit_time": datetime(2023, 1, 3, 14, 5, 00),
                "entry_rate": 120,
                "exit_rate": 125,
                "gain": -5,
            }
        )

        order_df.loc["2023-01-03 21:58:00", "long_entry"] = True  # 時間切れ
        expected.append(
            {
                "position_type": "long",
                "entry_time": datetime(2023, 1, 3, 21, 58, 00),
                "exit_time": datetime(2023, 1, 3, 22, 00, 00),
                "entry_rate": 598,
                "exit_rate": 600,
                "gain": 2,
            }
        )

        for i in range(len(index)):
            simulator.step(
                index[i],
                order_df["rate"].iloc[i],
                order_df["long_entry"].iloc[i],
                order_df["short_entry"].iloc[i],
                order_df["long_exit"].iloc[i],
                order_df["short_exit"].iloc[i],
            )

        pd.testing.assert_frame_equal(
            pd.DataFrame(expected), simulator.export_results()
        )
