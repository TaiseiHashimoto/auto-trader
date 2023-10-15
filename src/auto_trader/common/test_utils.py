import numpy as np
import pandas as pd
from pytest import approx

from auto_trader.common import utils


class TestOrder:
    # def prepare_long_order(self):
    def test_gain(self):
        order_long = utils.Order(
            position_type=utils.PositionType.LONG,
            entry_timestamp=pd.Timestamp("2022-01-01 10:00:00"),
            entry_rate=100.1,
        )
        order_long.exit(
            exit_timestamp=pd.Timestamp("2022-01-01 10:05:00"),
            exit_rate=100.5,
        )
        assert order_long.gain == approx(0.4)

        order_short = utils.Order(
            position_type=utils.PositionType.SHORT,
            entry_timestamp=pd.Timestamp("2022-01-02 10:00:00"),
            entry_rate=100.1,
        )
        order_short.exit(
            exit_timestamp=pd.Timestamp("2022-01-02 10:05:00"),
            exit_rate=100.5,
        )
        assert order_short.gain == approx(-0.4)


class TestOrderSimulator:
    def test_step(self):
        index = pd.date_range("2022-01-03 00:00:00", "2022-01-03 23:59:59", freq="1min")

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

        simulator = utils.OrderSimulator(start_hour=2, end_hour=22, thresh_loss_cut=5.0)

        expected = {
            "position_type": [],
            "entry_timestamp": [],
            "exit_timestamp": [],
            "entry_rate": [],
            "exit_rate": [],
            "gain": [],
        }

        order_df.loc["2022-01-03 01:59:00", "long_entry"] = True  # 無効

        order_df.loc["2022-01-03 02:00:00", "long_entry"] = True
        order_df.loc["2022-01-03 02:00:00", "long_exit"] = True  # 無効
        order_df.loc["2022-01-03 02:01:00", "long_exit"] = True
        expected["position_type"].append("long")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 02:00:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 02:01:00"))
        expected["entry_rate"].append(719 - 120)
        expected["exit_rate"].append(719 - 121)
        expected["gain"].append(-1)

        order_df.loc["2022-01-03 02:01:00", "short_exit"] = True
        order_df.loc["2022-01-03 02:01:00", "short_entry"] = True
        order_df.loc["2022-01-03 02:59:00", "short_exit"] = True
        expected["position_type"].append("short")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 02:01:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 02:59:00"))
        expected["entry_rate"].append(719 - 121)
        expected["exit_rate"].append(719 - 179)
        expected["gain"].append(58)

        order_df.loc["2022-01-03 03:00:00", "long_entry"] = True  # 損切り
        expected["position_type"].append("long")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 03:00:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 03:05:00"))
        expected["entry_rate"].append(719 - 180)
        expected["exit_rate"].append(719 - 185)
        expected["gain"].append(-5)

        order_df.loc["2022-01-03 13:00:00", "long_entry"] = True
        order_df.loc["2022-01-03 14:00:00", "short_exit"] = True  # 無効
        order_df.loc["2022-01-03 14:00:00", "long_exit"] = True
        expected["position_type"].append("long")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 13:00:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 14:00:00"))
        expected["entry_rate"].append(60)
        expected["exit_rate"].append(120)
        expected["gain"].append(60)

        order_df.loc["2022-01-03 14:00:00", "short_entry"] = True  # 損切り
        expected["position_type"].append("short")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 14:00:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 14:05:00"))
        expected["entry_rate"].append(120)
        expected["exit_rate"].append(125)
        expected["gain"].append(-5)

        order_df.loc["2022-01-03 21:58:00", "long_entry"] = True  # 時間切れ
        expected["position_type"].append("long")
        expected["entry_timestamp"].append(pd.Timestamp("2022-01-03 21:58:00"))
        expected["exit_timestamp"].append(pd.Timestamp("2022-01-03 22:00:00"))
        expected["entry_rate"].append(598)
        expected["exit_rate"].append(600)
        expected["gain"].append(2)

        for i in range(len(index)):
            simulator.step(
                index[i],
                order_df["rate"][i],
                order_df["long_entry"][i],
                order_df["short_entry"][i],
                order_df["long_exit"][i],
                order_df["short_exit"][i],
            )

        expected = pd.DataFrame(expected)
        actual = simulator.export_results()

        pd.testing.assert_frame_equal(expected, actual)


def test_calc_yyyymm():
    assert utils.calc_yyyymm(202001, month_delta=2) == 202003
    assert utils.calc_yyyymm(202001, month_delta=11) == 202012
    assert utils.calc_yyyymm(202001, month_delta=12) == 202101
    assert utils.calc_yyyymm(202001, month_delta=-1) == 201912
    assert utils.calc_yyyymm(202002, month_delta=-1) == 202001
    assert utils.calc_yyyymm(202002, month_delta=0) == 202002
