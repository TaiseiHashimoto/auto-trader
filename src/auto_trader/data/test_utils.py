import pandas as pd

from auto_trader.data import utils


def test_remove_flat_data():
    index_all = pd.date_range("2019-7-9 00:00:00", "2020-7-8 23:59:00", freq="1min")
    df_all = pd.DataFrame(index=index_all)
    df_returned = utils.remove_flat_data(df_all)
    index_returned = df_returned.index

    # 月~金のみを含む
    assert set(index_returned.dayofweek) == {0, 1, 2, 3, 4}
    # 月~金は1分間隔ですべて含む (1/1, 12/25 は特別扱い)
    count_all = (index_all.dayofweek < 5).sum()
    count_returned = (index_returned.dayofweek < 5).sum()
    assert count_all == count_returned + 60 * (24 + 14)
    # 1/1 は含まない
    assert ((index_returned.month == 1) & (index_returned.day == 1)).sum() == 0
    # 12/25 は 9:59 まで含む
    assert ((index_returned.month == 12) & (index_returned.day == 25)).sum() == 60 * 10
