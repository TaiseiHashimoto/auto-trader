from datetime import datetime, timedelta

import numpy as np

from auto_trader.modeling import strategy


def test_time_limit_strategy() -> None:
    strategy_ = strategy.TimeLimitStrategy(
        thresh_long_entry=1.0,
        thresh_short_entry=-1.0,
        entry_time_max=10,
    )
    score = np.zeros(60)
    score[10] = 1.0
    score[25] = 1.0
    score[26] = -1.0
    score[40] = -1.0
    score[42] = 1.0
    expected = [(False, False, False, False) for _ in range(len(score))]
    # long entry
    expected[10] = (True, False, False, False)
    # 時間切れで long exit
    expected[20] = (False, True, False, False)
    # long entry
    expected[25] = (True, False, False, False)
    # short entry により同時に long exit
    expected[26] = (False, True, True, False)
    # 時間切れで short exit
    expected[36] = (False, False, False, True)
    # short entry
    expected[40] = (False, False, True, False)
    # long entry により同時に short exit
    expected[42] = (True, False, False, True)
    # 時間切れで long exit
    expected[52] = (False, True, False, False)

    start_dt = datetime(2023, 1, 1, 0, 0)
    for i in range(len(score)):
        actual = strategy_.make_decision(start_dt + timedelta(minutes=i), score[i])
        assert actual == expected[i], str(i)
