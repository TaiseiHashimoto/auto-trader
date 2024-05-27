from datetime import datetime
from typing import Optional


class TimeLimitStrategy:
    def __init__(
        self, thresh_long_entry: float, thresh_short_entry: float, entry_time_max: int
    ):
        self.thresh_long_entry = thresh_long_entry
        self.thresh_short_entry = thresh_short_entry
        self.entry_time_max = entry_time_max

        self.dt_long_entry: Optional[datetime] = None
        self.dt_short_entry: Optional[datetime] = None

    def make_decision(
        self, dt: datetime, score: float
    ) -> tuple[bool, bool, bool, bool]:
        long_entry = self.dt_long_entry is None and score >= self.thresh_long_entry
        short_entry = self.dt_short_entry is None and score <= self.thresh_short_entry
        long_exit = self.dt_long_entry is not None and (
            (dt - self.dt_long_entry).total_seconds() // 60 >= self.entry_time_max
            or short_entry
        )
        short_exit = self.dt_short_entry is not None and (
            (dt - self.dt_short_entry).total_seconds() // 60 >= self.entry_time_max
            or long_entry
        )

        if long_exit:
            self.dt_long_entry = None
        if short_exit:
            self.dt_short_entry = None
        if long_entry:
            self.dt_long_entry = dt
        if short_entry:
            self.dt_short_entry = dt

        return long_entry, long_exit, short_entry, short_exit
