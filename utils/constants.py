from __future__ import annotations
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "OHLCV.csv"
SESSION_START = "07:00"
SESSION_END = "16:59"
FREQ = "1min"
WEEKDAYS_ONLY = True
CRISIS_START = "2021-01-01"
CRISIS_END = "2023-07-31"
REGIME_ORDER_THREE_STATE = ["pre-crisis", "crisis", "post-crisis"]
DEFAULT_MAX_GAP_MINUTES = 10