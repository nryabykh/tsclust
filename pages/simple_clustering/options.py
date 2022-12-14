from dataclasses import dataclass
from datetime import date
from typing import List


@dataclass
class DataOptions:
    date: date
    eps: float
    min_samples: int
    excluded_series: List[str]
