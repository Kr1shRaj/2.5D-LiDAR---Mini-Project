from dataclasses import dataclass

import numpy as np

TRAVERSABLE, HAZARDOUS, OBSTACLE, UNKNOWN = 0, 1, 2, 3
CLASS_NAMES = ["traversable", "potentially hazardous", "obstacle", "unknown"]


# geometric safety thresholds. heights metres, slope rise / run.
@dataclass(frozen=True)
class Thresholds:
    obstacle_h: float
    step: float
    slope: float
    rough: float


# geometric rule -> class per spot. obstacle beats hazard beats traversable. not enough data -> unknown.
# candidate hazard label from geometry only, NOT collision prediction.
def geometric_classes(f: dict[str, np.ndarray], enough_data: np.ndarray, th: Thresholds) -> np.ndarray:
    out = np.full(len(enough_data), TRAVERSABLE, dtype=np.int64)
    hazard = (f["step"] >= th.step) | (f["slope"] >= th.slope) | (f["roughness"] >= th.rough)
    out[hazard] = HAZARDOUS
    out[f["rel_max"] >= th.obstacle_h] = OBSTACLE
    out[~enough_data] = UNKNOWN
    return out
