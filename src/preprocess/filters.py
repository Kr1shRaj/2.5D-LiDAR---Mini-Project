import numpy as np

# keep mask: finite xyz only. nan or inf point = drop.
def finite_mask(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points[:, :3]), axis=1)


# keep mask: horizontal range sqrt(x^2 + y^2) inside [r_min, r_max). sensor frame only.
def range_gate_mask(points_velo: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    if not 0.0 <= r_min < r_max:
        raise ValueError(f"need 0 <= r_min < r_max, got {r_min}, {r_max}")
    r = np.hypot(points_velo[:, 0], points_velo[:, 1])
    return (r >= r_min) & (r < r_max)


# keep mask: drop points inside ego car box (x_min, x_max, y_min, y_max). sensor frame only.
def ego_vehicle_mask(points_velo: np.ndarray, ego_box: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = ego_box
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(f"bad ego box {ego_box}")
    x, y = points_velo[:, 0], points_velo[:, 1]
    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return ~inside


# combine all sensor frame filters into one keep mask. use mask on labels too.
def preprocess_mask(
    points_velo: np.ndarray,
    r_min: float,
    r_max: float,
    ego_box: tuple[float, float, float, float],
) -> np.ndarray:
    keep = finite_mask(points_velo)
    keep &= range_gate_mask(points_velo, r_min, r_max)
    keep &= ego_vehicle_mask(points_velo, ego_box)
    return keep
