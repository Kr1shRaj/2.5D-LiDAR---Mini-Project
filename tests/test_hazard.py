import numpy as np
import pytest

from src.mapping.uniform_grid import build_uniform_grid, centered_spec
from src.models.features import FEATURES, cell_features, feature_matrix, grid_sampler
from src.models.targets import HAZARDOUS, OBSTACLE, TRAVERSABLE, UNKNOWN, Thresholds, geometric_classes

GROUND_Z = -1.73
TH = Thresholds(obstacle_h=0.3, step=0.08, slope=0.15, rough=0.05)


# seeded test yard: flat road, 15 cm curb at x >= 4, 25 % ramp at y >= 4, 1 m box, empty patch
@pytest.fixture
def yard():
    rng = np.random.default_rng(seed=31)
    n = 400000
    x, y = rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)
    z = np.full(n, GROUND_Z)
    z[(x >= 4.0) & (y < 4.0)] += 0.15  # sidewalk
    ramp = (y >= 4.0) & (x < 4.0)
    z[ramp] += 0.25 * (y[ramp] - 4.0)
    keep = ~((x > -6) & (x < -4) & (y > -6) & (y < -4))  # empty patch, nothing seen
    pts = np.column_stack([x, y, z])[keep]
    box = np.column_stack([rng.uniform(-3, -2, 3000), rng.uniform(-3, -2, 3000), rng.uniform(GROUND_Z, GROUND_Z + 1, 3000)])
    pts = np.vstack([pts, box])
    return build_uniform_grid(pts, centered_spec(16.0, 0.1, "velo"), "velo", ground_percentile=10.0)


# true terrain of yard: flat, ramp follows slope (sidewalk counts as raised, not terrain)
def terrain(xy: np.ndarray) -> np.ndarray:
    return GROUND_Z + np.where((xy[:, 1] >= 4.0) & (xy[:, 0] < 4.0), 0.25 * (xy[:, 1] - 4.0), 0.0)


def classify(layers, xy: np.ndarray) -> np.ndarray:
    f = cell_features(grid_sampler(layers), xy, terrain(xy), np.zeros(2), 0.0)
    return geometric_classes(f, np.nan_to_num(f["count"]) > 0, TH)


def test_rule_labels_each_feature(yard) -> None:
    xy = np.array([
        [0.05, 0.05],  # flat road
        [4.05, 0.05],  # curb edge
        [6.05, 0.05],  # flat sidewalk, only 15 cm up
        [0.05, 6.05],  # steep ramp
        [-2.45, -2.45],  # box
        [-4.95, -4.95],  # empty patch
    ])
    assert classify(yard, xy).tolist() == [TRAVERSABLE, HAZARDOUS, TRAVERSABLE, HAZARDOUS, OBSTACLE, UNKNOWN]


def test_features_values(yard) -> None:
    xy = np.array([[4.05, 0.05], [0.05, 6.05], [-2.45, -2.45]])
    f = cell_features(grid_sampler(yard), xy, np.full(3, GROUND_Z), np.array([0.0, 0.0]), 0.0)
    assert f["step"][0] == pytest.approx(0.15, abs=0.01)
    assert f["slope"][1] == pytest.approx(0.25, abs=0.02)
    assert f["rel_max"][2] == pytest.approx(1.0, abs=0.1)  # random top point in cell
    assert f["range"][2] == pytest.approx(np.hypot(2.45, 2.45))
    assert np.isnan(f["weight"]).all()  # plain grid has no fused confidence
    assert feature_matrix(f).shape == (3, len(FEATURES))


def test_missing_terrain_never_obstacle(yard) -> None:
    xy = np.array([[-2.45, -2.45]])
    f = cell_features(grid_sampler(yard), xy, np.array([np.nan]), np.zeros(2), 0.0)
    assert geometric_classes(f, np.array([True]), TH).tolist() != [OBSTACLE]


class CountingModel:
    # fake model: label = 1 for every row, counts rows seen
    def __init__(self) -> None:
        self.rows = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.rows += len(X)
        return np.ones(len(X), dtype=np.int64)


def test_labeler_relabels_only_touched_cells_in_zone() -> None:
    from src.mapping.foveated_grid import Band
    from src.models.labeler import HazardLabeler
    from src.temporal.fusion import FusionParams, TemporalFoveatedMap

    bands = (Band(0.0, 5.0, 0.2), Band(5.0, 15.0, 0.5))
    params = FusionParams(tau_s=3.0, prune_weight=0.05, obstacle_h=0.3, l_hit=0.85, l_miss=-0.4, l_min=-2.0,
                          l_max=3.5, terrain_cell=2.0, ground_percentile=10.0)
    rng = np.random.default_rng(seed=32)
    pts = np.column_stack([rng.uniform(-14, 14, 60000), rng.uniform(-14, 14, 60000), np.full(60000, GROUND_Z)])
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 14.0]
    m = TemporalFoveatedMap(bands, params)
    model = CountingModel()
    lab = HazardLabeler(model, zone_m=5.0)
    m.integrate(pts, np.eye(4), t=0.0)
    first = lab.update(m.render(), m.terrain, 0.0)
    assert first > 0 and model.rows == first
    near_cells = int(m.render().bands[0].observed.sum())
    assert first <= near_cells  # coarse band past zone never labelled
    # same flat scene again at 0.1 s: nothing changed, labels young -> almost no model work (few new edge cells)
    m.integrate(pts, np.eye(4), t=0.1)
    assert lab.update(m.render(), m.terrain, 0.1) < first / 50
    # box appears in small patch: only changed cells re-labelled (event)
    box = np.column_stack([rng.uniform(1.6, 2.4, 3000), rng.uniform(-0.4, 0.4, 3000), rng.uniform(GROUND_Z, GROUND_Z + 1, 3000)])
    m.integrate(np.vstack([pts, box]), np.eye(4), t=0.2)
    second = lab.update(m.render(), m.terrain, 0.2)
    assert 0 < second < first / 10
    # 0.3 s after first label: periodic refresh of every touched cell in zone
    m.integrate(pts, np.eye(4), t=0.3)
    assert lab.update(m.render(), m.terrain, 0.3) > first / 2
    got = lab.labels_at(m.render(), np.array([[1.0, 1.0], [10.0, 0.0]]))
    assert got.tolist() == [1, -1]  # inside zone labelled, outside zone left to rule
