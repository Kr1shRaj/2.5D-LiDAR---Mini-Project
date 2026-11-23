import numpy as np
import pytest

from src.models.targets import HAZARDOUS, OBSTACLE, TRAVERSABLE
from src.planning.corridor import (
    BLOCKED,
    CLEAR,
    NO_LABEL,
    CorridorParams,
    arc_probes_velo,
    check_corridor,
    probes_to_world,
)

P = CorridorParams()
STRAIGHT = len(P.curvatures) // 2


# pose: yaw about z then shift. world = velo frame at start.
def pose(x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:2, 3] = [x, y]
    return t


# label map in world xy: everything traversable, plus obstacle / hazard boxes
def world_labels(boxes: list[tuple[float, float, float, float, int]], default: int = TRAVERSABLE):
    def label_at(xy: np.ndarray) -> np.ndarray:
        out = np.full(len(xy), default, dtype=np.int64)
        for x0, x1, y0, y1, lab in boxes:
            out[(xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)] = lab
        return out

    return label_at


def test_positive_curvature_turns_left() -> None:
    probes = arc_probes_velo(P)
    centre = probes[:, :, probes.shape[2] // 2]
    assert centre[-1, -1, 1] > 5.0  # k > 0 ends left (y > 0)
    assert centre[0, -1, 1] < -5.0
    np.testing.assert_allclose(centre[STRAIGHT, :, 1], 0.0, atol=1e-9)
    # every centreline is arc length 20 m: end point sits on circle of radius 1/k through origin
    k = P.curvatures[-1]
    end = centre[-1, -1]
    assert np.hypot(end[0], end[1] - 1 / k) == pytest.approx(1 / k)


def test_wall_ahead_blocks_straight_side_arc_clear() -> None:
    wall = world_labels([(7.0, 8.0, -0.6, 0.6, OBSTACLE)])  # 1.2 m wide post 7 m ahead
    r = check_corridor(wall, pose(0.0, 0.0), P)
    assert r.verdict[STRAIGHT] == BLOCKED
    assert r.block_m[STRAIGHT] == pytest.approx(7.0, abs=0.25)
    assert r.verdict[0] == CLEAR and r.verdict[-1] == CLEAR  # hard turns miss the wall
    assert r.pick >= 0 and r.verdict[r.pick] != BLOCKED


def test_pose_moves_arcs_to_world_frame() -> None:
    # car at (100, 50) facing +y: wall must sit 6 m along world +y to block straight
    wall = world_labels([(98.5, 101.5, 56.0, 57.0, OBSTACLE)])
    r = check_corridor(wall, pose(100.0, 50.0, np.pi / 2), P)
    assert r.verdict[STRAIGHT] == BLOCKED
    probes = probes_to_world(arc_probes_velo(P), pose(100.0, 50.0, np.pi / 2))
    assert probes[STRAIGHT, -1, probes.shape[2] // 2] == pytest.approx([100.0, 70.0])


def test_unseen_map_gives_no_candidate() -> None:
    r = check_corridor(world_labels([], default=NO_LABEL), pose(0.0, 0.0), P)
    assert (r.unknown == 1.0).all()
    assert r.pick == -1  # unseen is not free


def test_hazard_band_steers_pick_away() -> None:
    # rough strip straight ahead, no obstacle: straight is caution, pick prefers a clean arc
    rough = world_labels([(2.0, 20.0, -1.5, 1.5, HAZARDOUS)])
    r = check_corridor(rough, pose(0.0, 0.0), P)
    assert r.verdict[STRAIGHT] != BLOCKED and r.hazard[STRAIGHT] > 0.5
    assert r.pick != STRAIGHT and r.hazard[r.pick] < r.hazard[STRAIGHT]


def test_previous_pick_keeps_choice_steady() -> None:
    open_road = world_labels([])
    left = check_corridor(open_road, pose(0.0, 0.0), P, prev_curvature=P.curvatures[STRAIGHT + 1])
    assert left.pick in (STRAIGHT, STRAIGHT + 1)
    assert check_corridor(open_road, pose(0.0, 0.0), P).pick == STRAIGHT


# one noisy obstacle cell does not block; a real 40 cm object does
def test_lone_obstacle_probe_ignored_cluster_blocks() -> None:
    lone = world_labels([(5.0, 5.1, -0.05, 0.05, OBSTACLE)])  # one 10 cm cell on centreline
    assert check_corridor(lone, pose(0.0, 0.0), P).verdict[STRAIGHT] != BLOCKED
    cone = world_labels([(5.0, 5.4, -0.2, 0.2, OBSTACLE)])
    r = check_corridor(cone, pose(0.0, 0.0), P)
    assert r.verdict[STRAIGHT] == BLOCKED and r.block_m[STRAIGHT] == pytest.approx(5.0, abs=0.25)
