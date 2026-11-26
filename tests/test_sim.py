import numpy as np

from src.planning.corridor import BLOCKED, CAUTION, CLEAR, CorridorResult
from src.sim.drive import CREEP, DRIVE, STOP, CarState, DriveParams, choose_arc, contacts, step
from src.sim.lidar import LidarModel, hit_box, pose_matrix, ray_dirs, scan
from src.sim.world import Box, build_route, demo_world

# SIMULATION DEMO (NON-BENCHMARK) tests: geometry + frame convention + determinism of the sim pieces only.

M = LidarModel()


# flat road sits at velo z = -mount height (-1.73 m), same convention as KITTI
def test_flat_ground_at_minus_mount_height() -> None:
    pts = scan(pose_matrix(0.0, 0.0, 0.3, M), [], M, np.random.default_rng(0))
    assert np.all(pts[:, 2] < 0)
    assert abs(np.median(pts[:, 2]) + M.mount_h_m) < 0.01


# box straight ahead: first hit at its near face, ground behind it hidden
def test_box_occludes_and_range_is_exact() -> None:
    box = Box(12.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0, "car")
    o = np.array([0.0, 0.0, M.mount_h_m])
    assert np.isclose(hit_box(o, np.array([[1.0, 0.0, 0.0]]), box)[0], 11.0)
    pts = scan(pose_matrix(0.0, 0.0, 0.0, M), [box], M, np.random.default_rng(0))
    ahead = pts[(np.abs(pts[:, 1]) < 0.5) & (pts[:, 0] > 0)]
    assert ahead[:, 0].max() < 11.1  # nothing seen past the near face in that column
    assert np.any(ahead[:, 2] > -1.0)  # box face points above ground


# azimuth culling gives same hits as testing every ray against every box
def test_culling_matches_brute_force() -> None:
    w = demo_world(0)
    p, h = w.at_station(40.0)
    T = pose_matrix(p[0], p[1], h, M)
    d = ray_dirs(M)
    fast = scan(T, w.boxes(0.0), M, np.random.default_rng(1), d)
    R, o = T[:3, :3], T[:3, 3]
    dw = d @ R.T
    t = np.where(dw[:, 2] < 0, -o[2] / dw[:, 2], np.inf)
    for b in w.boxes(0.0):
        t = np.minimum(t, hit_box(o, dw, b))
    assert fast.shape[0] == int(np.sum(t < M.max_range_m))


# same seed, same scan; route turns sum to built angle
def test_deterministic_and_route_heading() -> None:
    T = pose_matrix(3.0, -1.0, 0.1, M)
    w = demo_world(0)
    a = scan(T, w.boxes(0.0), M, np.random.default_rng(7))
    b = scan(T, w.boxes(0.0), M, np.random.default_rng(7))
    assert np.array_equal(a, b)
    _, heading, _ = build_route([("straight", 10.0), ("turn", 90.0, 20.0), ("turn", -45.0, 10.0)])
    assert np.isclose(np.degrees(heading[-1]), 45.0, atol=1.0)


# arc choice: nearest route among pickable; creep only when rejected for unseen, never past a block
def test_choose_arc_modes() -> None:
    k = np.linspace(-0.1, 0.1, 5)
    r = CorridorResult(curvatures=k, verdict=np.array([CLEAR, CLEAR, BLOCKED, CAUTION, CLEAR]),
                       block_m=np.full(5, 20.0), hazard=np.zeros(5), unknown=np.zeros(5),
                       score=np.array([1.0, 1.0, -np.inf, 1.0, 1.0]), pick=0)
    p = DriveParams()
    assert choose_arc(r, 0.0, 20.0, p) == (1, DRIVE)  # straight blocked: neighbour clear arc, not caution one
    r.score[:] = -np.inf
    assert choose_arc(r, 0.0, 20.0, p)[1] == CREEP
    assert choose_arc(r, 0.0, 20.0, p)[0] != 2
    r.verdict[:] = BLOCKED
    assert choose_arc(r, 0.0, 20.0, p) == (-1, STOP)


# kinematics: rate-limited curvature, speed limits; footprint contact with box in path
def test_step_and_contact() -> None:
    p = DriveParams()
    c = step(CarState(0.0, 0.0, 0.0, 5.0), 0.1, 20.0, 0.1, p)
    assert np.isclose(c.k, p.dk_max) and np.isclose(c.v, 5.0 + p.accel * 0.1)
    ego = (-1.2, 2.8, -0.9, 0.9)
    assert contacts(CarState(0.0, 0.0, 0.0, 0.0), ego, [Box(2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.5, "car")]) == ["car"]
    assert contacts(CarState(0.0, 0.0, 0.0, 0.0), ego, [Box(6.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.5, "car")]) == []
