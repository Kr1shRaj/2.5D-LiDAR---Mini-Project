import numpy as np
import pytest

from src.mapping.foveated_grid import Band
from src.mapping.uniform_grid import build_uniform_grid, spec_from_bounds
from src.temporal.fusion import (
    FusionParams,
    TemporalFoveatedMap,
    decode_keys,
    empty_store,
    encode_keys,
    integrate_obs,
    merge_children,
    scan_observation,
    split_into_children,
)
from src.temporal.terrain import local_terrain

GROUND_Z = -1.73
BANDS = (Band(0.0, 6.0, 0.1), Band(6.0, 12.0, 0.2), Band(12.0, 20.0, 0.5))
PARAMS = FusionParams(tau_s=3.0, prune_weight=0.05, obstacle_h=0.3, l_hit=0.85, l_miss=-0.4,
                      l_min=-2.0, l_max=3.5, terrain_cell=2.0, ground_percentile=10.0)


# pose: yaw about z then shift. world = velo frame at start.
def pose(x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:2, 3] = [x, y]
    return t


# world points -> sensor (velodyne) frame for given pose
def to_velo(world: np.ndarray, t_world_velo: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(t_world_velo)
    return world @ inv[:3, :3].T + inv[:3, 3]


# seeded flat road around xy, sensor sees disk r < 20
def road(rng: np.random.Generator, cx: float, cy: float, n: int = 40000) -> np.ndarray:
    r = 20.0 * np.sqrt(rng.uniform(0.0, 1.0, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, GROUND_Z)])


# solid 1.5 m tall box (car) sampled on top and sides
def box(rng: np.random.Generator, x0: float, x1: float, y0: float, y1: float, n: int = 3000) -> np.ndarray:
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(GROUND_Z, GROUND_Z + 1.5, n)])


def test_key_roundtrip_negative() -> None:
    ix = np.array([-5, 0, 3, -2_000_000, 7])
    iy = np.array([4, -1, 0, 9, -3_000_000])
    gx, gy = decode_keys(encode_keys(ix, iy))
    assert gx.tolist() == ix.tolist() and gy.tolist() == iy.tolist()
    keys = encode_keys(ix, iy)
    assert np.array_equal(np.argsort(keys), np.lexsort((iy, ix)))  # sorted by ix then iy


def test_single_scan_matches_uniform_grid() -> None:
    rng = np.random.default_rng(seed=21)
    pts = road(rng, 0.0, 0.0)
    obs = scan_observation(pts, 0.2, 10.0)
    spec = spec_from_bounds(pts[:, 0].min(), pts[:, 0].max() + 1e-9, pts[:, 1].min(), pts[:, 1].max() + 1e-9, 0.2, "world")
    g = build_uniform_grid(pts, spec, "world", 10.0)
    assert obs.keys.size == int(g.observed.sum())
    assert int(obs.n.sum()) == len(pts)
    assert np.allclose(obs.ground, GROUND_Z)


def test_weighted_merge_matches_pooled_stats() -> None:
    store = empty_store(1.0)
    z_scans = [np.array([0.0, 1.0]), np.array([2.0, 4.0])]
    for t, z in enumerate(z_scans):
        pts = np.column_stack([np.full(len(z), 0.5), np.full(len(z), 0.5), z])
        obs = scan_observation(pts, 1.0, 50.0)
        store = integrate_obs(store, obs, np.array([np.nan]), float(t), PARAMS)
    # two scans weight 1 each: mean of scan means, var = mean within-var + var of means
    assert store.mean[0] == pytest.approx((0.5 + 3.0) / 2)
    assert store.m2[0] / store.weight[0] == pytest.approx((0.25 + 1.0) / 2 + 1.5625)
    assert store.ground[0] == pytest.approx((0.5 + 3.0) / 2)
    assert store.n_obs[0] == 2 and store.last_seen[0] == 1.0


def test_merge_children_keeps_weighted_mean_and_max() -> None:
    fine = empty_store(0.1)
    for x, z in ((0.05, 0.0), (0.15, 1.0)):
        obs = scan_observation(np.array([[x, 0.05, z]]), 0.1, 10.0)
        fine = integrate_obs(fine, obs, np.array([np.nan]), 0.0, PARAMS)
    coarse = merge_children(empty_store(0.2), fine)
    assert coarse.size == 1
    assert coarse.mean[0] == pytest.approx(0.5)
    assert coarse.z_max[0] == pytest.approx(1.0) and coarse.z_min[0] == pytest.approx(0.0)
    assert coarse.m2[0] / coarse.weight[0] == pytest.approx(0.25)
    assert coarse.weight[0] == pytest.approx(1.0)  # same scan, not double counted


def test_terrain_reaches_under_car() -> None:
    rng = np.random.default_rng(seed=22)
    car = box(rng, 4.0, 8.0, -1.0, 1.0)
    ground = road(rng, 0.0, 0.0)
    ground = ground[~((ground[:, 0] > 3.5) & (ground[:, 0] < 8.5) & (np.abs(ground[:, 1]) < 1.5))]  # occluded
    terrain = local_terrain(np.vstack([ground, car]), "world", 2.0, 10.0)
    assert terrain.sample(np.array([[6.0, 0.0]])) == pytest.approx([GROUND_Z])


def test_static_scene_coverage_grows_and_ground_stays() -> None:
    rng = np.random.default_rng(seed=23)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    observed = []
    for i in range(6):
        p = pose(0.8 * i, 0.0, yaw=0.05 * i)
        world = road(rng, p[0, 3], 0.0, n=6000)
        m.integrate(to_velo(world, p), p, t=0.1 * i)
        fov = m.render()
        observed.append(sum(int(b.observed.sum()) for b in fov.bands[:1]))
        for b in fov.bands:
            assert np.allclose(b.ground[b.observed], GROUND_Z)
    assert observed[-1] > 2 * observed[0]
    assert m.render().center == pytest.approx((4.0, 0.0))


def test_moving_car_trail_cleared_static_car_kept() -> None:
    rng = np.random.default_rng(seed=24)
    naive = TemporalFoveatedMap(BANDS, FusionParams(**{**PARAMS.__dict__, "clear_ghosts": False}))
    fused = TemporalFoveatedMap(BANDS, PARAMS)
    p = pose(0.0, 0.0)
    for i in range(25):
        car_x = -8.0 + 1.5 * i if i < 6 else 50.0  # drives past in 0.6 s, then gone
        world = road(rng, 0.0, 0.0)
        world = np.vstack([world, box(rng, 3.0, 5.0, -9.0, -7.0)])  # parked car
        if car_x < 50.0:
            world = np.vstack([world, box(rng, car_x, car_x + 4.0, 3.0, 5.0)])
        for m in (naive, fused):
            m.integrate(to_velo(world, p), p, t=0.1 * i)
    trail = np.array([[x, 4.0] for x in np.arange(-6.0, 3.0, 0.5)])
    parked = np.array([[4.0, -8.0]])
    tall = GROUND_Z + 0.3
    assert np.all(naive.render().sample(trail, "z_max") > tall)  # naive keeps ghost trail
    assert np.all(fused.render().sample(trail, "z_max") < tall)  # fused clears it
    assert np.all(fused.render().sample(parked, "z_max") > tall)  # real obstacle stays
    assert np.all(fused.render().sample(parked, "logodds") > 0)


def test_stale_cell_decays_to_unknown() -> None:
    rng = np.random.default_rng(seed=25)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    p = pose(0.0, 0.0)
    m.integrate(to_velo(road(rng, 0.0, 0.0), p), p, t=0.0)
    q = np.array([[14.0, 0.0]])  # 0.5 m cell, many points
    assert np.isfinite(m.render().sample(q, "ground")).all()
    m.integrate(np.empty((0, 3)), p, t=0.5)
    assert m.render().sample(q, "weight") == pytest.approx([np.exp(-0.5 / 3.0)])
    m.integrate(np.empty((0, 3)), p, t=30.0)  # 10 tau unseen
    assert np.isnan(m.render().sample(q, "ground")).all()
    assert m.n_cells == 0


def test_handoff_moves_fine_history_to_coarse() -> None:
    rng = np.random.default_rng(seed=26)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    p0 = pose(0.0, 0.0)
    world = np.vstack([road(rng, 0.0, 0.0), box(rng, 2.0, 3.0, 2.0, 3.0, n=500)])
    m.integrate(to_velo(world, p0), p0, t=0.0)
    n_fine = m.stores[0].size
    p1 = pose(5.0, 0.0)  # box now ~ 3.6..4.2 m away -> still fine; move further
    p2 = pose(10.0, 0.0)  # box ~ 7.5 m away -> mid ring
    m.integrate(np.empty((0, 3)), p1, t=0.1)
    m.integrate(np.empty((0, 3)), p2, t=0.2)
    assert int((~m.stores[0].inherited).sum()) < n_fine  # real fine cells left ring
    top = m.render().sample(np.array([[2.5, 2.5]]), "z_max")
    assert top == pytest.approx([GROUND_Z + 1.5], abs=0.05)  # old fine obstacle kept after hand-off
    band_idx, _ = m.render().lookup(np.array([[2.5, 2.5]]))
    assert band_idx.tolist() == [1]


def test_split_children_cover_parent_and_merge_back() -> None:
    for coarse_res, fine_res, kids in ((0.2, 0.1, 4), (0.5, 0.1, 25)):
        parent = empty_store(coarse_res)
        obs = scan_observation(np.array([[-0.3, 0.7, 1.0], [-0.3, 0.7, 2.0]]), coarse_res, 10.0)
        parent = integrate_obs(parent, obs, np.array([np.nan]), 0.0, PARAMS)
        ch = split_into_children(parent, fine_res)
        assert ch.size == kids and ch.inherited.all()
        assert np.allclose(ch.mean, 1.5) and np.allclose(ch.z_max, 2.0)
        back = merge_children(empty_store(coarse_res), ch)
        assert back.size == 1 and back.keys[0] == parent.keys[0]
        assert back.mean[0] == pytest.approx(1.5) and back.inherited[0]


def test_split_non_nested_lattice_partitions_children() -> None:
    parents = empty_store(0.5)
    xy = np.array([[x + 0.25, 0.25] for x in np.arange(-2.0, 2.0, 0.5)])
    obs = scan_observation(np.column_stack([xy, np.zeros(len(xy))]), 0.5, 10.0)
    parents = integrate_obs(parents, obs, np.full(len(xy), np.nan), 0.0, PARAMS)
    ch = split_into_children(parents, 0.2)
    assert np.unique(ch.keys).size == ch.size  # no child shared by two parents
    ix, _ = decode_keys(ch.keys)
    assert sorted(set(ix.tolist())) == list(range(-10, 10))  # every 0.2 m column from -2 to 2 m covered once


def test_first_fine_scan_replaces_inherited_stats() -> None:
    parent = empty_store(0.2)
    obs = scan_observation(np.array([[0.1, 0.1, 5.0]]), 0.2, 10.0)
    parent = integrate_obs(parent, obs, np.array([GROUND_Z]), 0.0, PARAMS)  # tall: hit
    ch = split_into_children(parent, 0.1)
    assert ch.logodds[0] > 0
    fine_obs = scan_observation(np.array([[0.05, 0.05, GROUND_Z]]), 0.1, 10.0)
    ch = integrate_obs(ch, fine_obs, np.array([GROUND_Z]), 0.1, PARAMS)
    k = int(np.searchsorted(ch.keys, fine_obs.keys[0]))
    assert not ch.inherited[k]
    assert ch.z_max[k] == pytest.approx(GROUND_Z) and ch.ground[k] == pytest.approx(GROUND_Z)
    assert ch.n_obs[k] == 1 and ch.weight[k] == pytest.approx(1.0) and ch.logodds[k] < 0
    assert ch.inherited.sum() == 3  # siblings still placeholders


def test_cells_ahead_prefilled_from_coarse_history() -> None:
    rng = np.random.default_rng(seed=27)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    p0 = pose(0.0, 0.0)
    m.integrate(to_velo(road(rng, 0.0, 0.0), p0), p0, t=0.0)
    ahead = np.array([[8.0, 0.0], [9.0, 1.0]])  # mid ring now
    assert np.isfinite(m.render().sample(ahead, "ground")).all()
    p1 = pose(4.0, 0.0)  # drive forward, no scan: those spots now in fine ring
    m.integrate(np.empty((0, 3)), p1, t=0.1)
    fov = m.render()
    band_idx, _ = fov.lookup(ahead)
    assert band_idx.tolist() == [0, 0]
    assert np.allclose(fov.sample(ahead, "ground"), GROUND_Z)
    assert np.all(fov.sample(ahead, "inherited") == 1.0)
