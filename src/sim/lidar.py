from dataclasses import dataclass

import numpy as np

from src.sim.world import Box

# SIMULATION DEMO (NON-BENCHMARK). virtual spinning lidar, HDL-64E-like, raycast against flat ground + boxes.
# output points are in the VELODYNE frame of the virtual car (x fwd, y left, z up), same as KITTI .bin xyz.
# frame chain: velo ray -> world by T_world_velo (rotation about z by yaw, translation (x, y, mount height)).
# sim ground is world z = 0, so flat road sits at velo z = -mount_h_m (-1.73 m), as on KITTI.


@dataclass(frozen=True)
class LidarModel:
    mount_h_m: float = 1.73  # sensor height over road, as KITTI HDL-64E
    n_azimuth: int = 1024  # columns per turn (~0.35 deg)
    max_range_m: float = 60.0  # map uses 50 m; a bit more keeps the outer ring edge honest
    range_sigma_m: float = 0.02  # gaussian range noise
    # 64 beams: upper block +2.0..-8.33 deg (32 beams), lower block -8.83..-24.8 deg (32 beams)
    elevations_deg: tuple[float, ...] = tuple(np.r_[np.linspace(2.0, -8.33, 32), np.linspace(-8.83, -24.8, 32)])


# unit ray directions in velo frame, (n_beams * n_azimuth, 3)
def ray_dirs(m: LidarModel) -> np.ndarray:
    el = np.radians(np.array(m.elevations_deg))[:, None]
    az = (np.arange(m.n_azimuth) / m.n_azimuth * 2 * np.pi)[None, :]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az)], axis=-1)
    return d.reshape(-1, 3)


# 4x4 T_world_velo for virtual car at (x, y) heading yaw. sensor mounted mount_h_m over flat ground z = 0.
def pose_matrix(x: float, y: float, yaw: float, m: LidarModel) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, m.mount_h_m]
    return T


# distance along each world ray to first hit of oriented box, inf = miss. slab test in box frame.
def hit_box(o: np.ndarray, d: np.ndarray, b: Box) -> np.ndarray:
    c, s = np.cos(-b.yaw), np.sin(-b.yaw)
    ox, oy = o[0] - b.cx, o[1] - b.cy
    lo = np.array([c * ox - s * oy, s * ox + c * oy, o[2]])
    ld = np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]])
    bmin = np.array([-b.length / 2, -b.width / 2, b.z0])
    bmax = np.array([b.length / 2, b.width / 2, b.z1])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / ld
        t1, t2 = (bmin - lo) * inv, (bmax - lo) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0.0), np.inf)


# one scan from virtual car pose: (n, 3) velo-frame points of first hits (ground or box), seeded range noise.
# speed: each box only tests ray columns inside its azimuth span seen from the sensor (+ one column margin).
def scan(T_world_velo: np.ndarray, boxes: list[Box], m: LidarModel, rng: np.random.Generator,
         dirs: np.ndarray | None = None) -> np.ndarray:
    dv = ray_dirs(m) if dirs is None else dirs
    R, o = T_world_velo[:3, :3], T_world_velo[:3, 3]
    dw = dv @ R.T
    with np.errstate(divide="ignore"):
        t = np.where(dw[:, 2] < 0, -o[2] / dw[:, 2], np.inf)  # flat ground z = 0
    n_az, n_el = m.n_azimuth, len(m.elevations_deg)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cols = np.arange(n_az)
    for b in boxes:
        dist = np.hypot(b.cx - o[0], b.cy - o[1])
        rad = 0.5 * np.hypot(b.length, b.width)
        if dist - rad > m.max_range_m:
            continue
        if dist <= rad + 0.5:
            idx = np.arange(n_el * n_az)  # sensor inside / touching box footprint: test all rays
        else:
            az = np.arctan2(b.cy - o[1], b.cx - o[0]) - yaw  # box centre azimuth in velo frame
            span = np.arcsin(min(rad / dist, 1.0)) + 2 * np.pi / n_az
            c0 = az / (2 * np.pi) * n_az
            dc = (cols - c0 + n_az / 2) % n_az - n_az / 2  # wrapped column offset
            sel = cols[np.abs(dc) <= span / (2 * np.pi) * n_az]
            idx = (np.arange(n_el)[:, None] * n_az + sel[None, :]).ravel()
        if idx.size:
            t[idx] = np.minimum(t[idx], hit_box(o, dw[idx], b))
    keep = t < m.max_range_m
    r = t[keep] + rng.normal(0.0, m.range_sigma_m, keep.sum())
    return dv[keep] * r[:, None]
