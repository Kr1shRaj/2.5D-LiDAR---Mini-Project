import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src import config
from src.io.kitti_loader import load_calib, load_poses, load_scan
from src.mapping.uniform_grid import GridLayers, GridSpec, build_uniform_grid, cell_centers, spec_from_bounds
from src.preprocess.filters import preprocess_mask
from src.preprocess.transforms import velo_poses_from_cam_poses, velo_to_world
from src.temporal.accumulate import accumulate_world_points
from src.viz.grid_viz import plot_elevation_and_roughness

CORRIDOR_M = 15.0
FLAT_STD_M = 0.05  # reference cell this smooth = flat surface. isolates pose error from structure
RESIDUAL_STRIDE = 10


# mask of cells within radius of any sensor position
def near_path_mask(spec: GridSpec, path_xy: np.ndarray, radius: float) -> np.ndarray:
    centers = cell_centers(spec).reshape(-1, 2)
    dist, _ = cKDTree(path_xy).query(centers)
    return (dist < radius).reshape(spec.shape)


# load + filter scans, keep velodyne frame
def load_filtered(paths: list[Path]) -> list[np.ndarray]:
    out = []
    for p in paths:
        pts = load_scan(p)
        out.append(pts[preprocess_mask(pts, config.RANGE_MIN_M, config.RANGE_MAX_M, config.EGO_BOX_VELO)])
    return out


# ground diff of one scan vs reference on shared cells near that scan
def ground_residual(single: GridLayers, ref: GridLayers, near: np.ndarray) -> np.ndarray:
    both = single.observed & ref.observed & near
    return single.ground[both] - ref.ground[both]


# build dense world frame reference map from seq slice. report coverage + alignment.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/sequences/00"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--resolution", type=float, default=0.2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted((args.seq_dir / "velodyne").glob("*.bin"))
    ids = [int(p.stem) for p in paths]
    calib = load_calib(args.seq_dir / "calib.txt")
    poses = velo_poses_from_cam_poses(load_poses(args.seq_dir / "poses.txt"), calib["Tr"])[ids]
    path_xy = poses[:, :2, 3]

    t0 = time.perf_counter()
    scans = load_filtered(paths)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    world = accumulate_world_points(scans, poses)
    lo = world[:, :2].min(axis=0)
    hi = world[:, :2].max(axis=0)
    spec = spec_from_bounds(lo[0], hi[0], lo[1], hi[1], args.resolution, frame="world")
    ref = build_uniform_grid(world, spec, points_frame="world", ground_percentile=config.GROUND_PERCENTILE)
    t_build = time.perf_counter() - t0

    print(f"SCANS {len(scans)} ({ids[0]:06d}..{ids[-1]:06d}). PATH {np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)):.1f} M")
    print(f"POINTS {len(world)} ({world.nbytes / 1e6:.0f} MB XYZ F32). LOAD {t_load:.1f} S. ACCUMULATE + GRID {t_build:.1f} S")
    print(f"GRID {spec.nx}x{spec.ny} @ {spec.resolution} M = {spec.nx * spec.resolution:.0f}x{spec.ny * spec.resolution:.0f} M. {ref.nbytes / 1e6:.1f} MB")

    corridor = near_path_mask(spec, path_xy, CORRIDOR_M)
    first = build_uniform_grid(velo_to_world(scans[0][:, :3], poses[0]), spec, points_frame="world", ground_percentile=config.GROUND_PERCENTILE)
    near0 = near_path_mask(spec, path_xy[:1], CORRIDOR_M)
    print(f"COVERAGE <{CORRIDOR_M:.0f} M:")
    print(f"  SCAN 0 ALONE, AROUND SCAN 0   {first.observed[near0].mean() * 100:5.1f} %")
    print(f"  ACCUMULATED, AROUND SCAN 0    {ref.observed[near0].mean() * 100:5.1f} %")
    print(f"  ACCUMULATED, WHOLE CORRIDOR   {ref.observed[corridor].mean() * 100:5.1f} %")
    print(f"  POINTS PER OBSERVED CELL      median {np.median(ref.count[ref.observed & corridor]):.0f}")

    print(f"ALIGN: SCAN GROUND MINUS REFERENCE GROUND, CELLS <{CORRIDOR_M:.0f} M OF SCAN (P{config.GROUND_PERCENTILE:.0f}):")
    all_res, flat_res = [], []
    flat = ref.observed & (ref.roughness < FLAT_STD_M)
    for i in range(0, len(scans), RESIDUAL_STRIDE):
        single = build_uniform_grid(velo_to_world(scans[i][:, :3], poses[i]), spec, points_frame="world", ground_percentile=config.GROUND_PERCENTILE)
        near_i = near_path_mask(spec, path_xy[i : i + 1], CORRIDOR_M)
        res = ground_residual(single, ref, near_i)
        all_res.append(res)
        flat_res.append(ground_residual(single, ref, near_i & flat))
        print(f"  SCAN {ids[i]:06d}  cells {len(res):5d}  median {np.median(res):+.3f}  |p95| {np.percentile(np.abs(res), 95):.3f}")
    all_res = np.concatenate(all_res)
    flat_res = np.concatenate(flat_res)
    print(f"  FLAT ONLY cells {len(flat_res):5d}  median {np.median(flat_res):+.3f}  |median| {np.median(np.abs(flat_res)):.3f}  |p95| {np.percentile(np.abs(flat_res), 95):.3f}")
    print(f"  ALL       cells {len(all_res):5d}  median {np.median(all_res):+.3f}  |median| {np.median(np.abs(all_res)):.3f}  |p95| {np.percentile(np.abs(all_res), 95):.3f}")

    np.savez_compressed(
        args.out / "seq00_reference_map.npz",
        x_min=spec.x_min, y_min=spec.y_min, resolution=spec.resolution, frame=spec.frame,
        count=ref.count, z_min=ref.z_min, z_max=ref.z_max, z_mean=ref.z_mean, z_var=ref.z_var, ground=ref.ground,
        ground_percentile=ref.ground_percentile, scan_ids=np.array(ids), poses_world_velo=poses,
    )
    plot_elevation_and_roughness(ref, args.out / "seq00_reference_map.png", path_xy=path_xy)
    print(f"SAVED {args.out / 'seq00_reference_map.npz'} + PNG")


if __name__ == "__main__":
    main()
