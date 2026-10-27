import argparse
from pathlib import Path

import numpy as np

from src import config
from src.io.kitti_loader import load_calib, load_poses, load_scan
from src.mapping.uniform_grid import GridLayers, build_uniform_grid, cell_centers, centered_spec
from src.preprocess.filters import preprocess_mask
from src.preprocess.transforms import velo_poses_from_cam_poses, velo_to_world
from src.viz.grid_viz import plot_elevation_and_roughness

NEAR_MAX_M = 15.0


# near field cell ground heights within NEAR_MAX_M of (cx, cy)
def near_ground(layers: GridLayers, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    centers = cell_centers(layers.spec)
    r = np.hypot(centers[..., 0] - cx, centers[..., 1] - cy)
    return layers.ground[layers.observed & (r < NEAR_MAX_M)]


# check real ground sits near -1.73 m in velodyne frame across slice
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/sequences/00"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--resolution", type=float, default=0.2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scans = sorted((args.seq_dir / "velodyne").glob("*.bin"))
    spec = centered_spec(extent=2 * config.RANGE_MAX_M, resolution=args.resolution, frame="velo")
    medians, q25s, q75s = [], [], []
    for i, path in enumerate(scans):
        pts = load_scan(path)
        keep = preprocess_mask(pts, config.RANGE_MIN_M, config.RANGE_MAX_M, config.EGO_BOX_VELO)
        layers = build_uniform_grid(pts[keep], spec, points_frame="velo", ground_percentile=config.GROUND_PERCENTILE)
        q25, med, q75 = np.percentile(near_ground(layers), [25, 50, 75])
        medians.append(med)
        q25s.append(q25)
        q75s.append(q75)
        if i == 0:
            print(f"SCAN 0: {len(pts)} PTS, KEEP {keep.sum()}, OBSERVED CELLS {layers.observed.sum()}/{layers.count.size}")
            print(f"SCAN 0: GRID {layers.nbytes / 1e6:.1f} MB")
            plot_elevation_and_roughness(layers, args.out / "seq00_scan000_velo.png")
    medians = np.array(medians)
    print(f"SCANS {len(scans)}. NEAR <{NEAR_MAX_M:.0f} M CELL GROUND (P{config.GROUND_PERCENTILE:.0f}), VELO FRAME:")
    print(f"  MEDIAN  mean {medians.mean():+.3f}  min {medians.min():+.3f}  max {medians.max():+.3f}")
    print(f"  IQR     mean {np.mean(q25s):+.3f} .. {np.mean(q75s):+.3f}")
    print(f"  SCAN {len(scans) - 1} MEDIAN {medians[-1]:+.3f}")

    # world frame check: pose chain should move sensor along road, keep ground near -1.73 below sensor
    calib = load_calib(args.seq_dir / "calib.txt")
    poses = velo_poses_from_cam_poses(load_poses(args.seq_dir / "poses.txt"), calib["Tr"])
    last = len(scans) - 1
    t = poses[last][:3, 3]
    print(f"POSES {len(poses)}. VELO POSE {last} IN WORLD: x {t[0]:+.2f} y {t[1]:+.2f} z {t[2]:+.2f}")
    pts = load_scan(scans[last])
    keep = preprocess_mask(pts, config.RANGE_MIN_M, config.RANGE_MAX_M, config.EGO_BOX_VELO)
    world = velo_to_world(pts[keep], poses[last])
    wspec = centered_spec(extent=2 * config.RANGE_MAX_M, resolution=args.resolution, frame="world", cx=t[0], cy=t[1])
    wl = build_uniform_grid(world, wspec, points_frame="world", ground_percentile=config.GROUND_PERCENTILE)
    g = float(np.median(near_ground(wl, t[0], t[1])))
    print(f"SCAN {last} WORLD GROUND MEDIAN {g:+.3f}. MINUS SENSOR Z = {g - t[2]:+.3f}")


if __name__ == "__main__":
    main()
