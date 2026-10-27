import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.io.kitti_loader import load_calib, load_poses, load_scan
from src.mapping.foveated_grid import Band, FoveatedLayers, build_foveated_grid
from src.mapping.uniform_grid import GridLayers, build_uniform_grid, cell_centers, centered_spec, spec_from_bounds
from src.preprocess.filters import preprocess_mask
from src.preprocess.transforms import apply_transform, velo_poses_from_cam_poses
from src.temporal.accumulate import accumulate_world_points
from src.viz.grid_viz import plot_foveated

R = config.RANGE_MAX_M
REF_RES = 0.1
EVAL_RINGS = ((2.0, 10.0), (10.0, 25.0), (25.0, 50.0))
ERROR_STRIDE = 10

# name -> bands. single band = uniform lattice, same eval path.
CONFIGS: dict[str, tuple[Band, ...]] = {
    "U-20": (Band(0.0, R, 0.2),),
    "U-10": (Band(0.0, R, 0.1),),
    "F-spec 5/20/50": (Band(0.0, 10.0, 0.05), Band(10.0, 25.0, 0.2), Band(25.0, R, 0.5)),
    "F-A 10/20/50": (Band(0.0, 10.0, 0.1), Band(10.0, 25.0, 0.2), Band(25.0, R, 0.5)),
    "F-B 10/25/50": (Band(0.0, 15.0, 0.1), Band(15.0, 30.0, 0.25), Band(30.0, R, 0.5)),
}


# load + filter scans in velodyne frame
def load_filtered(paths: list[Path]) -> list[np.ndarray]:
    out = []
    for p in paths:
        pts = load_scan(p)
        out.append(pts[preprocess_mask(pts, config.RANGE_MIN_M, config.RANGE_MAX_M, config.EGO_BOX_VELO)][:, :3])
    return out


# time one grid build per scan. returns ms list.
def time_builds(scans: list[np.ndarray], bands: tuple[Band, ...]) -> np.ndarray:
    ms = []
    for pts in scans:
        t0 = time.perf_counter()
        if len(bands) == 1:
            build_uniform_grid(pts, centered_spec(2 * R, bands[0].resolution, "velo"), "velo", config.GROUND_PERCENTILE)
        else:
            build_foveated_grid(pts, bands, "velo", "velo", config.GROUND_PERCENTILE)
        ms.append((time.perf_counter() - t0) * 1e3)
    return np.array(ms)


# reference cells near scan, moved into scan velodyne frame: xy, ground z, z max z
def reference_probes(ref: GridLayers, t_world_velo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor = t_world_velo[:2, 3]
    centers = cell_centers(ref.spec)
    near = ref.observed & (np.hypot(centers[..., 0] - sensor[0], centers[..., 1] - sensor[1]) < R + 1.0)
    xy = centers[near]
    t_velo_world = np.linalg.inv(t_world_velo)
    # same world xy at two heights -> velodyne frame. ground point sets lookup xy.
    g = apply_transform(np.column_stack([xy, ref.ground[near]]), t_velo_world)
    m = apply_transform(np.column_stack([xy, ref.z_max[near]]), t_velo_world)
    return g[:, :2], g[:, 2], m[:, 2]


# per ring error stats of one map vs reference, pooled over scans
def ring_errors(fovs: list[FoveatedLayers], probes: list[tuple]) -> list[dict]:
    pooled = {ring: {"g": [], "m": [], "n": 0, "hit": 0} for ring in EVAL_RINGS}
    for fov, (xy, ref_g, ref_m) in zip(fovs, probes):
        r = np.hypot(xy[:, 0], xy[:, 1])
        got_g = fov.sample(xy, "ground")
        got_m = fov.sample(xy, "z_max")
        for ring in EVAL_RINGS:
            sel = (r >= ring[0]) & (r < ring[1])
            hit = sel & np.isfinite(got_g)
            pooled[ring]["n"] += int(sel.sum())
            pooled[ring]["hit"] += int(hit.sum())
            pooled[ring]["g"].append(got_g[hit] - ref_g[hit])
            pooled[ring]["m"].append(got_m[hit] - ref_m[hit])
    rows = []
    for ring, p in pooled.items():
        g, m = np.abs(np.concatenate(p["g"])), np.abs(np.concatenate(p["m"]))
        rows.append({
            "ring": f"{ring[0]:g}-{ring[1]:g} m",
            "coverage": p["hit"] / max(p["n"], 1),
            "g_med": np.median(g), "g_p95": np.percentile(g, 95),
            "m_med": np.median(m), "m_p95": np.percentile(m, 95),
        })
    return rows


# phase 2 eval: memory, latency, height error per ring for uniform vs foveated
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/sequences/00"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted((args.seq_dir / "velodyne").glob("*.bin"))
    ids = [int(p.stem) for p in paths]
    calib = load_calib(args.seq_dir / "calib.txt")
    poses = velo_poses_from_cam_poses(load_poses(args.seq_dir / "poses.txt"), calib["Tr"])[ids]
    scans = load_filtered(paths)

    world = accumulate_world_points(scans, poses)
    lo, hi = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
    ref_spec = spec_from_bounds(lo[0], hi[0], lo[1], hi[1], REF_RES, frame="world")
    ref = build_uniform_grid(world, ref_spec, points_frame="world", ground_percentile=config.GROUND_PERCENTILE)
    del world
    eval_ids = list(range(0, len(scans), ERROR_STRIDE))
    probes = [reference_probes(ref, poses[i]) for i in eval_ids]
    print(f"SCANS {len(scans)}. REFERENCE {REF_RES} M WORLD GRID, {int(ref.observed.sum())} CELLS. ERROR ON {len(eval_ids)} SCANS.")
    print(f"GROUND = P{config.GROUND_PERCENTILE:g}. ERRORS ARE |MAP - REFERENCE| IN METRES, VELODYNE FRAME.")
    print()

    summary, errors = [], {}
    for name, bands in CONFIGS.items():
        ms = time_builds(scans, bands)
        fov0 = build_foveated_grid(scans[0], bands, "velo", "velo", config.GROUND_PERCENTILE)
        if len(bands) == 1:
            dense = build_uniform_grid(scans[0], centered_spec(2 * R, bands[0].resolution, "velo"), "velo", config.GROUND_PERCENTILE)
            mem_mb, cells = dense.nbytes / 1e6, dense.count.size
        else:
            mem_mb, cells = fov0.nbytes / 1e6, fov0.n_cells
        summary.append((name, cells, mem_mb, np.median(ms[1:]), np.percentile(ms[1:], 95)))
        fovs = [build_foveated_grid(scans[i], bands, "velo", "velo", config.GROUND_PERCENTILE) for i in eval_ids]
        errors[name] = ring_errors(fovs, probes)
        if name.startswith("F-A"):
            plot_foveated(fov0, args.out / "seq00_scan000_foveated_FA.png", title=f"{name}  |  seq 00 scan 0  |  {fov0.nbytes / 1e6:.2f} MB")

    base_mb = summary[0][2]
    print("MEMORY + LATENCY (grid build only, preprocess excluded, scans 1..99)")
    print(f"| {'config':16s} | {'cells':>9s} | {'MB':>7s} | {'vs U-20':>7s} | {'ms med':>7s} | {'ms p95':>7s} |")
    print(f"|{'-' * 18}|{'-' * 11}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")
    for name, cells, mb, med, p95 in summary:
        print(f"| {name:16s} | {cells:9d} | {mb:7.2f} | {mb / base_mb:6.2f}x | {med:7.1f} | {p95:7.1f} |")
    print()
    print("HEIGHT ERROR PER RING (cov = share of observed reference cells this map also observed)")
    print(f"| {'config':16s} | {'ring':9s} | {'cov':>5s} | {'ground med':>10s} | {'ground p95':>10s} | {'zmax med':>8s} | {'zmax p95':>8s} |")
    print(f"|{'-' * 18}|{'-' * 11}|{'-' * 7}|{'-' * 12}|{'-' * 12}|{'-' * 10}|{'-' * 10}|")
    for name, rows in errors.items():
        for r in rows:
            print(f"| {name:16s} | {r['ring']:9s} | {r['coverage'] * 100:4.0f}% | {r['g_med']:10.3f} | {r['g_p95']:10.3f} | {r['m_med']:8.3f} | {r['m_p95']:8.3f} |")


if __name__ == "__main__":
    main()
