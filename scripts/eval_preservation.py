import argparse
from pathlib import Path

import numpy as np

from src import config
from src.io.kitti_loader import load_calib, load_poses
from src.mapping.features import ground_step_edges
from src.mapping.foveated_grid import Band, FoveatedLayers, build_foveated_grid
from src.mapping.uniform_grid import (
    GridLayers,
    build_uniform_grid,
    cell_centers,
    centered_spec,
    sample_layer,
    spec_from_bounds,
)
from src.preprocess.transforms import apply_transform, velo_poses_from_cam_poses
from src.temporal.accumulate import accumulate_world_points
from scripts.eval_foveated import load_filtered

R = config.RANGE_MAX_M
REF_RES = 0.1
TRUTH_RES = 0.05
EVAL_RINGS = ((2.0, 10.0), (10.0, 25.0), (25.0, 50.0))
ERROR_STRIDE = 10

STEP_MIN_M = 0.08  # curb-like step: 8..30 cm. taller = wall / car, not curb
STEP_MAX_M = 0.30
SIDE_FLAT_STD_M = 0.05  # both sides of step must be smooth in reference
PROBE_D = (0.1, 0.2)  # probe offset each side of edge: 20 cm and 40 cm apart
KEEP_RATIO = 0.5  # step kept if map shows >= half reference step, same sign
OBSTACLE_H_M = 0.3  # z max this far above reference terrain = obstacle cell

CONFIGS: dict[str, tuple[Band, ...]] = {
    "U-20": (Band(0.0, R, 0.2),),
    "U-10": (Band(0.0, R, 0.1),),
    "F-A 10/20/50": config.FOVEATED_BANDS,
}


# ring label for each velodyne xy
def ring_of(xy: np.ndarray) -> np.ndarray:
    r = np.hypot(xy[:, 0], xy[:, 1])
    out = np.full(len(xy), -1)
    for k, (lo, hi) in enumerate(EVAL_RINGS):
        out[(r >= lo) & (r < hi)] = k
    return out


# reference curb edges in world frame that stay flat on both sides at probe offset d
def reference_edges(ref: GridLayers, d: float) -> tuple[np.ndarray, np.ndarray]:
    xy, n, _ = ground_step_edges(ref, STEP_MIN_M, STEP_MAX_M)
    hi_side, lo_side = xy + d * n, xy - d * n
    step = sample_layer(ref, hi_side, "ground") - sample_layer(ref, lo_side, "ground")
    flat = (sample_layer(ref, hi_side, "roughness") < SIDE_FLAT_STD_M) & (
        sample_layer(ref, lo_side, "roughness") < SIDE_FLAT_STD_M
    )
    keep = flat & (step >= STEP_MIN_M) & (step <= STEP_MAX_M)
    return xy[keep], n[keep]


# world probe at reference ground height -> velodyne frame xy and z
def to_velo(ref: GridLayers, xy_world: np.ndarray, t_velo_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = apply_transform(np.column_stack([xy_world, sample_layer(ref, xy_world, "ground")]), t_velo_world)
    return p[:, :2], p[:, 2]


# step stats for one scan: per edge ring, observed, kept, ratio, cell-flag
def step_scores(fov: FoveatedLayers, ref: GridLayers, edges_xy: np.ndarray, edges_n: np.ndarray,
                d: float, t_world_velo: np.ndarray) -> dict[str, np.ndarray]:
    sensor = t_world_velo[:2, 3]
    near = np.hypot(*(edges_xy - sensor).T) < R + 1.0
    xy, n = edges_xy[near], edges_n[near]
    t_velo_world = np.linalg.inv(t_world_velo)
    hi_xy, hi_z = to_velo(ref, xy + d * n, t_velo_world)
    lo_xy, lo_z = to_velo(ref, xy - d * n, t_velo_world)
    mid_xy, _ = to_velo(ref, xy, t_velo_world)
    ref_step = hi_z - lo_z
    map_step = fov.sample(hi_xy, "ground") - fov.sample(lo_xy, "ground")
    observed = np.isfinite(map_step)
    kept = observed & (map_step >= KEEP_RATIO * ref_step)
    # coarse cell may hold whole step inside: z max - ground of cell on edge
    cell_relief = fov.sample(mid_xy, "z_max") - fov.sample(mid_xy, "ground")
    flagged = np.isfinite(cell_relief) & (cell_relief >= KEEP_RATIO * ref_step)
    return {"ring": ring_of(mid_xy), "observed": observed, "kept": kept,
            "ratio": np.where(observed, map_step / ref_step, np.nan), "flagged": flagged}


# fine 5 cm truth cells of one scan with terrain from reference. returns xy, true obstacle, terrain z.
def obstacle_truth(scan: np.ndarray, ref: GridLayers, t_world_velo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fine = build_uniform_grid(scan, centered_spec(2 * R, TRUTH_RES, "velo"), "velo", config.GROUND_PERCENTILE)
    obs = fine.observed
    xy_v = cell_centers(fine.spec)[obs]
    # cell centre at its own ground -> world xy -> reference terrain -> back to velodyne z
    w = apply_transform(np.column_stack([xy_v, fine.ground[obs]]), t_world_velo)
    terrain_w = sample_layer(ref, w[:, :2], "ground")
    terrain_v = apply_transform(np.column_stack([w[:, :2], terrain_w]), np.linalg.inv(t_world_velo))[:, 2]
    known = np.isfinite(terrain_w)
    xy_v, zmax_v, terrain_v = xy_v[known], fine.z_max[obs][known], terrain_v[known]
    return xy_v, (zmax_v - terrain_v) >= OBSTACLE_H_M, terrain_v


# print one markdown table
def table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


# phase 2: do foveated cells keep curbs, steps, obstacles vs uniform baseline
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/sequences/00"))
    args = ap.parse_args()

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
    maps = {name: [build_foveated_grid(scans[i], bands, "velo", "velo", config.GROUND_PERCENTILE) for i in eval_ids]
            for name, bands in CONFIGS.items()}
    print(f"SCANS {len(scans)}. REFERENCE {REF_RES} M WORLD GRID. SCORED ON {len(eval_ids)} SCANS.")
    print()

    for d in PROBE_D:
        edges_xy, edges_n = reference_edges(ref, d)
        print(f"STEPS {STEP_MIN_M * 100:g}-{STEP_MAX_M * 100:g} CM, PROBES +-{d * 100:g} CM ({2 * d * 100:g} CM APART). "
              f"{len(edges_xy)} REFERENCE EDGE CELLS.")
        print(f"obs = both probes observed. kept = map step >= {KEEP_RATIO:g}x ref step (given obs). "
              f"cell flag = edge cell z_max - ground >= {KEEP_RATIO:g}x ref step.")
        rows = []
        for name in CONFIGS:
            per = [step_scores(m, ref, edges_xy, edges_n, d, poses[i]) for m, i in zip(maps[name], eval_ids)]
            cat = {k: np.concatenate([p[k] for p in per]) for k in per[0]}
            for k, (rlo, rhi) in enumerate(EVAL_RINGS):
                s = cat["ring"] == k
                o = s & cat["observed"]
                rows.append([name, f"{rlo:g}-{rhi:g} m", f"{s.sum()}", f"{o.sum() / max(s.sum(), 1) * 100:.0f}%",
                             f"{cat['kept'][o].mean() * 100:.0f}%" if o.any() else "-",
                             f"{np.nanmedian(cat['ratio'][o]):.2f}" if o.any() else "-",
                             f"{cat['flagged'][s].mean() * 100:.0f}%" if s.any() else "-"])
        table(["config", "ring", "edges", "obs", "kept", "step ratio med", "cell flag"], rows)

    print(f"OBSTACLES: 5 CM SCAN CELL WITH Z MAX >= {OBSTACLE_H_M * 100:g} CM OVER REFERENCE TERRAIN = TRUE OBSTACLE.")
    print("recall = true obstacle cells whose map cell also flags. inflation = flagged area / true area (5 cm cells observed).")
    truths = [obstacle_truth(scans[i], ref, poses[i]) for i in eval_ids]
    rows = []
    for name in CONFIGS:
        acc = {k: [0, 0, 0] for k in range(len(EVAL_RINGS))}
        for m, (xy_v, true, terrain_v) in zip(maps[name], truths):
            flag = (m.sample(xy_v, "z_max") - terrain_v) >= OBSTACLE_H_M
            ring = ring_of(xy_v)
            for k in acc:
                s = ring == k
                acc[k][0] += int((s & true).sum())
                acc[k][1] += int((s & true & flag).sum())
                acc[k][2] += int((s & flag).sum())
        for k, (rlo, rhi) in enumerate(EVAL_RINGS):
            t, tp, f = acc[k]
            rows.append([name, f"{rlo:g}-{rhi:g} m", f"{t * TRUTH_RES**2:.0f}", f"{tp / max(t, 1) * 100:.1f}%",
                         f"{f / max(t, 1):.2f}x"])
    table(["config", "ring", "true area m2", "recall", "inflation"], rows)


if __name__ == "__main__":
    main()
