import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.io.kitti_loader import load_calib, load_poses, load_times
from src.mapping.foveated_grid import Band, FoveatedLayers
from src.mapping.uniform_grid import GridLayers, build_uniform_grid, cell_centers, sample_layer, spec_from_bounds
from src.preprocess.transforms import velo_poses_from_cam_poses, velo_to_world
from src.temporal.accumulate import accumulate_world_points
from src.temporal.fusion import FusionParams, TemporalFoveatedMap
from src.temporal.terrain import local_terrain
from src.viz.grid_viz import plot_foveated
from scripts.eval_foveated import load_filtered
from scripts.eval_preservation import KEEP_RATIO, PROBE_D, reference_edges

R = config.RANGE_MAX_M
REF_RES = 0.1
TRUTH_RES = 0.05
EVAL_RINGS = ((2.0, 10.0), (10.0, 25.0), (25.0, 50.0))
EVAL_FROM = 10  # give fused maps 1 s of history before scoring
EVAL_STRIDE = 10
H = config.OBSTACLE_H_M

PARAMS = FusionParams(
    tau_s=config.TEMPORAL_TAU_S, prune_weight=config.TEMPORAL_PRUNE_WEIGHT, obstacle_h=H,
    l_hit=config.LOGODDS_HIT, l_miss=config.LOGODDS_MISS, l_min=config.LOGODDS_MIN, l_max=config.LOGODDS_MAX,
    terrain_cell=config.TERRAIN_CELL_M, ground_percentile=config.GROUND_PERCENTILE,
)
NO_SEED = FusionParams(**{**PARAMS.__dict__, "seed_children": False})

# name -> (bands, params, fused?). single = fresh map per scan, same code path.
CONFIGS: dict[str, tuple[tuple[Band, ...], FusionParams, bool]] = {
    "single F-A": (config.FOVEATED_BANDS, PARAMS, False),
    "fused U-20": ((Band(0.0, R, 0.2),), PARAMS, True),
    "fused F-A no seed": (config.FOVEATED_BANDS, NO_SEED, True),
    "fused F-A seeded": (config.FOVEATED_BANDS, PARAMS, True),
}


# ring index of world xy around sensor xy
def ring_of(xy: np.ndarray, sensor: np.ndarray) -> np.ndarray:
    r = np.hypot(xy[:, 0] - sensor[0], xy[:, 1] - sensor[1])
    out = np.full(len(xy), -1)
    for k, (lo, hi) in enumerate(EVAL_RINGS):
        out[(r >= lo) & (r < hi)] = k
    return out


# per ring: reference ground cells, how many map observes, how many with own (not inherited) data, ground error
def coverage(fov: FoveatedLayers, ref: GridLayers, sensor: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    centers = cell_centers(ref.spec)
    near = ref.observed & (np.hypot(centers[..., 0] - sensor[0], centers[..., 1] - sensor[1]) < R)
    xy, g_ref = centers[near], ref.ground[near]
    g_map = fov.sample(xy, "ground")
    native = fov.sample(xy, "inherited") == 0.0
    ring = ring_of(xy, sensor)
    out = []
    for k in range(len(EVAL_RINGS)):
        s = ring == k
        hit = s & np.isfinite(g_map)
        out.append((int(s.sum()), int(hit.sum()), int((hit & native).sum()), np.abs(g_map[hit] - g_ref[hit])))
    return out


# curb edges near sensor: ring, both probes observed, step kept
def curbs(fov: FoveatedLayers, ref: GridLayers, exy: np.ndarray, en: np.ndarray, d: float, sensor: np.ndarray):
    ring = ring_of(exy, sensor)
    s = ring >= 0
    exy, en, ring = exy[s], en[s], ring[s]
    hi, lo = exy + d * en, exy - d * en
    ref_step = sample_layer(ref, hi, "ground") - sample_layer(ref, lo, "ground")
    map_step = fov.sample(hi, "ground") - fov.sample(lo, "ground")
    obs = np.isfinite(map_step)
    return ring, obs, obs & (map_step >= KEEP_RATIO * ref_step)


# ghosts: map obstacle cells current scan sees free. confirmed: map obstacle where scan sees obstacle.
def ghosts(fov: FoveatedLayers, world_scan: np.ndarray) -> tuple[int, int, int]:
    terrain = local_terrain(world_scan, "world", config.TERRAIN_CELL_M, config.GROUND_PERCENTILE)
    lo, hi = world_scan[:, :2].min(axis=0), world_scan[:, :2].max(axis=0)
    g = build_uniform_grid(world_scan, spec_from_bounds(lo[0], hi[0] + 1e-9, lo[1], hi[1] + 1e-9, 0.2, "world"),
                           "world", config.GROUND_PERCENTILE)
    xy = cell_centers(g.spec)[g.observed]
    tz = terrain.sample(xy)
    ok = np.isfinite(tz)
    xy, tz, scan_h = xy[ok], tz[ok], g.z_max[g.observed][ok] - tz[ok]
    map_obst = (fov.sample(xy, "z_max") - tz) >= H
    free_now = scan_h < H
    return int((map_obst & free_now).sum()), int((map_obst & ~free_now).sum()), int((~free_now).sum())


# true obstacles of current scan at 5 cm (height over reference terrain) -> recall + area bloat
def obstacles(fov: FoveatedLayers, ref: GridLayers, world_scan: np.ndarray) -> tuple[int, int, int]:
    lo, hi = world_scan[:, :2].min(axis=0), world_scan[:, :2].max(axis=0)
    g = build_uniform_grid(world_scan, spec_from_bounds(lo[0], hi[0] + 1e-9, lo[1], hi[1] + 1e-9, TRUTH_RES, "world"),
                           "world", config.GROUND_PERCENTILE)
    xy = cell_centers(g.spec)[g.observed]
    terr = sample_layer(ref, xy, "ground")
    ok = np.isfinite(terr)
    xy, terr, zmax = xy[ok], terr[ok], g.z_max[g.observed][ok]
    true = (zmax - terr) >= H
    flag = (fov.sample(xy, "z_max") - terr) >= H
    return int(true.sum()), int((true & flag).sum()), int(flag.sum())


# print markdown table
def table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


# phase 3 eval: stream seq slice through fused maps, score coverage, curbs, ghosts, obstacles, cost
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
    times = load_times(args.seq_dir / "times.txt")[ids]
    scans = load_filtered(paths)
    world_scans = [velo_to_world(s, p) for s, p in zip(scans, poses)]

    world = accumulate_world_points(scans, poses)
    lo, hi = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
    ref = build_uniform_grid(world, spec_from_bounds(lo[0], hi[0], lo[1], hi[1], REF_RES, "world"), "world",
                             config.GROUND_PERCENTILE)
    del world
    edges = {d: reference_edges(ref, d) for d in PROBE_D}
    eval_ids = list(range(EVAL_FROM, len(scans), EVAL_STRIDE)) + ([len(scans) - 1] if (len(scans) - 1 - EVAL_FROM) % EVAL_STRIDE else [])
    print(f"SCANS {len(scans)} ({times[-1] - times[0]:.1f} S). SCORED AT SCANS {eval_ids}.")
    print(f"FUSION: tau {PARAMS.tau_s} s, prune w < {PARAMS.prune_weight}, obstacle {H} m over terrain, "
          f"log-odds hit {PARAMS.l_hit} miss {PARAMS.l_miss} clamp [{PARAMS.l_min}, {PARAMS.l_max}].")
    print()

    res: dict[str, dict] = {}
    for name, (bands, params, fused) in CONFIGS.items():
        m = TemporalFoveatedMap(bands, params)
        ms, mem_store, mem_render, cells = [], [], [], []
        acc = {"cov": [[0, 0, 0, []] for _ in EVAL_RINGS], "curb": {d: [] for d in PROBE_D},
               "ghost": [0, 0, 0], "obst": [0, 0, 0]}
        for i in range(len(scans)):
            if not fused:
                m = TemporalFoveatedMap(bands, params)
            t0 = time.perf_counter()
            m.integrate(scans[i], poses[i], float(times[i]))
            fov = m.render()
            ms.append((time.perf_counter() - t0) * 1e3)
            mem_store.append(m.nbytes / 1e6)
            mem_render.append(fov.nbytes / 1e6)
            cells.append(m.n_cells)
            if i not in eval_ids:
                continue
            sensor = poses[i][:2, 3]
            for k, (n, h, nat, err) in enumerate(coverage(fov, ref, sensor)):
                acc["cov"][k][0] += n
                acc["cov"][k][1] += h
                acc["cov"][k][2] += nat
                acc["cov"][k][3].append(err)
            for d in PROBE_D:
                acc["curb"][d].append(curbs(fov, ref, *edges[d], d, sensor))
            for j, v in enumerate(ghosts(fov, world_scans[i])):
                acc["ghost"][j] += v
            for j, v in enumerate(obstacles(fov, ref, world_scans[i])):
                acc["obst"][j] += v
            if name == "fused F-A seeded" and i == eval_ids[-1]:
                plot_foveated(fov, args.out / f"seq00_scan{ids[i]:03d}_fused_FA.png",
                              title=f"Fused F-A  |  seq 00 scan {ids[i]}  |  {fov.nbytes / 1e6:.2f} MB rendered")
        res[name] = {"ms": np.array(ms[1:]), "store": np.array(mem_store), "render": np.array(mem_render),
                     "cells": np.array(cells), **acc}

    rows = [[n, f"{np.median(r['ms']):.1f}", f"{np.percentile(r['ms'], 95):.1f}", f"{r['store'][-1]:.2f}",
             f"{r['store'].max():.2f}", f"{r['render'][-1]:.2f}"] for n, r in res.items()]
    print("COST PER SCAN (integrate + render, preprocess excluded)")
    table(["config", "ms med", "ms p95", "store MB end", "store MB max", "rendered MB"], rows)

    rows = []
    for n, r in res.items():
        for k, (lo_, hi_) in enumerate(EVAL_RINGS):
            tot, hit, nat, errs = r["cov"][k]
            e = np.concatenate(errs)
            rows.append([n, f"{lo_:g}-{hi_:g} m", f"{hit / max(tot, 1) * 100:.0f}%", f"{nat / max(tot, 1) * 100:.0f}%",
                         f"{np.median(e):.3f}" if e.size else "-", f"{np.percentile(e, 95):.3f}" if e.size else "-"])
    print("GROUND COVERAGE + ERROR vs reference (cover = share of observed 0.1 m reference cells map also observes;")
    print("own = cover with cell's own data, not inherited placeholder)")
    table(["config", "ring", "cover", "own", "ground err med", "ground err p95"], rows)

    for d in PROBE_D:
        rows = []
        for n, r in res.items():
            ring = np.concatenate([c[0] for c in r["curb"][d]])
            obs = np.concatenate([c[1] for c in r["curb"][d]])
            kept = np.concatenate([c[2] for c in r["curb"][d]])
            for k, (lo_, hi_) in enumerate(EVAL_RINGS):
                s = ring == k
                rows.append([n, f"{lo_:g}-{hi_:g} m", f"{s.sum()}", f"{obs[s].mean() * 100:.0f}%",
                             f"{kept[s & obs].mean() * 100:.0f}%" if (s & obs).any() else "-",
                             f"{kept[s].mean() * 100:.0f}%"])
        print(f"CURBS: probes +-{d * 100:g} cm ({2 * d * 100:g} cm apart). kept all = share of ALL edges kept.")
        table(["config", "ring", "edges", "obs", "kept | obs", "kept all"], rows)

    rows = []
    for n, r in res.items():
        gc, gk, scan_obst = r["ghost"]
        t, tp, f = r["obst"]
        rows.append([n, f"{gc * 0.04:.0f}", f"{gk / max(scan_obst, 1) * 100:.0f}%", f"{tp / max(t, 1) * 100:.1f}%",
                     f"{f / max(t, 1):.2f}x"])
    print("GHOSTS + OBSTACLES (summed over scored scans). ghost m2 = map says obstacle, current scan sees free (0.2 m cells).")
    print("confirm = scan obstacle cells map also flags. recall/bloat vs current scan 5 cm obstacle truth.")
    table(["config", "ghost m2", "confirm", "obstacle recall", "area bloat"], rows)


if __name__ == "__main__":
    main()
