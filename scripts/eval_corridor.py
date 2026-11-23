import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src import config
from src.mapping.foveated_grid import Band
from src.models.features import cell_features, grid_sampler
from src.models.labeler import HazardLabeler
from src.mapping.uniform_grid import GridLayers, build_uniform_grid, spec_from_bounds
from src.models.targets import geometric_classes
from src.temporal.accumulate import accumulate_world_points
from src.temporal.terrain import Terrain, local_terrain
from src.planning.corridor import (
    BLOCKED,
    CLEAR,
    NO_LABEL,
    CorridorParams,
    arc_probes_velo,
    check_corridor,
    confirm_obstacles,
    probes_to_world,
    score_arcs,
)
from src.temporal.fusion import TemporalFoveatedMap
from scripts.benchmark import ZONE_M
from scripts.eval_temporal import PARAMS
from scripts.export_replay import load_or_train
from scripts.train_hazard import REF_RES, TH, load_sequence

FROM, STRIDE = 10, 5
REF_HALF = 5  # reference per scored scan = scans i-5 .. i+5 (~1 s). whole-slice stack smears moving cars along lane
U20 = (Band(0.0, config.RANGE_MAX_M, 0.2),)
MAPS = {"foveated": config.FOVEATED_BANDS, "uniform": U20}
# config -> (map, label source, use map obstacle evidence)
CONFIGS = {
    "foveated + model + evidence": ("foveated", "model", True),
    "foveated + model, label only": ("foveated", "model", False),
    "foveated + rule + evidence": ("foveated", "rule", True),
    "uniform + model + evidence": ("uniform", "model", True),
}
MAIN = "foveated + model + evidence"
AHEAD_M = (10.0, 20.0)  # driven path point used for driver curvature: first pose this far along path


# curvature of path human driver really took next: first future pose 10-20 m along path, in current velo frame.
# k = 2 y / (x^2 + y^2) = circle through sensor tangent to x axis. None if slice ends first.
def driven_curvature(poses: np.ndarray, i: int) -> float | None:
    xy = poses[i:, :2, 3]
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    j = np.flatnonzero((along >= AHEAD_M[0]) & (along <= AHEAD_M[1]))
    if j.size == 0:
        return None
    p = np.linalg.inv(poses[i]) @ poses[i + j[0]][:, 3]
    return float(2 * p[1] / (p[0] ** 2 + p[1] ** 2))


# rule labels on any sampler at xy; unobserved spot (count 0 / below min) -> NO_LABEL or unknown
def rule_labels(sample_many, terrain_z: np.ndarray, xy: np.ndarray, sensor: np.ndarray, t: float,
                min_count: float) -> np.ndarray:
    f = cell_features(sample_many, xy, terrain_z, sensor, t)
    seen = f["count"] > 0
    return np.where(seen, geometric_classes(f, f["count"] >= min_count, TH), NO_LABEL)


# dense truth for one moment: stack of nearby scans only, 0.1 m grid around sensor + its local terrain
def window_reference(scans: list[np.ndarray], poses: np.ndarray, i: int, reach: float) -> tuple[GridLayers, Terrain]:
    a, b = max(i - REF_HALF, 0), min(i + REF_HALF + 1, len(scans))
    world = accumulate_world_points(scans[a:b], poses[a:b])
    cx, cy = poses[i][:2, 3]
    world = world[(np.abs(world[:, 0] - cx) < reach) & (np.abs(world[:, 1] - cy) < reach)]
    spec = spec_from_bounds(cx - reach, cx + reach, cy - reach, cy + reach, REF_RES, "world")
    ref = build_uniform_grid(world, spec, "world", config.GROUND_PERCENTILE)
    return ref, local_terrain(world, "world", config.TERRAIN_CELL_M, config.GROUND_PERCENTILE)


# stream slice through each map; at scored scans collect probe labels for live configs + dense reference.
# main config also timed every scan (label fetch + evidence + scoring) = corridor ms.
def collect(seq: Path, train: Path, cp: CorridorParams) -> dict:
    scans, poses, times = load_sequence(seq)
    probes_velo = arc_probes_velo(cp)
    out = {"labels": {c: [] for c in CONFIGS}, "ref": [], "driven": [], "ms": []}
    keep = [i for i in range(len(scans)) if i >= FROM and (i - FROM) % STRIDE == 0]
    for name, bands in MAPS.items():
        model = load_or_train(Path(f"outputs/replay/model_{name}_seq00_mapv{config.MAP_VERSION}.joblib"), train,
                              bands=bands)
        m = TemporalFoveatedMap(bands, PARAMS)
        labeler = HazardLabeler(model, ZONE_M)
        for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
            m.integrate(scan, pose, float(t))
            fov = m.render()
            labeler.update(fov, m.terrain, float(t))
            occupied = lambda xy: np.nan_to_num(fov.sample(xy, "logodds")) > 0
            if name == "foveated":
                t0 = time.perf_counter()
                check_corridor(lambda xy: labeler.labels_at(fov, xy), pose, cp, None, occupied)
                out["ms"].append((time.perf_counter() - t0) * 1e3)
            if i not in keep:
                continue
            w = probes_to_world(probes_velo, pose)
            flat, shape = w.reshape(-1, 2), w.shape[:3]
            sensor = pose[:2, 3]
            model_lab = labeler.labels_at(fov, flat)
            rule_lab = rule_labels(fov.sample_many, m.terrain.sample(flat), flat, sensor, float(t), 1)
            occ = occupied(flat)
            for c, (mk, src, ev) in CONFIGS.items():
                if mk != name:
                    continue
                lab = model_lab if src == "model" else rule_lab
                out["labels"][c].append((confirm_obstacles(lab, occ) if ev else lab).reshape(shape))
            if name == "foveated":
                ref, ref_terrain = window_reference(scans, poses, i, cp.length_m + 5.0)
                out["ref"].append(rule_labels(grid_sampler(ref), ref_terrain.sample(flat), flat, sensor, float(t),
                                              config.REF_MIN_POINTS).reshape(shape))
                out["driven"].append(driven_curvature(poses, i))
    return out


# per config: verdict agreement + error rates vs reference verdicts, for one parameter set
def score(data: dict, cp: CorridorParams) -> dict[str, dict[str, float]]:
    ref = [score_arcs(r, cp) for r in data["ref"]]
    k = np.array(cp.curvatures)
    ref_drv = [(rr, int(np.argmin(np.abs(k - d)))) for rr, d in zip(ref, data["driven"]) if d is not None and abs(d) <= k.max()]
    res = {}
    for c, labs in data["labels"].items():
        live = [score_arcs(lab, cp) for lab in labs]
        lv = np.concatenate([r.verdict for r in live])
        rv = np.concatenate([r.verdict for r in ref])
        rb = rv == BLOCKED
        picks = [(lr.pick, rr) for lr, rr in zip(live, ref) if lr.pick >= 0]
        drv = [(lr, int(np.argmin(np.abs(k - d)))) for lr, d in zip(live, data["driven"])
               if d is not None and abs(d) <= k.max()]
        res[c] = {
            "agree": float(np.mean(lv == rv)),
            "blocked_agree": float(np.mean((lv == BLOCKED) == rb)),
            "missed_block": float(np.mean(lv[rb] != BLOCKED)) if rb.any() else float("nan"),
            "false_clear": float(np.mean(lv[rb] == CLEAR)) if rb.any() else float("nan"),
            "false_block": float(np.mean(rv[lv == BLOCKED] != BLOCKED)) if (lv == BLOCKED).any() else float("nan"),
            "pick_ref_blocked": float(np.mean([rr.verdict[p] == BLOCKED for p, rr in picks])) if picks else float("nan"),
            "no_pick": float(np.mean([r.pick < 0 for r in live])),
            "driven_blocked": float(np.mean([lr.verdict[d] == BLOCKED for lr, d in drv])) if drv else float("nan"),
            "ref_blocked_share": float(np.mean(rb)),
            "ref_driven_blocked": float(np.mean([rr.verdict[d] == BLOCKED for rr, d in ref_drv])) if ref_drv else float("nan"),
        }
    return res


# table of scores per config
def table(title: str, res: dict[str, dict[str, float]]) -> None:
    print(title)
    print("| config | verdict agree | blocked yes/no agree | ref-blocked arcs live missed | of which live CLEAR "
          "(false clear) | live-blocked arcs ref not blocked | picked arc ref-blocked | no candidate | driven arc blocked |")
    print("|---" * 9 + "|")
    for c, r in res.items():
        print(f"| {c} | {r['agree']:.1%} | {r['blocked_agree']:.1%} | {r['missed_block']:.1%} | {r['false_clear']:.1%} | "
              f"{r['false_block']:.1%} | {r['pick_ref_blocked']:.1%} | {r['no_pick']:.1%} | {r['driven_blocked']:.1%} |")
    print()


# step 5: candidate corridor check vs dense reference map. --tune sweeps min_hits (train seq only).
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--tune", action="store_true", help="sweep min_hits. only on training sequence")
    args = ap.parse_args()
    if args.tune and args.seq.resolve() != args.train.resolve():
        raise SystemExit("--tune only on the training sequence. test sequence is for final numbers.")
    cp = CorridorParams()
    data = collect(args.seq, args.train, cp)
    n = len(data["ref"])
    ms = np.array(data["ms"][1:])
    print(f"CORRIDOR vs REFERENCE seq {args.seq.name}: {n} scans scored (every {STRIDE} from {FROM}), "
          f"{len(cp.curvatures)} arcs each. map v{config.MAP_VERSION}. candidate check only, nothing steered.")
    print(f"reference = dense 0.1 m stack of scans i-{REF_HALF}..i+{REF_HALF} around each scored scan + same thresholds "
          "(holds ~0.5 s of future scans; truth is geometric, not human labels).")
    print(f"corridor ms ({MAIN}, all scans): median {np.median(ms):.2f}, p95 {np.percentile(ms, 95):.2f}")
    print()
    for hits in ((1, 2, 4, 6, 8, 12) if args.tune else (cp.min_hits,)):
        p = replace(cp, min_hits=hits)
        res = score(data, p)
        table(f"min_hits {hits} (window {p.window_m:g} m, stop {p.stop_m:g} m). reference blocked share "
              f"{res[MAIN]['ref_blocked_share']:.1%} of arcs; reference blocks driven arc in "
              f"{res[MAIN]['ref_driven_blocked']:.1%} of scans (truth sanity, want ~0).", res)


if __name__ == "__main__":
    main()
