import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.models.labeler import HazardLabeler
from src.models.targets import CLASS_NAMES
from src.planning.corridor import BLOCKED, VERDICTS
from src.preprocess.filters import preprocess_mask
from src.sim.drive import (
    CarState,
    DriveParams,
    choose_arc,
    contacts,
    off_road,
    route_curvature,
    step,
    target_speed,
)
from src.sim.lidar import LidarModel, pose_matrix, ray_dirs, scan
from src.sim.world import KERB_H_M, ROAD_HALF_M, demo_world
from src.temporal.fusion import TemporalFoveatedMap
from src.viz.replay import COLUMNS, FLAG_INHERITED, FLAG_MODEL, FORMAT, POINT_STRIDE, POINTS_FORMAT, nan_sentinel
from scripts.benchmark import ZONE_M
from scripts.eval_corridor import driven_curvature
from scripts.eval_temporal import PARAMS
from scripts.export_replay import (
    BANDS,
    CP,
    GHOST_MIN_AGE,
    STRAIGHT,
    cleared_cells,
    corridor,
    cpu_name,
    export_frame,
    export_points,
    load_or_train,
    occupancy_age,
    reference_verdicts,
    update_index,
)
from scripts.train_hazard import TH

# SIMULATION DEMO (NON-BENCHMARK). synthetic street, raycast lidar, virtual car follows corridor-safe arc nearest route.
# map, labeler (KITTI seq 00 model) and corridor check are the real, unchanged code. numbers here are never results.

DT = 0.1  # 10 Hz scans, as KITTI
SEED = 0
BANNER = "SIMULATION DEMO (NON-BENCHMARK)"


# closed loop: scan -> map -> labels -> corridor -> arc choice -> move. returns per-scan logs + stored scans
def drive(model, steps: int, end_margin_m: float = 25.0) -> dict:
    world = demo_world(SEED)
    lm, dp = LidarModel(), DriveParams()
    dirs = ray_dirs(lm)
    rng = np.random.default_rng(SEED)
    p0, h0 = world.at_station(5.0)
    car = CarState(float(p0[0] + 1.75 * np.sin(h0)), float(p0[1] - 1.75 * np.cos(h0)), h0, 0.0)  # ego lane, standing
    m = TemporalFoveatedMap(BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    log = {"scans": [], "poses": [], "times": [], "car": [], "result": [], "choice": [], "k_route": [], "ms": [],
           "contact": [], "off_road": [], "movers": [], "rays_ms": [], "mode": []}
    prev = None
    for i in range(steps):
        t = i * DT
        s_now = world.nearest_station(np.array([car.x, car.y]))
        if s_now > world.station[-1] - end_margin_m:
            break
        for mv in world.movers:
            if not np.isfinite(mv.t0) and s_now >= mv.trigger_station:
                mv.t0 = t
        boxes = world.boxes(t)
        pose = pose_matrix(car.x, car.y, car.yaw, lm)
        t0 = time.perf_counter()
        pts = scan(pose, boxes, lm, rng, dirs)
        rays_ms = (time.perf_counter() - t0) * 1e3
        pts = pts[preprocess_mask(pts, config.RANGE_MIN_M, config.RANGE_MAX_M, config.EGO_BOX_VELO)]
        # timed like the real export: integrate + render, labeler, corridor. raycast not counted (sim only)
        t0 = time.perf_counter()
        m.integrate(pts, pose, t)
        fov = m.render()
        map_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        labelled = labeler.update(fov, m.terrain, t)
        label_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        r = corridor(labeler, fov, pose, prev)
        corridor_ms = (time.perf_counter() - t0) * 1e3
        prev = r.curvatures[r.pick] if r.pick >= 0 else None
        k_route = route_curvature(world, car, dp)
        choice, mode = choose_arc(r, k_route, CP.length_m, dp)
        k_cmd = float(r.curvatures[choice]) if choice >= 0 else car.k
        v_cmd = target_speed(r, choice, mode, STRAIGHT, dp)
        yield_ = {"i": i, "t": t, "pose": pose, "fov": fov, "map": m, "labeler": labeler, "result": r, "scan": pts}
        log["scans"].append(pts.astype(np.float32))
        log["poses"].append(pose)
        log["times"].append(t)
        log["car"].append([car.x, car.y, car.yaw, car.v, car.k])
        log["result"].append(r)
        log["choice"].append(choice)
        log["mode"].append(mode)
        log["k_route"].append(k_route)
        log["ms"].append({"map_ms": round(map_ms, 2), "label_ms": round(label_ms, 2),
                          "corridor_ms": round(corridor_ms, 2), "labelled": int(labelled)})
        log["rays_ms"].append(rays_ms)
        log["contact"].append(contacts(car, config.EGO_BOX_VELO, boxes))
        log["off_road"].append(bool(off_road(world, car, (config.EGO_BOX_VELO[3] - config.EGO_BOX_VELO[2]) / 2)))
        log["movers"].append([[round(b.cx, 2), round(b.cy, 2), round(b.yaw, 3), b.length, b.width, b.z1, b.kind]
                              for b in (mv.at(t) for mv in world.movers)])
        yield yield_
        car = step(car, k_cmd, v_cmd, DT, dp)
    log["world"] = world
    yield {"done": log}


# run loop once, export replay frames as it goes (same format as scripts/export_replay.py), manifest at end
def main() -> None:
    ap = argparse.ArgumentParser(description=f"{BANNER}: virtual car on synthetic street. never a benchmark.")
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--model", type=Path, default=Path(f"outputs/replay/model_foveated_seq00_mapv{config.MAP_VERSION}.joblib"))
    ap.add_argument("--out", type=Path, default=Path("outputs/replay"))
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()

    model = load_or_train(args.model, args.train)
    out = args.out / "sim_demo_street"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    print(f"{BANNER}. synthetic scene, closed loop. writing {out}")
    frames, occ_age, raw, packed, pts_bytes, log = [], {}, 0, 0, 0, None
    for ev in drive(model, args.steps):
        if "done" in ev:
            log = ev["done"]
            break
        i, fov, m, pose = ev["i"], ev["fov"], ev["map"], ev["pose"]
        blob, heads, stats, occ_xy = export_frame(fov, m.terrain, ev["labeler"], ev["t"])
        cleared = cleared_cells(fov, m.terrain, occ_age, pose[:2, 3])
        occ_age = occupancy_age(occ_xy, fov, m.terrain, occ_age)
        gz = gzip.compress(blob, compresslevel=6, mtime=0)
        (out / f"frames/{i:06d}.bin.gz").write_bytes(gz)
        pts = gzip.compress(export_points(ev["scan"], pose, m.terrain), compresslevel=6, mtime=0)
        (out / f"frames/{i:06d}.pts.gz").write_bytes(pts)
        raw, packed, pts_bytes = raw + len(blob), packed + len(gz), pts_bytes + len(pts)
        r = ev["result"]
        frames.append({"i": i, "scan": i, "t": ev["t"], "file": f"frames/{i:06d}.bin.gz", "bands": heads,
                       "pose": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]), float(np.arctan2(pose[1, 0], pose[0, 0]))],
                       "center": list(fov.center), "store_mb": round(m.nbytes / 1e6, 3),
                       "render_mb": round(fov.nbytes / 1e6, 3), "cleared": cleared,
                       "refined": int(sum(h["n_obs"] for h, bl in zip(heads, fov.bands) if bl.overlay)),
                       "points": f"frames/{i:06d}.pts.gz", **stats,
                       "corridor": {"v": r.verdict.tolist(), "b": r.block_m.round(2).tolist(),
                                    "h": r.hazard.round(3).tolist(), "u": r.unknown.round(3).tolist(), "pick": r.pick}})
        if i % 50 == 0:
            print(f"SIM FRAME {i}: {len(gz) / 1e3:.0f} KB")

    poses = np.array(log["poses"])
    events = []
    for i, f in enumerate(frames):
        f.update(log["ms"][i])
        drv = driven_curvature(poses, i)
        f["corridor"]["drv"] = None if drv is None else round(drv, 4)
        f["corridor"]["ref"] = None if args.no_reference else reference_verdicts(log["scans"], poses, i, f["t"])
        x, y, yaw, v, k = log["car"][i]
        f["sim"] = {"choice": log["choice"][i], "mode": log["mode"][i], "k_route": round(log["k_route"][i], 4), "speed": round(v, 2),
                    "contact": log["contact"][i], "off_road": log["off_road"][i], "movers": log["movers"][i]}
        cor = f["corridor"]
        if cor["v"][STRAIGHT] == BLOCKED and cor["pick"] >= 0 and cor["pick"] != STRAIGHT:
            events.append(i)

    world = log["world"]
    tot = np.array([f["map_ms"] + f["label_ms"] + f["corridor_ms"] for f in frames[1:]])
    contact_scans = sum(1 for c in log["contact"] if c)
    off_scans = sum(log["off_road"])
    manifest = {
        "format": FORMAT, "synthetic": True, "banner": BANNER,
        "scene": "SIM · demo street, 3 turns, obstacles in lane",
        "frame": {"name": "world", "definition": "SYNTHETIC sim world: flat ground z = 0, metres, z up. "
                  "virtual velodyne mounted 1.73 m over ground, so road sits at velo z -1.73 as on KITTI."},
        "sequence": "sim", "scans": [0, len(frames) - 1], "train": "seq 00 (KITTI). model never sees sim data",
        "bands": [{**bl.band.__dict__, "overlay": bl.overlay} for bl in fov.bands], "zone_m": ZONE_M,
        "classes": CLASS_NAMES, "thresholds": TH.__dict__, "ego_box_velo": list(config.EGO_BOX_VELO[:4]),
        "columns": [{"name": n, "dtype": d, "scale": s, "nan": nan_sentinel(d)} for n, d, s in COLUMNS],
        "flags": {"inherited": FLAG_INHERITED, "model": FLAG_MODEL},
        "ghosts": {"min_age": GHOST_MIN_AGE, "note": "same rule as scripts/export_replay.py"},
        "points": {"format": POINTS_FORMAT, "stride": POINT_STRIDE, "note": "SYNTHETIC raycast scan, every n-th point"},
        "timing": {"cpu": cpu_name(), "note": "SIMULATION DEMO, not a benchmark. ms timed inside the sim loop "
                   "(integrate + render, labeler, corridor); raycast not counted.",
                   "total_ms_median": round(float(np.median(tot)), 1), "total_ms_p95": round(float(np.percentile(tot, 95)), 1),
                   "over_100ms_share": round(float(np.mean(tot > 100)), 4)},
        "map_version": config.MAP_VERSION,
        "corridor": {**{k: list(v) if isinstance(v, tuple) else v for k, v in CP.__dict__.items()},
                     "verdicts": VERDICTS, "straight": STRAIGHT, "events": events,
                     "reference": None if args.no_reference else "rule labels on dense stack of SIM scans i-5..i+5",
                     "driven": "curvature to first pose 10-20 m along the path the VIRTUAL car drove",
                     "note": "SIMULATION DEMO: virtual car drives the non-blocked arc nearest the route curvature."},
        "sim": {
            "seed": SEED, "dt_s": DT, "road_half_m": ROAD_HALF_M, "kerb_h_m": KERB_H_M, "lane_d_m": -1.75,
            "route": np.round(world.route[::4], 2).tolist(),
            "boxes": [[round(b.cx, 2), round(b.cy, 2), round(b.yaw, 3), b.length, b.width, b.z1, b.kind]
                      for b in world.static if b.kind != "kerb"],
            "choice_note": "sim arc = corridor-pickable arc nearest route pure-pursuit curvature (route intent is sim "
                           "only; corridor check never sees the route). cyan = corridor pick, yellow = sim choice. "
                           "mode creep = no pickable arc only because map has not seen enough: 1.5 m/s so it fills. "
                           "mode stop = every arc blocked.",
            "modes": {mo: sum(1 for x in log["mode"] if x == mo) for mo in ("drive", "creep", "stop")},
            "contact_scans": contact_scans, "off_road_scans": off_scans,
            "raycast_ms_median": round(float(np.median(log["rays_ms"])), 1),
        },
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    update_index(args.out, out.name, manifest["scene"], True)
    print(f"{BANNER}: {len(frames)} scans, route end reached: {len(frames) < args.steps}. "
          f"contact scans {contact_scans}, off-road scans {off_scans}, corridor events {len(events)}.")
    print(f"WROTE {out}. gzip {packed / 1e6:.1f} MB + points {pts_bytes / 1e6:.1f} MB. "
          f"pipeline ms median {np.median(tot):.1f}, p95 {np.percentile(tot, 95):.1f} (sim, not a result).")


if __name__ == "__main__":
    main()
