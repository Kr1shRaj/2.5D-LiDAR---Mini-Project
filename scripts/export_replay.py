import argparse
import gzip
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.models.features import cell_features, grid_sampler
from src.models.labeler import HazardLabeler
from src.models.targets import CLASS_NAMES, OBSTACLE, geometric_classes
from src.planning.corridor import (
    BLOCKED,
    VERDICTS,
    CorridorParams,
    CorridorResult,
    arc_probes_velo,
    check_corridor,
    probes_to_world,
    score_arcs,
)
from src.preprocess.transforms import velo_to_world
from src.temporal.fusion import TemporalFoveatedMap
from src.viz.replay import (
    COLUMNS,
    FLAG_INHERITED,
    FLAG_MODEL,
    FORMAT,
    POINT_STRIDE,
    POINTS_FORMAT,
    encode_band,
    encode_points,
    export_cells,
    nan_sentinel,
)
from scripts.benchmark import ZONE_M
from scripts.eval_temporal import PARAMS
from scripts.train_hazard import SEED, TH, build_rows, fit_model, load_sequence

BANDS = config.FOVEATED_BANDS
CP = CorridorParams()
STRAIGHT = len(CP.curvatures) // 2


# cpu name for timing note
def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# same F-A model as phase 5 benchmark (seed 0, train slice only). cached after first fit.
def load_or_train(path: Path, train_dir: Path, bands: tuple = BANDS):
    if path.exists():
        print(f"LOAD model {path}")
        return joblib.load(path)["model"]
    t0 = time.perf_counter()
    model = fit_model(build_rows(train_dir, np.random.default_rng(SEED), bands=bands))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "train": str(train_dir), "bands": [b.__dict__ for b in bands],
                 "map_version": config.MAP_VERSION}, path)
    print(f"TRAINED model on seq {train_dir.name}: {time.perf_counter() - t0:.0f} s. saved {path}")
    return model


# heading of velodyne x axis in world xy, radians
def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


# one rendered map -> frame bytes + per band header + label counts + occupied cell probe xy (log-odds > 0).
# terrain, features, labels read at cell probe points.
def export_frame(fov, terrain, labeler: HazardLabeler, t: float) -> tuple[bytes, list[dict], dict, np.ndarray]:
    cells = [export_cells(bl, fov.center) for bl in fov.bands]
    sels = [sel for sel, _ in cells]
    xy = np.vstack([probe for _, probe in cells])
    tz = terrain.sample(xy)
    f = cell_features(fov.sample_many, xy, tz, np.array(fov.center), t)
    rule = geometric_classes(f, f["count"] > 0, TH)
    model = labeler.labels_at(fov, xy)
    label = np.where(model >= 0, model, rule)
    blobs, heads, at = [], [], 0
    for bl, sel in zip(fov.bands, sels):
        n = sel.size
        part = slice(at, at + n)
        at += n
        flags = np.where(bl.layer("inherited")[sel] > 0, FLAG_INHERITED, 0) | np.where(model[part] >= 0, FLAG_MODEL, 0)
        cols = {"z_max": bl.z_max[sel], "ground": bl.ground[sel], "terrain": tz[part],
                "rough": np.sqrt(bl.z_var[sel]), "weight": bl.layer("weight")[sel], "n_obs": bl.count[sel],
                "logodds": bl.layer("logodds")[sel], "label": label[part], "flags": flags}
        blobs.append(encode_band(bl.spec.nx, bl.spec.ny, bl.flat_ids[sel], cols))
        heads.append({"x_min": bl.spec.x_min, "y_min": bl.spec.y_min, "nx": bl.spec.nx, "ny": bl.spec.ny, "n_obs": int(n)})
    occ = np.concatenate([bl.layer("logodds")[s] > 0 for bl, s in zip(fov.bands, sels)])
    stats = {"classes": np.bincount(label, minlength=len(CLASS_NAMES)).tolist(),
             "model_labels": int(np.sum(model >= 0)), "occupied": int(occ.sum()),
             "obstacle": int(np.sum(label == OBSTACLE))}
    return b"".join(blobs), heads, stats, xy[occ]


# current scan, every POINT_STRIDE-th filtered point, world frame + height over this scan's local terrain
def export_points(scan: np.ndarray, pose: np.ndarray, terrain) -> bytes:
    pts = scan[::POINT_STRIDE, :3]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < config.RANGE_MAX_M]
    world = velo_to_world(pts, pose)
    return encode_points(world, pose[:2, 3], world[:, 2] - terrain.sample(world[:, :2]))


GHOST_MIN_AGE = 3  # cell must have held obstacle evidence this many scans in a row: a trail, not 1-scan flicker


# occupied cell spots keyed on 1 cm grid -> (scans in a row with log-odds > 0, height over terrain, cell size, own scan)
# own scan = not an inherited placeholder copied from coarse ring
def occupancy_age(occ_xy: np.ndarray, fov, terrain, prev_age: dict) -> dict:
    s = fov.sample_many(occ_xy, ["z_max", "resolution", "inherited"])
    h = s["z_max"] - terrain.sample(occ_xy)
    keys = map(tuple, np.round(occ_xy, 2).tolist())  # 1 cm: cell centres sit on 5 cm offsets, 0.1 m would land on edges
    return {k: (prev_age.get(k, (0,))[0] + 1, hk, rk, lk)
            for k, hk, rk, lk in zip(keys, h.tolist(), s["resolution"].tolist(), (s["inherited"] == 0).tolist())}


# ghost cleared this scan: spot held own-scan obstacle evidence >= GHOST_MIN_AGE scans with obstacle height (z max over
# terrain >= TH.obstacle_h); now same-size cell (not a ring hand-off) was seen free (log-odds <= 0) and its z max
# fell below obstacle height = clearing dropped an old z max. sensor-relative, 0.1 m. unobserved now = unseen, not cleared.
def cleared_cells(fov, terrain, prev_age: dict, sensor_xy: np.ndarray) -> list[list[float]]:
    old = [(k, r) for k, (a, h, r, own) in prev_age.items() if a >= GHOST_MIN_AGE and h >= TH.obstacle_h and own]
    if not old:
        return []
    xy, res = np.array([k for k, _ in old]), np.array([r for _, r in old])
    s = fov.sample_many(xy, ["logodds", "z_max", "resolution", "inherited"])
    h = s["z_max"] - terrain.sample(xy)
    gone = (np.isfinite(s["logodds"]) & (s["logodds"] <= 0) & np.isfinite(h) & (h < TH.obstacle_h)
            & (s["resolution"] == res) & (s["inherited"] == 0))
    return np.round(xy[gone] - sensor_xy, 1).tolist()


# reference verdicts for one scan: rule labels on short-window dense truth (same as eval_corridor), no evidence rule
def reference_verdicts(scans: list[np.ndarray], poses: np.ndarray, i: int, t: float) -> list[int]:
    from scripts.eval_corridor import rule_labels, window_reference

    w = probes_to_world(arc_probes_velo(CP), poses[i])
    flat = w.reshape(-1, 2)
    ref, ref_terrain = window_reference(scans, poses, i, CP.length_m + 5.0)
    lab = rule_labels(grid_sampler(ref), ref_terrain.sample(flat), flat, poses[i][:2, 3], t, config.REF_MIN_POINTS)
    return score_arcs(lab.reshape(w.shape[:3]), CP).verdict.tolist()


# candidate corridor check on rendered map: model labels, obstacle only with map evidence (log-odds > 0)
def corridor(labeler: HazardLabeler, fov, pose: np.ndarray, prev: float | None) -> CorridorResult:
    return check_corridor(lambda xy: labeler.labels_at(fov, xy), pose, CP, prev,
                          lambda xy: np.nan_to_num(fov.sample(xy, "logodds")) > 0)


# pass 1: pipeline alone, timed like benchmark (integrate + render, then labeler). no export work between scans.
def time_pipeline(model, scans: list[np.ndarray], poses: np.ndarray, times: np.ndarray) -> list[dict]:
    m = TemporalFoveatedMap(BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    out, prev = [], None
    for scan, pose, t in zip(scans, poses, times):
        t0 = time.perf_counter()
        m.integrate(scan, pose, float(t))
        fov = m.render()
        map_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        labelled = labeler.update(fov, m.terrain, float(t))
        label_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        r = corridor(labeler, fov, pose, prev)
        corridor_ms = (time.perf_counter() - t0) * 1e3
        prev = r.curvatures[r.pick] if r.pick >= 0 else None
        out.append({"map_ms": round(map_ms, 2), "label_ms": round(label_ms, 2), "corridor_ms": round(corridor_ms, 2),
                    "labelled": int(labelled)})
    return out


# scene list for viewer menu: <out>/index.json, one entry per replay folder (real or sim). re-written, sorted by folder
def update_index(root: Path, folder: str, scene: str, synthetic: bool) -> None:
    path = root / "index.json"
    items = json.loads(path.read_text())["scenes"] if path.exists() else []
    items = [s for s in items if s["folder"] != folder and (root / s["folder"] / "manifest.json").exists()]
    items.append({"folder": folder, "scene": scene, "synthetic": synthetic})
    path.write_text(json.dumps({"scenes": sorted(items, key=lambda s: (s["synthetic"], s["folder"]))}, indent=1))


# pass 2: same deterministic pipeline again, untimed, writes replay frames + manifest
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--test", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--out", type=Path, default=Path("outputs/replay"))
    ap.add_argument("--model", type=Path, default=Path(f"outputs/replay/model_foveated_seq00_mapv{config.MAP_VERSION}.joblib"))
    ap.add_argument("--scene", default="", help="label shown in viewer scene menu, e.g. 'seq 08 turns'")
    ap.add_argument("--no-reference", action="store_true", help="skip per-scan reference verdicts (faster)")
    args = ap.parse_args()
    from scripts.eval_corridor import driven_curvature

    model = load_or_train(args.model, args.train)
    scans, poses, times = load_sequence(args.test)
    ids = sorted(int(p.stem) for p in (args.test / "velodyne").glob("*.bin"))
    out = args.out / f"{args.test.name}_{ids[0]:04d}-{ids[-1]:04d}"
    (out / "frames").mkdir(parents=True, exist_ok=True)

    timing = time_pipeline(model, scans, poses, times)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    frames, raw, packed, prev, events = [], 0, 0, None, []
    occ_age, pts_bytes = {}, 0
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        fov = m.render()
        labelled = labeler.update(fov, m.terrain, float(t))
        if labelled != timing[i]["labelled"]:
            raise AssertionError(f"frame {i}: export pass labelled {labelled}, timed pass {timing[i]['labelled']}")
        r = corridor(labeler, fov, pose, prev)
        prev = r.curvatures[r.pick] if r.pick >= 0 else None
        if r.verdict[STRAIGHT] == BLOCKED and r.pick >= 0 and r.pick != STRAIGHT:
            events.append(i)
        drv = driven_curvature(poses, i)
        cor = {"v": r.verdict.tolist(), "b": r.block_m.round(2).tolist(), "h": r.hazard.round(3).tolist(),
               "u": r.unknown.round(3).tolist(), "pick": r.pick,
               "drv": None if drv is None else round(drv, 4),
               "ref": None if args.no_reference else reference_verdicts(scans, poses, i, float(t))}
        blob, heads, stats, occ_xy = export_frame(fov, m.terrain, labeler, float(t))
        cleared = cleared_cells(fov, m.terrain, occ_age, pose[:2, 3])
        occ_age = occupancy_age(occ_xy, fov, m.terrain, occ_age)
        pts = gzip.compress(export_points(scan, pose, m.terrain), compresslevel=6, mtime=0)
        (out / f"frames/{i:06d}.pts.gz").write_bytes(pts)
        pts_bytes += len(pts)
        n_refined = sum(h["n_obs"] for h, bl in zip(heads, fov.bands) if bl.overlay)
        # band list incl. refined-patch overlay (same every frame): rings near to far, then overlays
        layer_bands = [{**bl.band.__dict__, "overlay": bl.overlay} for bl in fov.bands]
        name = f"frames/{i:06d}.bin.gz"
        gz = gzip.compress(blob, compresslevel=6, mtime=0)
        (out / name).write_bytes(gz)
        raw, packed = raw + len(blob), packed + len(gz)
        frames.append({"i": i, "scan": ids[i], "t": float(t), "file": name, "bands": heads,
                       "pose": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]), yaw_of(pose)],
                       "center": list(fov.center), **timing[i], "store_mb": round(m.nbytes / 1e6, 3),
                       "render_mb": round(fov.nbytes / 1e6, 3), "corridor": cor, "cleared": cleared,
                       "refined": int(n_refined), "points": f"frames/{i:06d}.pts.gz", **stats})
        if i % 50 == 0:
            print(f"FRAME {i}: {len(gz) / 1e3:.0f} KB")

    tot = np.array([f["map_ms"] + f["label_ms"] + f["corridor_ms"] for f in frames[1:]])
    manifest = {
        "format": FORMAT,
        "frame": {"name": "world", "definition": "KITTI velodyne frame at sequence scan 0: T_world_velo = Tr^-1 P_i Tr. "
                  "metres. x, y horizontal, z up. yaw = heading of velodyne x axis."},
        "scene": args.scene or f"seq {args.test.name} {ids[0]}-{ids[-1]}", "synthetic": False,
        "ghosts": {"min_age": GHOST_MIN_AGE, "note": "spots with obstacle evidence and obstacle height >= min_age scans "
                   "in a row (own scans, not hand-off placeholders), now seen free (log-odds <= 0) with z max below obstacle height: ghost clearing dropped "
                   "the old z max. viewer flashes them."},
        "points": {"format": POINTS_FORMAT, "stride": POINT_STRIDE,
                   "note": "current scan only, ego box removed, every n-th point, within map range"},
        "sequence": args.test.name, "scans": [ids[0], ids[-1]], "train": f"seq {args.train.name} (model never saw this slice)",
        "bands": layer_bands, "zone_m": ZONE_M, "classes": CLASS_NAMES,
        "thresholds": TH.__dict__, "ego_box_velo": list(config.EGO_BOX_VELO[:4]),
        "columns": [{"name": n, "dtype": d, "scale": s, "nan": nan_sentinel(d)} for n, d, s in COLUMNS],
        "flags": {"inherited": FLAG_INHERITED, "model": FLAG_MODEL},
        "timing": {"cpu": cpu_name(), "note": "ms from a clean timed pass before export (integrate + render, labeler update, "
                   "corridor check). differs run to run and machine to machine.",
                   "total_ms_median": round(float(np.median(tot)), 1), "total_ms_p95": round(float(np.percentile(tot, 95)), 1),
                   "over_100ms_share": round(float(np.mean(tot > 100)), 4)},
        "map_version": config.MAP_VERSION,
        "corridor": {**{k: list(v) if isinstance(v, tuple) else v for k, v in CP.__dict__.items()},
                     "verdicts": VERDICTS, "straight": STRAIGHT, "events": events,
                     "reference": None if args.no_reference else "rule labels on dense stack of scans i-5..i+5 "
                     "(0.1 m), same as scripts/eval_corridor.py. truth side only.",
                     "driven": "curvature to first pose 10-20 m along the path the KITTI driver really took",
                     "note": "candidate corridor check only: the KITTI car really drove; nothing was steered."},
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    update_index(args.out, out.name, manifest["scene"], False)
    print(f"CORRIDOR events (straight blocked, other arc candidate): {len(events)} frames")
    print(f"WROTE {len(frames)} frames to {out}. raw {raw / 1e6:.1f} MB, gzip {packed / 1e6:.1f} MB. "
          f"points gzip {pts_bytes / 1e6:.1f} MB.")
    print(f"TOTAL ms (frames 1+): median {np.median(tot):.1f}, p95 {np.percentile(tot, 95):.1f}, "
          f"over 100 ms {np.mean(tot > 100):.1%}. cpu: {cpu_name()}")


if __name__ == "__main__":
    main()
