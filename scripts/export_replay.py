import argparse
import gzip
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.models.features import cell_features
from src.models.labeler import HazardLabeler
from src.models.targets import CLASS_NAMES, OBSTACLE, geometric_classes
from src.temporal.fusion import TemporalFoveatedMap
from src.viz.replay import (
    COLUMNS,
    FLAG_INHERITED,
    FLAG_MODEL,
    FORMAT,
    encode_band,
    export_cells,
    nan_sentinel,
)
from scripts.benchmark import ZONE_M
from scripts.eval_temporal import PARAMS
from scripts.train_hazard import SEED, TH, build_rows, fit_model, load_sequence

BANDS = config.FOVEATED_BANDS


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
def load_or_train(path: Path, train_dir: Path):
    if path.exists():
        print(f"LOAD model {path}")
        return joblib.load(path)["model"]
    t0 = time.perf_counter()
    model = fit_model(build_rows(train_dir, np.random.default_rng(SEED), bands=BANDS))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "train": str(train_dir), "bands": [b.__dict__ for b in BANDS]}, path)
    print(f"TRAINED model on seq {train_dir.name}: {time.perf_counter() - t0:.0f} s. saved {path}")
    return model


# heading of velodyne x axis in world xy, radians
def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


# one rendered map -> frame bytes + per band header + label counts. terrain, features, labels read at cell probe points.
def export_frame(fov, terrain, labeler: HazardLabeler, t: float) -> tuple[bytes, list[dict], dict]:
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
    return b"".join(blobs), heads, stats


# pass 1: pipeline alone, timed like benchmark (integrate + render, then labeler). no export work between scans.
def time_pipeline(model, scans: list[np.ndarray], poses: np.ndarray, times: np.ndarray) -> list[dict]:
    m = TemporalFoveatedMap(BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    out = []
    for scan, pose, t in zip(scans, poses, times):
        t0 = time.perf_counter()
        m.integrate(scan, pose, float(t))
        fov = m.render()
        map_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        labelled = labeler.update(fov, m.terrain, float(t))
        label_ms = (time.perf_counter() - t0) * 1e3
        out.append({"map_ms": round(map_ms, 2), "label_ms": round(label_ms, 2), "labelled": int(labelled)})
    return out


# pass 2: same deterministic pipeline again, untimed, writes replay frames + manifest
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--test", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--out", type=Path, default=Path("outputs/replay"))
    ap.add_argument("--model", type=Path, default=Path("outputs/replay/model_fa_seq00.joblib"))
    args = ap.parse_args()

    model = load_or_train(args.model, args.train)
    scans, poses, times = load_sequence(args.test)
    ids = sorted(int(p.stem) for p in (args.test / "velodyne").glob("*.bin"))
    out = args.out / f"{args.test.name}_{ids[0]:04d}-{ids[-1]:04d}"
    (out / "frames").mkdir(parents=True, exist_ok=True)

    timing = time_pipeline(model, scans, poses, times)
    m = TemporalFoveatedMap(BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    frames, raw, packed = [], 0, 0
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        fov = m.render()
        labelled = labeler.update(fov, m.terrain, float(t))
        if labelled != timing[i]["labelled"]:
            raise AssertionError(f"frame {i}: export pass labelled {labelled}, timed pass {timing[i]['labelled']}")
        blob, heads, stats = export_frame(fov, m.terrain, labeler, float(t))
        name = f"frames/{i:06d}.bin.gz"
        gz = gzip.compress(blob, compresslevel=6, mtime=0)
        (out / name).write_bytes(gz)
        raw, packed = raw + len(blob), packed + len(gz)
        frames.append({"i": i, "scan": ids[i], "t": float(t), "file": name, "bands": heads,
                       "pose": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]), yaw_of(pose)],
                       "center": list(fov.center), **timing[i], "store_mb": round(m.nbytes / 1e6, 3),
                       "render_mb": round(fov.nbytes / 1e6, 3), **stats})
        if i % 50 == 0:
            print(f"FRAME {i}: {len(gz) / 1e3:.0f} KB")

    tot = np.array([f["map_ms"] + f["label_ms"] for f in frames[1:]])
    manifest = {
        "format": FORMAT,
        "frame": {"name": "world", "definition": "KITTI velodyne frame at sequence scan 0: T_world_velo = Tr^-1 P_i Tr. "
                  "metres. x, y horizontal, z up. yaw = heading of velodyne x axis."},
        "sequence": args.test.name, "scans": [ids[0], ids[-1]], "train": f"seq {args.train.name} (model never saw this slice)",
        "bands": [b.__dict__ for b in BANDS], "zone_m": ZONE_M, "classes": CLASS_NAMES,
        "thresholds": TH.__dict__, "ego_box_velo": list(config.EGO_BOX_VELO[:4]),
        "columns": [{"name": n, "dtype": d, "scale": s, "nan": nan_sentinel(d)} for n, d, s in COLUMNS],
        "flags": {"inherited": FLAG_INHERITED, "model": FLAG_MODEL},
        "timing": {"cpu": cpu_name(), "note": "ms from a clean timed pass before export (integrate + render, then labeler "
                   "update), same split as scripts/benchmark.py. differs run to run and machine to machine.",
                   "total_ms_median": round(float(np.median(tot)), 1), "total_ms_p95": round(float(np.percentile(tot, 95)), 1),
                   "over_100ms_share": round(float(np.mean(tot > 100)), 4)},
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"WROTE {len(frames)} frames to {out}. raw {raw / 1e6:.1f} MB, gzip {packed / 1e6:.1f} MB.")
    print(f"TOTAL ms (frames 1+): median {np.median(tot):.1f}, p95 {np.percentile(tot, 95):.1f}, "
          f"over 100 ms {np.mean(tot > 100):.1%}. cpu: {cpu_name()}")


if __name__ == "__main__":
    main()
