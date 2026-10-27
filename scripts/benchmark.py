import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src import config
from src.mapping.foveated_grid import Band, FoveatedLayers
from src.models.features import cell_features, feature_matrix
from src.models.targets import CLASS_NAMES, OBSTACLE, TRAVERSABLE, geometric_classes
from src.preprocess.transforms import velo_to_world
from src.temporal.fusion import TemporalFoveatedMap
from src.temporal.terrain import local_terrain
from scripts.eval_temporal import EVAL_RINGS, PARAMS, coverage, curbs, ghosts, obstacles
from scripts.eval_preservation import reference_edges
from scripts.train_hazard import (
    LABELS,
    SEED,
    TH,
    build_reference,
    build_rows,
    fit_model,
    load_sequence,
    predict,
    sample_spots,
)

R = config.RANGE_MAX_M
SCORE_FROM = 10
SCORE_STRIDE = 10
CURB_D = 0.1  # sharp curb test: probes 20 cm apart

MAPS: dict[str, tuple[Band, ...]] = {
    "uniform U-20": (Band(0.0, R, 0.2),),
    "foveated F-A": config.FOVEATED_BANDS,
}


# label observed rendered cells: features at cell centres + model. fresh_only = just cells this scan touched.
def label_map(fov: FoveatedLayers, model, scan_world: np.ndarray, sensor: np.ndarray, t: float,
              fresh_only: bool) -> float:
    t0 = time.perf_counter()
    centers = []
    for bl in fov.bands:
        sel = bl.observed & (bl.layer("last_seen") == t) if fresh_only else bl.observed
        ix, iy = np.divmod(bl.flat_ids[sel], bl.spec.ny)
        centers.append(np.column_stack([bl.spec.x_min + (ix + 0.5) * bl.spec.resolution,
                                        bl.spec.y_min + (iy + 0.5) * bl.spec.resolution]))
    xy = np.vstack(centers)
    terrain = local_terrain(scan_world, "world", config.TERRAIN_CELL_M, config.GROUND_PERCENTILE)
    f = cell_features(fov.sample_many, xy, terrain.sample(xy), sensor, t)
    model.predict(feature_matrix(f))
    return (time.perf_counter() - t0) * 1e3


# stream test slice through one map; collect cost, height, curb, obstacle, ghost, hazard rows
def run_map(bands: tuple[Band, ...], test: tuple, ref, ref_terrain, edges, model) -> dict:
    scans, poses, times = test
    m = TemporalFoveatedMap(bands, PARAMS)
    out = {"ms": [], "store": [], "render": [], "label_ms": [], "fresh_ms": [], "cov": [[0, 0, []] for _ in EVAL_RINGS],
           "curb": [], "ghost": [0, 0, 0], "obst": [0, 0, 0], "rows": []}
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        t0 = time.perf_counter()
        m.integrate(scan, pose, float(t))
        fov = m.render()
        out["ms"].append((time.perf_counter() - t0) * 1e3)
        out["store"].append(m.nbytes / 1e6)
        out["render"].append(fov.nbytes / 1e6)
        if i < SCORE_FROM or (i - SCORE_FROM) % SCORE_STRIDE:
            continue
        sensor, world = pose[:2, 3], velo_to_world(scan, pose)
        for k, (n, h, _nat, err) in enumerate(coverage(fov, ref, sensor)):
            out["cov"][k][0] += n
            out["cov"][k][1] += h
            out["cov"][k][2].append(err)
        out["curb"].append(curbs(fov, ref, *edges, CURB_D, sensor))
        for j, v in enumerate(ghosts(fov, world)):
            out["ghost"][j] += v
        for j, v in enumerate(obstacles(fov, ref, world)):
            out["obst"][j] += v
        # same seed per scan for every map: spot draw differs only where maps differ
        out["rows"].append(sample_spots(fov, ref, ref_terrain, world, sensor, float(t), np.random.default_rng([SEED, i])))
        out["label_ms"].append(label_map(fov, model, world, sensor, float(t), fresh_only=False))
        out["fresh_ms"].append(label_map(fov, model, world, sensor, float(t), fresh_only=True))
    out["rows"] = {k: np.concatenate([r[k] for r in out["rows"]]) for k in out["rows"][0]}
    return out


# print markdown table
def table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


# hazard scores for one prediction set
def hazard_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {"macro": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
            "f1": f1, "pr": pr, "rc": rc, "miss": float(np.mean(p[y == OBSTACLE] == TRAVERSABLE))}


# phase 5: uniform vs foveated vs foveated + model on same unseen test slice
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--test", type=Path, default=Path("data/kitti_long/sequences/08"))
    args = ap.parse_args()

    models = {}
    for name, bands in MAPS.items():
        t0 = time.perf_counter()
        models[name] = fit_model(build_rows(args.train, np.random.default_rng(SEED), bands=bands))
        print(f"TRAINED model on {name} features, seq {args.train.name}: {time.perf_counter() - t0:.0f} s")

    test = load_sequence(args.test)
    ref, ref_terrain = build_reference(test[0], test[1])
    edges = reference_edges(ref, CURB_D)
    n = len(test[0])
    path = np.sum(np.linalg.norm(np.diff(test[1][:, :2, 3], axis=0), axis=1))
    print(f"TEST seq {args.test.name}: {n} scans, {test[2][-1] - test[2][0]:.1f} s, {path:.0f} m driven. "
          f"scored every {SCORE_STRIDE} scans from {SCORE_FROM}. TRAIN seq {args.train.name}. no shared scans.")
    print()
    res = {name: run_map(bands, test, ref, ref_terrain, edges, models[name]) for name, bands in MAPS.items()}

    triad = [("1 uniform", "uniform U-20", False), ("2 foveated", "foveated F-A", False),
             ("3 foveated + model", "foveated F-A", True), ("(extra) uniform + model", "uniform U-20", True)]

    print("COST")
    rows = []
    for label, mk, use_model in triad:
        r = res[mk]
        ms = np.array(r["ms"][1:])
        lab = np.array(r["label_ms"]) if use_model else np.zeros(1)
        fresh = np.array(r["fresh_ms"]) if use_model else np.zeros(1)
        rows.append([label, f"{np.median(ms):.1f}", f"{np.percentile(ms, 95):.1f}",
                     f"{np.median(lab):.1f} / {np.median(fresh):.1f}" if use_model else "-",
                     f"{np.median(ms) + np.median(fresh):.1f}",
                     f"{max(r['store']):.2f}", f"{np.median(r['render']):.2f}"])
    table(["config", "map ms med", "map ms p95", "+ label ms med (whole map / this-scan cells)",
           "total ms med (map + this-scan label)", "store MB max", "rendered MB"], rows)

    print("HEIGHT RECONSTRUCTION vs dense reference (ground, metres) + coverage")
    rows = []
    for mk in MAPS:
        for k, (lo, hi) in enumerate(EVAL_RINGS):
            tot, hit, errs = res[mk]["cov"][k]
            e = np.concatenate(errs)
            rows.append([mk, f"{lo:g}-{hi:g} m", f"{hit / max(tot, 1) * 100:.0f}%", f"{np.median(e):.3f}",
                         f"{np.percentile(e, 95):.3f}"])
    table(["map", "ring", "cover", "ground err med", "ground err p95"], rows)

    print(f"PRESERVATION: curbs (probes {2 * CURB_D * 100:g} cm apart, share of ALL edges kept), obstacles, ghosts")
    rows = []
    for mk in MAPS:
        r = res[mk]
        ring = np.concatenate([c[0] for c in r["curb"]])
        kept = np.concatenate([c[2] for c in r["curb"]])
        curb = [f"{kept[ring == k].mean() * 100:.0f}%" for k in range(len(EVAL_RINGS))]
        t_, tp, f = r["obst"]
        rows.append([mk, *curb, f"{tp / max(t_, 1) * 100:.1f}%", f"{f / max(t_, 1):.2f}x", f"{r['ghost'][0] * 0.04:.0f}"])
    table(["map", "curb kept 2-10", "curb kept 10-25", "curb kept 25-50", "obstacle recall", "area bloat", "ghost m2"], rows)

    print("HAZARD LABELS on test (P / R / F1 per class). configs 1-2 = threshold rule on map, 3 = trained model.")
    rows = []
    for label, mk, use_model in triad:
        d = res[mk]["rows"]
        p = predict(models[mk], d["X"], d["map_obs"]) if use_model else d["rule"]
        s = hazard_scores(d["y"], p)
        cells = [f"{s['pr'][k]:.2f} / {s['rc'][k]:.2f} / {s['f1'][k]:.2f}" for k in range(len(CLASS_NAMES))]
        rows.append([label, f"{s['macro']:.3f}", *cells, f"{s['miss']:.1%}"])
    table(["config", "macro F1", *CLASS_NAMES, "obstacle called traversable"], rows)
    print(f"spots per map: {', '.join(f'{mk} {len(res[mk]['rows']['y'])}' for mk in MAPS)}. "
          f"thresholds: obstacle {TH.obstacle_h} m, step {TH.step} m, slope {TH.slope:.0%}, rough {TH.rough} m.")


if __name__ == "__main__":
    main()
