import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src import config
from src.io.kitti_loader import load_calib, load_poses, load_times
from src.mapping.uniform_grid import GridLayers, build_uniform_grid, spec_from_bounds
from src.models.features import FEATURES, cell_features, feature_matrix, grid_sampler
from src.models.targets import CLASS_NAMES, OBSTACLE, TRAVERSABLE, UNKNOWN, Thresholds, geometric_classes
from src.preprocess.transforms import velo_poses_from_cam_poses, velo_to_world
from src.temporal.accumulate import accumulate_world_points
from src.temporal.fusion import TemporalFoveatedMap
from src.temporal.terrain import Terrain, local_terrain
from src.viz.grid_viz import plot_class_maps
from scripts.eval_foveated import load_filtered
from scripts.eval_temporal import PARAMS

SEED = 0
R = config.RANGE_MAX_M
REF_RES = 0.1
SAMPLE_FROM = 10  # fused map gets 1 s of history first
SAMPLE_STRIDE = 5
CELLS_PER_SCAN = 30000
EVAL_RINGS = ((2.0, 10.0), (10.0, 25.0), (25.0, 50.0))
TH = Thresholds(obstacle_h=config.OBSTACLE_H_M, step=config.HAZARD_STEP_M, slope=config.HAZARD_SLOPE,
                rough=config.HAZARD_ROUGH_M)
LABELS = list(range(len(CLASS_NAMES)))
MAX_GRADE = 0.10  # slice whose poses climb more than 10 % of path = broken ground truth poses (seq 08 frames 0-99)


# one sequence slice: scans in velodyne frame, poses, times
def load_sequence(seq_dir: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    paths = sorted((seq_dir / "velodyne").glob("*.bin"))
    ids = [int(p.stem) for p in paths]
    calib = load_calib(seq_dir / "calib.txt")
    poses = velo_poses_from_cam_poses(load_poses(seq_dir / "poses.txt"), calib["Tr"])[ids]
    path = np.sum(np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1))
    rise = abs(poses[-1, 2, 3] - poses[0, 2, 3])
    if rise > MAX_GRADE * max(path, 1.0):
        raise ValueError(f"{seq_dir}: poses climb {rise:.1f} m over {path:.0f} m path. ground truth poses look broken.")
    return load_filtered(paths), poses, load_times(seq_dir / "times.txt")[ids]


# dense reference from whole slice (truth side only, never model input) + its local terrain
def build_reference(scans: list[np.ndarray], poses: np.ndarray) -> tuple[GridLayers, Terrain]:
    world = accumulate_world_points(scans, poses)
    lo, hi = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
    ref = build_uniform_grid(world, spec_from_bounds(lo[0], hi[0], lo[1], hi[1], REF_RES, "world"), "world",
                             config.GROUND_PERCENTILE)
    terrain = local_terrain(world, "world", config.TERRAIN_CELL_M, config.GROUND_PERCENTILE)
    return ref, terrain


# reference lattice centres within R of sensor
def window_centers(ref: GridLayers, sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = ref.spec
    i0 = max(int((sensor[0] - R - s.x_min) / s.resolution), 0)
    i1 = min(int((sensor[0] + R - s.x_min) / s.resolution) + 1, s.nx)
    j0 = max(int((sensor[1] - R - s.y_min) / s.resolution), 0)
    j1 = min(int((sensor[1] + R - s.y_min) / s.resolution) + 1, s.ny)
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    xy = np.column_stack([s.x_min + (ii + 0.5) * s.resolution, s.y_min + (jj + 0.5) * s.resolution])
    keep = np.hypot(xy[:, 0] - sensor[0], xy[:, 1] - sensor[1]) < R
    return xy[keep], ref.count[ii[keep], jj[keep]]


# stream slice through fused map; sample spots; features from online map, targets from reference
def build_rows(seq_dir: Path, rng: np.random.Generator, keep_last_map: bool = False) -> dict:
    scans, poses, times = load_sequence(seq_dir)
    ref, ref_terrain = build_reference(scans, poses)
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    rows = {k: [] for k in ("X", "y", "rule", "map_obs", "range", "scan")}
    last = None
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        if i < SAMPLE_FROM or (i - SAMPLE_FROM) % SAMPLE_STRIDE:
            continue
        fov = m.render()
        sensor = pose[:2, 3]
        xy, ref_count = window_centers(ref, sensor)
        map_count = np.nan_to_num(fov.sample(xy, "count"))
        cand = np.flatnonzero((ref_count > 0) | (map_count > 0))  # spots nobody ever saw: skip, trivial
        pick = rng.choice(cand, size=min(CELLS_PER_SCAN, cand.size), replace=False)
        xy = xy[pick]
        scan_terrain = local_terrain(velo_to_world(scan, pose), "world", config.TERRAIN_CELL_M,
                                     config.GROUND_PERCENTILE)
        f_map = cell_features(fov.sample_many, xy, scan_terrain.sample(xy), sensor, float(t))
        f_ref = cell_features(grid_sampler(ref), xy, ref_terrain.sample(xy), sensor, float(t))
        rows["X"].append(feature_matrix(f_map))
        rows["y"].append(geometric_classes(f_ref, ref_count[pick] >= config.REF_MIN_POINTS, TH))
        rows["rule"].append(geometric_classes(f_map, map_count[pick] > 0, TH))
        rows["map_obs"].append(map_count[pick] > 0)
        rows["range"].append(f_map["range"])
        rows["scan"].append(np.full(len(xy), i))
        if keep_last_map:
            last = (fov, ref, ref_terrain, scan_terrain, sensor, float(t))
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["last"] = last
    return out


# model output: unknown where map saw nothing, classifier elsewhere
def predict(model: HistGradientBoostingClassifier, X: np.ndarray, map_obs: np.ndarray) -> np.ndarray:
    out = np.full(len(X), UNKNOWN, dtype=np.int64)
    if map_obs.any():
        out[map_obs] = model.predict(X[map_obs])
    return out


# per class precision / recall / f1 + support table
def report(name: str, y: np.ndarray, p: np.ndarray) -> None:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    print(f"{name}: macro F1 {f1_score(y, p, labels=LABELS, average='macro', zero_division=0):.3f}  "
          f"(3 known classes: {f1_score(y, p, labels=LABELS[:3], average='macro', zero_division=0):.3f})")
    print("| class | precision | recall | F1 | support |")
    print("|---|---|---|---|---|")
    for k, c in enumerate(CLASS_NAMES):
        print(f"| {c} | {pr[k]:.3f} | {rc[k]:.3f} | {f1[k]:.3f} | {sup[k]} |")
    print()


# confusion matrix as markdown. rows = truth, cols = predicted.
def confusion(name: str, y: np.ndarray, p: np.ndarray) -> None:
    cm = confusion_matrix(y, p, labels=LABELS)
    print(f"{name} confusion (rows truth, cols predicted)")
    print("| truth \\ pred | " + " | ".join(CLASS_NAMES) + " |")
    print("|---" * (len(CLASS_NAMES) + 1) + "|")
    for k, c in enumerate(CLASS_NAMES):
        print(f"| {c} | " + " | ".join(str(v) for v in cm[k]) + " |")
    print()


# phase 4: train hazard classifier on seq 00, test on seq 08. no shared scans, no shuffle across sequences.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti/sequences/00"))
    ap.add_argument("--test", type=Path, default=Path("data/kitti/sequences/08"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    train = build_rows(args.train, rng)
    test = build_rows(args.test, rng, keep_last_map=True)
    print(f"TRAIN {args.train.name}: {len(train['y'])} spots from {len(np.unique(train['scan']))} scans. "
          f"TEST {args.test.name}: {len(test['y'])} spots from {len(np.unique(test['scan']))} scans.")
    print(f"THRESHOLDS: obstacle >= {TH.obstacle_h} m over terrain, step >= {TH.step} m / 40 cm, "
          f"slope >= {TH.slope:.0%}, roughness >= {TH.rough} m. unknown: reference < {config.REF_MIN_POINTS} pts.")
    for name, d in (("train", train), ("test", test)):
        share = np.bincount(d["y"], minlength=4) / len(d["y"])
        print(f"  {name} truth mix: " + ", ".join(f"{c} {s:.1%}" for c, s in zip(CLASS_NAMES, share)))
    print()

    obs = train["map_obs"]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                           early_stopping=False, random_state=SEED)
    model.fit(train["X"][obs], train["y"][obs])
    joblib.dump({"model": model, "features": FEATURES, "classes": CLASS_NAMES}, args.out / "hazard_hgb.joblib")

    p_train = predict(model, train["X"], train["map_obs"])
    print(f"TRAIN (seq {args.train.name}) model macro F1 {f1_score(train['y'], p_train, labels=LABELS, average='macro'):.3f}"
          " (fit check only)")
    print()
    p_test = predict(model, test["X"], test["map_obs"])
    print(f"=== TEST seq {args.test.name} (never seen in training) ===")
    print()
    report("MODEL (gradient boosted trees on fused F-A map features)", test["y"], p_test)
    report("RULE BASELINE (same thresholds applied straight to fused map)", test["y"], test["rule"])
    confusion("MODEL", test["y"], p_test)
    confusion("RULE BASELINE", test["y"], test["rule"])

    # reference also holds what later scans saw. map-observed spots = what map could know at that time.
    o = test["map_obs"]
    print(f"MAP-OBSERVED SPOTS ONLY ({o.mean():.0%} of test spots): judge model where map has data")
    report("MODEL, map-observed", test["y"][o], p_test[o])
    report("RULE, map-observed", test["y"][o], test["rule"][o])
    obst = test["y"] == OBSTACLE
    print("SAFETY MISS: true obstacle called traversable (lower = safer)")
    print(f"  model {np.mean(p_test[obst] == TRAVERSABLE):.1%}   rule {np.mean(test['rule'][obst] == TRAVERSABLE):.1%}")
    print()

    print("MACRO F1 PER RING (test)")
    print("| ring | spots | model | rule |")
    print("|---|---|---|---|")
    for lo, hi in EVAL_RINGS:
        s = (test["range"] >= lo) & (test["range"] < hi)
        fm = f1_score(test["y"][s], p_test[s], labels=LABELS, average="macro", zero_division=0)
        fr = f1_score(test["y"][s], test["rule"][s], labels=LABELS, average="macro", zero_division=0)
        print(f"| {lo:g}-{hi:g} m | {s.sum()} | {fm:.3f} | {fr:.3f} |")
    print()

    # picture: truth vs model vs rule around last scored test scan, 0.2 m raster
    fov, ref, ref_terrain, scan_terrain, sensor, t = test["last"]
    g = np.arange(-40.0, 40.0, 0.2) + 0.1
    gx, gy = np.meshgrid(sensor[0] + g, sensor[1] + g, indexing="ij")
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    ref_count = np.nan_to_num(grid_sampler(ref)(xy, ["count"])["count"])
    map_count = np.nan_to_num(fov.sample(xy, "count"))
    f_map = cell_features(fov.sample_many, xy, scan_terrain.sample(xy), sensor, t)
    f_ref = cell_features(grid_sampler(ref), xy, ref_terrain.sample(xy), sensor, t)
    truth = geometric_classes(f_ref, ref_count >= config.REF_MIN_POINTS, TH)
    truth[(ref_count == 0) & (map_count == 0)] = -1  # never seen: blank
    pred = predict(model, feature_matrix(f_map), map_count > 0)
    rule = geometric_classes(f_map, map_count > 0, TH)
    for a in (pred, rule):
        a[(ref_count == 0) & (map_count == 0)] = -1
    shape = gx.shape
    plot_class_maps([truth.reshape(shape), pred.reshape(shape), rule.reshape(shape)],
                    ["Truth (dense reference + thresholds)", "Model (fused map)", "Rule on fused map"],
                    extent=(gx.min() - 0.1, gx.max() + 0.1, gy.min() - 0.1, gy.max() + 0.1),
                    class_names=CLASS_NAMES, out_path=args.out / f"hazard_{args.test.name}_classes.png",
                    title=f"seq {args.test.name} test scan, world frame. Candidate hazard labels, not collision prediction.")


if __name__ == "__main__":
    main()
