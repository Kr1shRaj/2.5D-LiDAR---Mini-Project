import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from src import config
from src.models.labeler import HazardLabeler
from src.planning.corridor import BLOCKED, CorridorParams, arc_probes_velo, probes_to_world, score_arcs
from src.temporal.fusion import TemporalFoveatedMap
from scripts.benchmark import ZONE_M
from scripts.eval_temporal import PARAMS
from scripts.export_replay import load_or_train
from scripts.train_hazard import load_sequence

BASE = CorridorParams()
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


# tune corridor cluster rule on TRAIN sequence only. truth signal: arc nearest driven path was driveable,
# so "driven arc blocked" = false block. lower is better; must not reach it by ignoring obstacles everywhere.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--model", type=Path, default=Path(f"outputs/replay/model_fa_seq00_mapv{config.MAP_VERSION}.joblib"))
    args = ap.parse_args()
    model = load_or_train(args.model, args.train)
    scans, poses, times = load_sequence(args.train)
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    probes_velo = arc_probes_velo(BASE)
    labels, driven = [], []
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        fov = m.render()
        labeler.update(fov, m.terrain, float(t))
        k = driven_curvature(poses, i)
        if i < 10 or k is None or abs(k) > max(BASE.curvatures):
            continue
        w = probes_to_world(probes_velo, pose)
        labels.append(labeler.labels_at(fov, w.reshape(-1, 2)).reshape(w.shape[:3]))
        driven.append(int(np.argmin(np.abs(np.array(BASE.curvatures) - k))))
    print(f"TUNE corridor on seq {args.train.name} (train). {len(labels)} scans with driven path 10-20 m ahead.")
    print("| min_hits | window m | driven arc blocked (false block) | arcs blocked, all | no candidate |")
    print("|---|---|---|---|---|")
    for window in (0.6, 1.0):
        for hits in (1, 2, 4, 6, 8, 12):
            p = replace(BASE, min_hits=hits, window_m=window)
            res = [score_arcs(lab, p) for lab in labels]
            fb = np.mean([r.verdict[d] == BLOCKED for r, d in zip(res, driven)])
            allb = np.mean([np.mean(r.verdict == BLOCKED) for r in res])
            none = np.mean([r.pick < 0 for r in res])
            print(f"| {hits} | {window:g} | {fb:.1%} | {allb:.1%} | {none:.1%} |")


if __name__ == "__main__":
    main()
