# targeted tuning: class weights and obstacle probability threshold curve on val seq 05
import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src import config
from src.models.features import FEATURES
from src.models.targets import CLASS_NAMES, OBSTACLE, TRAVERSABLE, UNKNOWN
from scripts.train_hazard import (
    LABELS,
    RF_PARAMS,
    SEED,
    build_rows,
)

ZONE_M = 25.0


# predict with obstacle probability threshold and logodds evidence gate
def predict_with_threshold(proba, classes, Xm, obst_thresh=0.5):
    if obst_thresh > 0.0 and OBSTACLE in classes:
        obst_idx = list(classes).index(OBSTACLE)
        pred = classes[np.argmax(proba, axis=1)]
        # where obstacle is called but below threshold, pick next best class
        low_conf = (pred == OBSTACLE) & (proba[:, obst_idx] < obst_thresh)
        if low_conf.any():
            masked = proba.copy()
            masked[low_conf, obst_idx] = -1.0
            pred[low_conf] = classes[np.argmax(masked[low_conf], axis=1)]
    else:
        pred = classes[np.argmax(proba, axis=1)]

    # logodds evidence gate: obstacle without positive map evidence -> hazardous
    lo_idx = FEATURES.index("logodds")
    lo = Xm[:, lo_idx]
    pred[(pred == OBSTACLE) & ((lo <= 0) | np.isnan(lo))] = 1
    return pred


# compute metrics in 0-25 m reaction zone
def evaluate_preds(y_zone, pred_zone):
    f1 = f1_score(y_zone, pred_zone, labels=LABELS, average="macro", zero_division=0)
    pr, rc, f1c, sup = precision_recall_fscore_support(y_zone, pred_zone, labels=LABELS, zero_division=0)
    obst = y_zone == OBSTACLE
    miss = float(np.mean(pred_zone[obst] == TRAVERSABLE)) if obst.any() else 0.0
    trav = y_zone == TRAVERSABLE
    ghost = int(np.sum(pred_zone[trav] == OBSTACLE))  # false obstacles on flat road
    return {
        "macro_f1": f1,
        "obst_pr": pr[OBSTACLE],
        "obst_rc": rc[OBSTACLE],
        "obst_f1": f1c[OBSTACLE],
        "miss": miss,
        "ghost_obst": ghost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--val", type=Path, default=Path("data/kitti_long/sequences/05"))
    args = ap.parse_args()

    print("=" * 75, flush=True)
    print("STEP 4: TARGETED TUNING ON VALIDATION SEQ 05", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    print("Loading sequence 00 (train) and sequence 05 (val)...", flush=True)
    rng = np.random.default_rng(SEED)
    train = build_rows(args.train, rng, bands=config.FOVEATED_BANDS)
    val = build_rows(args.val, np.random.default_rng(SEED), bands=config.FOVEATED_BANDS)
    print(f"Data ready in {time.time() - t0:.1f}s. Train spots: {len(train['y'])}, Val spots: {len(val['y'])}", flush=True)

    obs_train = train["map_obs"]
    obs_val = val["map_obs"]
    X_train, y_train = train["X"][obs_train], train["y"][obs_train]
    X_val, y_val = val["X"][obs_val], val["y"][obs_val]

    val_zone = val["range"][obs_val] < ZONE_M
    y_zone = y_val[val_zone]
    X_zone = X_val[val_zone]

    # Test the class weight options:
    # 1. 'balanced' (baseline: penalizes obstacle misses heavily -> triggers false alarms)
    # 2. 'balanced_subsample' (recomputed per tree, gentler)
    # 3. None (natural class frequency)
    cw_options = ["balanced", "balanced_subsample", None]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    best_overall = None

    for cw in cw_options:
        cw_name = str(cw) if cw else "None"
        print(f"\n--- Testing class_weight = {cw_name} ---", flush=True)
        t_fit = time.time()
        model = RandomForestClassifier(**RF_PARAMS, class_weight=cw, random_state=SEED)
        model.fit(X_train, y_train)
        print(f"Model trained in {time.time() - t_fit:.1f}s. Computing probabilities...", flush=True)

        proba_zone = model.predict_proba(X_zone)
        classes = model.classes_

        print(f"\n| obst_thresh | Macro F1 | Obst Precision | Obst Recall | Obst F1 | Safety Miss | Ghost Obstacles |", flush=True)
        print("|---|---|---|---|---|---|---|", flush=True)

        for th in thresholds:
            pred_z = predict_with_threshold(proba_zone, classes, X_zone, obst_thresh=th)
            res = evaluate_preds(y_zone, pred_z)
            print(f"| {th:.2f} | {res['macro_f1']:.3f} | {res['obst_pr']:.3f} | {res['obst_rc']:.3f} | "
                  f"{res['obst_f1']:.3f} | {res['miss']:.1%} | {res['ghost_obst']:6d} |", flush=True)

            entry = {"cw": cw_name, "th": th, **res, "model": model}
            if best_overall is None or (res["macro_f1"] > best_overall["macro_f1"]):
                best_overall = entry

    print("\n" + "=" * 75, flush=True)
    print("OPTIMAL TUNING POINT ON VALIDATION SEQ 05:", flush=True)
    print(f"  class_weight:       {best_overall['cw']}", flush=True)
    print(f"  obstacle threshold: {best_overall['th']:.2f}", flush=True)
    print(f"  Macro F1 (0-25m):   {best_overall['macro_f1']:.3f}", flush=True)
    print(f"  Obstacle Precision: {best_overall['obst_pr']:.1%}", flush=True)
    print(f"  Obstacle Recall:    {best_overall['obst_rc']:.1%}", flush=True)
    print(f"  Safety Miss:        {best_overall['miss']:.1%}", flush=True)
    print(f"  Ghost Obstacles:    {best_overall['ghost_obst']} (down from ~8200)", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
