# slice model predictions into buckets. find where and why it fails.
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src import config
from src.models.features import FEATURES, feature_matrix
from src.models.targets import CLASS_NAMES, OBSTACLE, TRAVERSABLE, UNKNOWN
from scripts.train_hazard import (
    LABELS,
    SEED,
    TH,
    build_rows,
    fit_model,
    predict,
)

ZONE_M = 25.0


# macro f1 and obstacle-called-traversable for a slice
def slice_scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) == 0:
        return 0.0, 0.0
    f1 = f1_score(y, p, labels=LABELS, average="macro", zero_division=0)
    obst = y == OBSTACLE
    miss = float(np.mean(p[obst] == TRAVERSABLE)) if obst.any() else 0.0
    return f1, miss


# print one diagnosis table
def diag_table(title: str, names: list[str], y: np.ndarray, p: np.ndarray,
               masks: list[np.ndarray]) -> None:
    print(f"\n{title}")
    print("| bucket | spots | macro F1 | obst->trav |")
    print("|---|---|---|---|")
    for name, mask in zip(names, masks):
        f1, miss = slice_scores(y[mask], p[mask])
        print(f"| {name} | {mask.sum()} | {f1:.3f} | {miss:.1%} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--test", type=Path, default=Path("data/kitti_long/sequences/08"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print("BUILDING TRAIN ROWS (seq 00)...")
    train_fov = build_rows(args.train, rng, bands=config.FOVEATED_BANDS)
    print("BUILDING TEST ROWS (seq 08)...")
    test_fov = build_rows(args.test, np.random.default_rng(SEED), bands=config.FOVEATED_BANDS)

    print("BUILDING UNIFORM TEST ROWS (seq 08)...")
    from src.mapping.foveated_grid import Band
    test_uni = build_rows(args.test, np.random.default_rng(SEED),
                          bands=(Band(0.0, config.RANGE_MAX_M, 0.2),))

    print("TRAINING FOVEATED MODEL...")
    model_fov = fit_model(train_fov)
    print("TRAINING UNIFORM MODEL...")
    train_uni = build_rows(args.train, rng, bands=(Band(0.0, config.RANGE_MAX_M, 0.2),))
    model_uni = fit_model(train_uni)

    # predictions
    p_fov = predict(model_fov, test_fov["X"], test_fov["map_obs"])
    p_uni = predict(model_uni, test_uni["X"], test_uni["map_obs"])
    y_fov = test_fov["y"]
    y_uni = test_uni["y"]

    # feature columns for foveated
    fi = {name: i for i, name in enumerate(FEATURES)}
    X = test_fov["X"]
    rng_col = test_fov["range"]
    res_col = X[:, fi["resolution"]]
    inherited_col = X[:, fi["inherited"]]
    logodds_col = X[:, fi["logodds"]]
    count_col = X[:, fi["count"]]
    roughness_col = X[:, fi["roughness"]]
    rel_max_col = X[:, fi["rel_max"]]
    weight_col = X[:, fi["weight"]]

    # points layer: density = points / resolution^2 (not in FEATURES yet, get from raw)
    # use count as proxy for now since points layer needs map sampler

    zone = rng_col < ZONE_M

    print("\n" + "=" * 60)
    print("DIAGNOSIS: FOVEATED MODEL, 0-25 m ZONE")
    print("=" * 60)

    # --- 1. BY CELL SIZE ---
    diag_table(
        "1. BY CELL RESOLUTION (foveated)",
        ["10 cm cells", "20 cm cells", "50 cm cells"],
        y_fov, p_fov,
        [zone & (np.abs(res_col - 0.1) < 0.01),
         zone & (np.abs(res_col - 0.2) < 0.01),
         zone & (np.abs(res_col - 0.5) < 0.01)],
    )

    # uniform comparison
    zone_uni = test_uni["range"] < ZONE_M
    f1_uni, miss_uni = slice_scores(y_uni[zone_uni], p_uni[zone_uni])
    print(f"| uniform 20 cm (reference) | {zone_uni.sum()} | {f1_uni:.3f} | {miss_uni:.1%} |")

    # --- 2. BY INHERITED FLAG ---
    inh = np.nan_to_num(inherited_col, nan=0.0) > 0.5
    diag_table(
        "2. BY INHERITED FLAG (foveated, 0-25 m)",
        ["inherited = NO (real data)", "inherited = YES (placeholder)"],
        y_fov, p_fov,
        [zone & ~inh, zone & inh],
    )

    # --- 3. NEAR RING EDGES ---
    near_10m = np.abs(rng_col - 10.0) < 0.3
    near_25m = np.abs(rng_col - 25.0) < 0.3
    away = ~near_10m & ~near_25m
    diag_table(
        "3. NEAR RING BOUNDARIES (foveated, 0-25 m)",
        ["within 30 cm of 10 m edge", "within 30 cm of 25 m edge", "away from edges"],
        y_fov, p_fov,
        [zone & near_10m, zone & near_25m, zone & away],
    )

    # --- 4. BY OBSERVATION COUNT ---
    diag_table(
        "4. BY OBSERVATION COUNT (foveated, 0-25 m)",
        ["count 0-2", "count 2-5", "count 5-10", "count 10+"],
        y_fov, p_fov,
        [zone & (count_col >= 0) & (count_col < 2),
         zone & (count_col >= 2) & (count_col < 5),
         zone & (count_col >= 5) & (count_col < 10),
         zone & (count_col >= 10)],
    )

    # --- 5. BY RANGE ---
    diag_table(
        "5. BY RANGE (foveated)",
        ["2-10 m (fine ring)", "10-25 m (mid ring)"],
        y_fov, p_fov,
        [(rng_col >= 2) & (rng_col < 10),
         (rng_col >= 10) & (rng_col < 25)],
    )

    # --- 6. FLAT-ROAD FALSE OBSTACLE TABLE ---
    print("\n" + "=" * 60)
    print("FLAT-ROAD FALSE OBSTACLES: model says OBSTACLE, truth says TRAVERSABLE")
    print("=" * 60)
    false_obst = zone & (p_fov == OBSTACLE) & (y_fov == TRAVERSABLE)
    true_obst = zone & (p_fov == OBSTACLE) & (y_fov == OBSTACLE)
    print(f"\nTotal false obstacle cells (0-25 m): {false_obst.sum()}")
    print(f"Total correct obstacle cells (0-25 m): {true_obst.sum()}")
    print(f"False obstacle rate: {false_obst.sum() / max((p_fov[zone] == OBSTACLE).sum(), 1):.1%}")

    if false_obst.any():
        print("\nFalse obstacle cell statistics (median / mean / std):")
        for feat_name in ["rel_max", "logodds", "roughness", "count", "inherited", "resolution", "weight"]:
            vals = X[false_obst, fi[feat_name]]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                print(f"  {feat_name:12s}: median {np.median(vals):7.3f}  mean {np.mean(vals):7.3f}  std {np.std(vals):7.3f}")

        print("\nrel_max histogram (false obstacles):")
        bins = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 5.0]
        vals = X[false_obst, fi["rel_max"]]
        vals = vals[~np.isnan(vals)]
        counts, _ = np.histogram(vals, bins=bins)
        for i in range(len(bins) - 1):
            bar = "#" * min(int(counts[i] / max(counts.max(), 1) * 40), 40)
            print(f"  {bins[i]:5.2f}-{bins[i+1]:5.2f} m: {counts[i]:6d} {bar}")

        print("\nlogodds histogram (false obstacles):")
        lo_bins = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
        vals = X[false_obst, fi["logodds"]]
        vals = vals[~np.isnan(vals)]
        counts, _ = np.histogram(vals, bins=lo_bins)
        for i in range(len(lo_bins) - 1):
            bar = "#" * min(int(counts[i] / max(counts.max(), 1) * 40), 40)
            print(f"  {lo_bins[i]:5.1f} to {lo_bins[i+1]:5.1f}: {counts[i]:6d} {bar}")

        print("\ninherited breakdown (false obstacles):")
        inh_vals = X[false_obst, fi["inherited"]]
        inh_vals = np.nan_to_num(inh_vals, nan=0.0)
        n_inh = (inh_vals > 0.5).sum()
        n_real = (inh_vals <= 0.5).sum()
        print(f"  inherited (placeholder): {n_inh} ({n_inh / len(inh_vals):.1%})")
        print(f"  real data:              {n_real} ({n_real / len(inh_vals):.1%})")

        print("\nresolution breakdown (false obstacles):")
        res_vals = X[false_obst, fi["resolution"]]
        for r in [0.1, 0.2, 0.5]:
            n = (np.abs(res_vals - r) < 0.01).sum()
            print(f"  {r*100:.0f} cm cells: {n} ({n / len(res_vals):.1%})")

    # --- 7. CONCLUSION ---
    print("\n" + "=" * 60)
    print("SUMMARY NUMBERS")
    print("=" * 60)
    f1_fov_zone, miss_fov_zone = slice_scores(y_fov[zone], p_fov[zone])
    print(f"Foveated + model, 0-25 m: macro F1 = {f1_fov_zone:.3f}, obst->trav = {miss_fov_zone:.1%}")
    print(f"Uniform  + model, 0-25 m: macro F1 = {f1_uni:.3f}, obst->trav = {miss_uni:.1%}")
    print(f"Gap: {f1_uni - f1_fov_zone:.3f}")
    print()
    print("WRITE YOUR CONCLUSION HERE AFTER READING THE TABLES ABOVE.")


if __name__ == "__main__":
    main()
