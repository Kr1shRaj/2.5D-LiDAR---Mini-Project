import argparse
from pathlib import Path

import numpy as np

from src import config
from src.temporal.fusion import TemporalFoveatedMap
from scripts.eval_temporal import PARAMS
from scripts.train_hazard import load_sequence

SEED = 0
FROM, STRIDE = 10, 10  # 1 s of history first, then every 10th scan
SPOTS = 20000  # random spots per offset slice per scored scan
OFFSETS = ((-0.3, -0.2), (-0.2, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4))


# observed share of map lookups in thin slices around each inner ring edge. seam = dip right past edge.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/08"))
    args = ap.parse_args()
    scans, poses, times = load_sequence(args.seq)
    edges = [b.r_max for b in config.FOVEATED_BANDS[:-1]]
    rng = np.random.default_rng(SEED)
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    hits = {(e, o): [] for e in edges for o in OFFSETS}
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        if i < FROM or (i - FROM) % STRIDE:
            continue
        fov = m.render()
        c = np.array(fov.center)
        for e in edges:
            for lo, hi in OFFSETS:
                r = rng.uniform(e + lo, e + hi, SPOTS)
                a = rng.uniform(0, 2 * np.pi, SPOTS)
                xy = c + np.column_stack([r * np.cos(a), r * np.sin(a)])
                hits[(e, (lo, hi))].append(np.mean(np.nan_to_num(fov.sample(xy, "count")) > 0))
    n = len(next(iter(hits.values())))
    print(f"SEAM CHECK seq {args.seq.name}: observed share of map lookups by offset from ring edge. "
          f"{n} scans scored (every {STRIDE} from {FROM}), {SPOTS} spots per slice.")
    print("| edge | " + " | ".join(f"{lo:+.1f}..{hi:+.1f} m" for lo, hi in OFFSETS) + " |")
    print("|---" * (len(OFFSETS) + 1) + "|")
    for e in edges:
        print(f"| {e:g} m | " + " | ".join(f"{np.mean(hits[(e, o)]):.1%}" for o in OFFSETS) + " |")


if __name__ == "__main__":
    main()
