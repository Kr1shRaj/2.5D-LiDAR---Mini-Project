import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from src import config
from src.temporal.fusion import TemporalFoveatedMap
from scripts.eval_temporal import PARAMS
from scripts.train_hazard import load_sequence


# profile fused map update (integrate + render) over one slice. prints top functions by own time.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    scans, poses, times = load_sequence(args.seq)
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    ms = []
    prof = cProfile.Profile()
    for scan, pose, t in zip(scans, poses, times):
        t0 = time.perf_counter()
        prof.enable()
        m.integrate(scan, pose, float(t))
        m.render()
        prof.disable()
        ms.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ms[1:])
    print(f"MAP (integrate + render) under profiler, seq {args.seq.name}: median {np.median(a):.1f} ms, "
          f"p95 {np.percentile(a, 95):.1f} ms (profiler adds overhead; use benchmark for real numbers)")
    pstats.Stats(prof).sort_stats("tottime").print_stats(args.top)
    pstats.Stats(prof).sort_stats("cumulative").print_stats("fusion.py|foveated_grid.py|terrain.py|uniform_grid.py", 25)


if __name__ == "__main__":
    main()
