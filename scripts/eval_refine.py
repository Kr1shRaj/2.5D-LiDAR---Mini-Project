import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src import config
from src.mapping.foveated_grid import Band
from src.preprocess.transforms import velo_to_world
from src.temporal.fusion import FusionParams, TemporalFoveatedMap
from scripts.benchmark import CURB_D
from scripts.eval_preservation import reference_edges
from scripts.eval_temporal import EVAL_RINGS, PARAMS, coverage, curbs, obstacles
from scripts.train_hazard import build_reference, load_sequence

FROM, STRIDE = 10, 10
OFF = replace(PARAMS, refine_res=0.0)
# name -> (bands, params). tune variants on TRAIN sequence; benchmark scores the chosen one on test.
FA = config.FOVEATED_BANDS
VARIANTS: dict[str, tuple[tuple[Band, ...], FusionParams]] = {
    "uniform U-20": ((Band(0.0, config.RANGE_MAX_M, 0.2),), OFF),
    "foveated, no refine": (FA, OFF),
    "map v5 config (budget 3000, cues every 2nd scan)": (FA, PARAMS),
    "no budget, cues every scan": (FA, replace(PARAMS, refine_budget=0, refine_every=1)),
}
FAR_CELLS = np.pi * (FA[-1].r_max ** 2 - FA[-1].r_min ** 2) / FA[-1].resolution ** 2  # outer ring area in coarse cells


# stream slice per variant: curbs kept per ring, ground cover / error, obstacle recall, map ms, memory, patch share
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--only", nargs="*", default=None, help="variant names to run")
    args = ap.parse_args()
    scans, poses, times = load_sequence(args.seq)
    ref, _ = build_reference(scans, poses)
    edges = reference_edges(ref, CURB_D)
    print(f"ADAPTIVE REFINEMENT seq {args.seq.name}: {len(scans)} scans, scored every {STRIDE} from {FROM}. "
          f"map v{config.MAP_VERSION}. curbs = sharp edges, probes {2 * CURB_D * 100:g} cm apart.")
    print("| variant | curb kept 2-10 | 10-25 | 25-50 | ground cover 25-50 | ground err med 25-50 | obstacle recall | "
          "map ms med / p95 | store MB max | far ring area refined |")
    print("|---" * 10 + "|")
    for name, (bands, params) in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        m = TemporalFoveatedMap(bands, params)
        ms, mb, cur, cov, obs, share = [], [], [], [[0, 0, []] for _ in EVAL_RINGS], [0, 0, 0], []
        for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
            t0 = time.perf_counter()
            m.integrate(scan, pose, float(t))
            fov = m.render()
            ms.append((time.perf_counter() - t0) * 1e3)
            mb.append(m.nbytes / 1e6)
            if i < FROM or (i - FROM) % STRIDE:
                continue
            sensor, world = pose[:2, 3], velo_to_world(scan, pose)
            cur.append(curbs(fov, ref, *edges, CURB_D, sensor))
            for k, (n, h, _nat, err) in enumerate(coverage(fov, ref, sensor)):
                cov[k][0] += n
                cov[k][1] += h
                cov[k][2].append(err)
            for j, v in enumerate(obstacles(fov, ref, world)):
                obs[j] += v
            share.append(m.ref_keys.size / FAR_CELLS if m.refine_on else 0.0)
        ring = np.concatenate([c[0] for c in cur])
        kept = np.concatenate([c[2] for c in cur])
        kc = [f"{kept[ring == k].mean() * 100:.0f}%" for k in range(len(EVAL_RINGS))]
        a = np.array(ms[1:])
        tot, hit, errs = cov[-1]
        print(f"| {name} | {kc[0]} | {kc[1]} | {kc[2]} | {hit / max(tot, 1) * 100:.0f}% | "
              f"{np.median(np.concatenate(errs)):.3f} | {obs[1] / max(obs[0], 1) * 100:.1f}% | "
              f"{np.median(a):.1f} / {np.percentile(a, 95):.1f} | {max(mb):.2f} | {np.mean(share):.0%} |")


if __name__ == "__main__":
    main()
