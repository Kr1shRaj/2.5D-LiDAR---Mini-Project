import argparse
from pathlib import Path

import numpy as np

from src import config
from src.temporal.fusion import TemporalFoveatedMap
from src.temporal.snapshot import save_snapshot
from scripts.eval_temporal import PARAMS
from scripts.train_hazard import load_sequence


# stream one slice through fused F-A map; save rendered map + terrain every `every` scans after `start`
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--out", type=Path, default=Path("outputs/snapshots"))
    ap.add_argument("--start", type=int, default=10, help="skip first scans: map needs history")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()
    scans, poses, times = load_sequence(args.seq)
    ids = sorted(int(p.stem) for p in (args.seq / "velodyne").glob("*.bin"))
    out = args.out / f"mapv{config.MAP_VERSION}" / f"{args.seq.name}_{ids[0]:04d}-{ids[-1]:04d}"
    out.mkdir(parents=True, exist_ok=True)
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    n = 0
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        if i < args.start or (i - args.start) % args.every:
            continue
        meta = {"map_version": config.MAP_VERSION, "sequence": args.seq.name, "scan": ids[i], "index": i,
                "t": float(t), "pose": pose, "frame_note": "world = KITTI velodyne frame at sequence scan 0"}
        save_snapshot(out / f"{ids[i]:06d}.npz", m.render(), m.terrain, meta)
        n += 1
    size = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6
    print(f"WROTE {n} snapshots (map v{config.MAP_VERSION}, every {args.every} from {args.start}) to {out}. {size:.0f} MB.")


if __name__ == "__main__":
    main()
