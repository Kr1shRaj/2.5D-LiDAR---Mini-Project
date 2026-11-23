import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.models.labeler import HazardLabeler
from src.planning.corridor import BLOCKED, VERDICTS, CorridorParams, check_corridor
from src.temporal.fusion import TemporalFoveatedMap
from scripts.benchmark import ZONE_M
from scripts.eval_temporal import PARAMS
from scripts.export_replay import load_or_train
from scripts.train_hazard import load_sequence

CP = CorridorParams()
STRAIGHT = len(CP.curvatures) // 2


# stream slice: map + labeler + timed corridor check per scan. list scans where straight is blocked
# and another arc is a candidate. candidate corridor check only: KITTI car really drove, nothing steered.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/kitti_long/sequences/00"))
    ap.add_argument("--seq", type=Path, default=Path("data/kitti_long/sequences/08"))
    ap.add_argument("--model", type=Path, default=Path(f"outputs/replay/model_fa_seq00_mapv{config.MAP_VERSION}.joblib"))
    ap.add_argument("--out", type=Path, default=Path("outputs/corridor"))
    args = ap.parse_args()
    model = load_or_train(args.model, args.train)
    scans, poses, times = load_sequence(args.seq)
    ids = sorted(int(p.stem) for p in (args.seq / "velodyne").glob("*.bin"))
    m = TemporalFoveatedMap(config.FOVEATED_BANDS, PARAMS)
    labeler = HazardLabeler(model, ZONE_M)
    rows, ms, prev = [], [], None
    for i, (scan, pose, t) in enumerate(zip(scans, poses, times)):
        m.integrate(scan, pose, float(t))
        fov = m.render()
        labeler.update(fov, m.terrain, float(t))
        t0 = time.perf_counter()
        # arcs stay inside 20 m < model zone: every observed probe has a model label, -1 = unobserved.
        # obstacle counts only with map evidence (log-odds > 0)
        r = check_corridor(lambda xy: labeler.labels_at(fov, xy), pose, CP, prev,
                           lambda xy: np.nan_to_num(fov.sample(xy, "logodds")) > 0)
        ms.append((time.perf_counter() - t0) * 1e3)
        prev = r.curvatures[r.pick] if r.pick >= 0 else None
        rows.append({"i": i, "scan": ids[i], "verdict": r.verdict.tolist(), "block_m": r.block_m.round(2).tolist(),
                     "hazard": r.hazard.round(3).tolist(), "unknown": r.unknown.round(3).tolist(), "pick": r.pick})
    events = [w for w in rows if w["verdict"][STRAIGHT] == BLOCKED and w["pick"] >= 0 and w["pick"] != STRAIGHT]
    none = sum(w["pick"] < 0 for w in rows)
    ms_arr = np.array(ms[1:])
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.seq.name}_{ids[0]:04d}-{ids[-1]:04d}.json"
    out.write_text(json.dumps({"params": CP.__dict__, "verdicts": VERDICTS, "map_version": config.MAP_VERSION,
                               "rows": rows, "events": [w["i"] for w in events]}))
    print(f"CORRIDOR CHECK seq {args.seq.name} ({len(rows)} scans, {len(CP.curvatures)} arcs x {CP.length_m:g} m). "
          f"candidate check only, nothing steered.")
    print(f"corridor ms median {np.median(ms_arr):.2f}, p95 {np.percentile(ms_arr, 95):.2f}, max {ms_arr.max():.2f}")
    print(f"straight blocked: {sum(w['verdict'][STRAIGHT] == BLOCKED for w in rows)} scans. "
          f"straight blocked + other candidate: {len(events)}. no candidate at all: {none}.")
    for w in events[:40]:
        k = CP.curvatures[w["pick"]]
        side = "left" if k > 0 else "right"
        print(f"  frame {w['i']:3d} (scan {w['scan']}): straight blocked at {w['block_m'][STRAIGHT]:.1f} m -> "
              f"candidate {side} k={k:+.3f} (R {abs(1 / k):.0f} m), {VERDICTS[w['verdict'][w['pick']]]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
