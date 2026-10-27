# 2.5D-LiDAR---Mini-Project
We are building a software-based **foveated 2.5D LiDAR elevation-mapping system** intended to support safer vehicle perception.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Layout

- `src/io/` — KITTI / SemanticKITTI loader (`.bin`, `.label`, `calib.txt`, `poses.txt`)
- `src/preprocess/` — range gate, ego box strip, frame transforms
- `src/mapping/` — uniform 2.5D grid (Phase 1), foveated ring-band grid (Phase 2)
- `src/temporal/` — offline reference accumulation, streaming foveated fusion (decay, ghost clearing, hand-off), local terrain
- `src/viz/` — top-down elevation + roughness plots
- `tests/` — synthetic tests

## Real data check (KITTI odometry seq 00)

```bash
python scripts/fetch_kitti_slice.py --seq 00 --count 100   # ~190 MB, range-read from KITTI zip
python -m scripts.check_ground                              # near-field ground stats + top-down PNG in outputs/
python -m scripts.build_reference_map                      # dense world-frame reference map + alignment stats
python -m scripts.eval_foveated                            # phase 2: uniform vs foveated memory, latency, error per ring
python -m scripts.eval_preservation                        # phase 2: curb / step / obstacle preservation per ring
python -m scripts.eval_temporal                            # phase 3: streaming fusion vs single scan: coverage, curbs, ghosts, cost
```

KITTI data is CC BY-NC-SA 3.0. Stays in `data/` (git ignored).
