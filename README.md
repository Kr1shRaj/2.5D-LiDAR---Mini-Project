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
- `src/mapping/` — uniform 2.5D grid (Phase 1)
- `src/temporal/` — world-frame scan accumulation (offline reference map)
- `src/viz/` — top-down elevation + roughness plots
- `tests/` — synthetic tests

## Real data check (KITTI odometry seq 00)

```bash
python scripts/fetch_kitti_slice.py --seq 00 --count 100   # ~190 MB, range-read from KITTI zip
python -m scripts.check_ground                              # near-field ground stats + top-down PNG in outputs/
python -m scripts.build_reference_map                      # dense world-frame reference map + alignment stats
```

KITTI data is CC BY-NC-SA 3.0. Stays in `data/` (git ignored).
