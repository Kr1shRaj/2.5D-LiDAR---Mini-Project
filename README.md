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
- `src/viz/` — top-down elevation + roughness plots
- `tests/` — synthetic tests
