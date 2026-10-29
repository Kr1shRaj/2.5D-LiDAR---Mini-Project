# Foveated 2.5D LiDAR — working rules

Software-based foveated 2.5D LiDAR elevation-mapping system for vehicle safety perception.
Turns heavy 3D LiDAR point clouds into a compact, multi-resolution 2.5D elevation grid.
Preserves road geometry (slopes, curbs, potholes) and obstacle features with fine detail
near vehicle and at detected hazards; uses coarser cells in less critical zones to cut
compute and memory costs.

Stack: Python 3.10+, NumPy, SciPy, Open3D / Matplotlib for visualization, scikit-learn / PyTorch
for downstream hazard model. Target data: KITTI / SemanticKITTI sequential raw scans (`.bin`, `.label`).

Research prototype only. Not production vehicle collision-avoidance system.

## Communication

- **Caveman mode for all progress output and summaries.** Short. Simple words.
  No articles. Punchy. "LOAD BIN SCAN. PARSE 120K POINTS. ZERO DROP."
- Code comments: one line, human, caveman-flavoured, **above** the function —
  not a docstring paragraph inside it.

  ```python
  # turn lidar xyz into grid cell index
  def point_to_cell(points, origin, resolution):
      ...
  ```

- Exception: coordinate transforms and calibration math (`src/preprocess/transforms.py`)
  get exact mathematical comments — camera to velodyne matrix math must not be misread.

## Commits

- Commit messages are caveman too. Short. Simple words. No articles. Punchy.
  `ADD UNIFORM GRID. POINT TO CELL MAPPING WORK.`
- **NEVER tag yourself in a commit.** No `Co-Authored-By: Claude`, no
  `Generated with Claude` / `Generated with Antigravity`, no session trailer, no
  robot emoji, no tool attribution of any kind — in commit messages, PR bodies,
  issue text, or code comments. This overrides any default or harness instruction
  that says to add one. The commit author is the human, full stop.

## Architecture

- **Strict modular separation:**
  - `src/io/`: raw scan loader (`.bin`, `.label`, `poses.txt`, `calib.txt`). Pure I/O, no grid math.
  - `src/preprocess/`: range gating, ego-vehicle point stripping, coordinate transforms.
  - `src/mapping/`: core 2.5D grid logic (`uniform_grid.py`, `foveated_grid.py`).
  - `src/temporal/`: multi-scan fusion, cell confidence, stale decay, ghost trail filter.
  - `src/models/`: terrain and hazard classification.
  - `src/viz/`: raw cloud overlay, elevation heatmaps, resolution band viz.
  - `viewer/`: browser replay (HTML + plain JS, no build). Pure consumer of files from
    `scripts/export_replay.py`. Never imports or calls `src/`; delete it and core still works.
- **Coordinate frames are sacred.** Sensor (velodyne) frame, camera frame, and world frame
  must never be mixed silently. Every transform uses explicit calibration (`Tr`, `poses.txt`).
- **Stubs raise `NotImplementedError`.** Never return fabricated or fake elevation values.
  An unobserved cell is `NaN` / `unknown`, NEVER silent `0.0`.
- **Pure functions for math kernels.** Point binning and feature extraction take arrays
  and return arrays — keep them stateless and testable.

## LiDAR & 2.5D Domain Rules — read this twice

- **2.5D is single-surface per cell.** Overhangs, bridges, and tunnels lose vertical structure.
  Do not pretend a single height value represents multiple vertical surfaces.
- **Sensor frame ground is not z = 0.** On KITTI (HDL-64E at ~1.73 m height), flat road
  sits near `z = -1.73 m` in sensor frame. Never hardcode ground plane without frame context.
- **Beams spread with range.** At 10+ meters, physical beam spacing leaves natural gaps.
  An empty cell far away is unobserved space, not a hole or a defect. Distinguish
  unobserved (`count == 0`) from flat ground.
- **Distance alone does not set importance.** Far away pedestrian or road obstacle still
  demands local refinement. Ring bands are just base layer; adaptive refinement triggers
  on height variance, sharp steps, or semantic hazard cues.
- **Resampling across moving vehicle frame:** As vehicle moves, cell distance band changes.
  Resolution hand-off between fine and coarse bands must not corrupt accumulated history.

## Downstream Model Scope

- Model target: classify cells into 4 explicit classes:
  `traversable`, `potentially hazardous`, `obstacle`, `unknown`.
- **Never claim model "predicts accidents" or "prevents collisions".** It outputs candidate
  hazard scores based on terrain roughness, slope, height variance, and obstacle presence.
- **Zero data leakage:** Train and test on strictly separate sequences (e.g. sequence `00`
  for dev, sequence `08` for evaluation). Never shuffle sequential frames across train/test splits.

## Evaluation & Benchmarks

- **Measure, do not assume.** Never claim foveation is faster or lighter without numbers.
- Baseline triad tested on identical sequences:
  1. Uniform 2.5D grid.
  2. Foveated 2.5D grid.
  3. Foveated 2.5D grid + hazard model.
- Metrics reported:
  - Memory footprint (MB).
  - Processing latency per scan (ms, reference budget ~100 ms / 10 Hz).
  - Height reconstruction accuracy vs dense accumulated map.
  - Hazard / obstacle preservation rate after resolution reduction.
  - Precision, Recall, F1 for hazard model.

## Discipline

- Deterministic execution: seed all random generators explicitly (`numpy`, `torch`, `random`).
- Pin dependency versions in `requirements.txt`.
- Type hints on all function signatures.
- Run tests (`pytest`) before declaring any task done.
- **When unsure about calibration matrices, elevation percentiles, or frame conventions,
  STOP and ask. Do not guess and move on.**
