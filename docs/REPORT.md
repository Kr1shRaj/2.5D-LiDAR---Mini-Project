# Foveated 2.5D LiDAR Elevation Mapping for Vehicle Safety — Project Report

Research prototype. Not a production collision-avoidance system. All hazard outputs are
**candidate labels from geometry**, not accident or collision predictions.

Data: KITTI odometry (real HDL-64E recordings, replayed offline). Train slice: sequence 00, scans 0–299.
Test slice: sequence 08, scans 1500–1799 (31 s, 252 m). Phases 1–3 were developed on sequence 00, scans 0–99.
Every number below was measured on these slices on a shared 4-core cloud VM; timings vary a few ms between runs.

---

## 1. Summary

| Question | Answer (measured) |
|---|---|
| Does foveation cut memory? | Yes. Rendered map 4.70 MB vs 9.47 MB uniform (−50 %); stored cells 6.96 vs 8.75 MB (−20 %). |
| Is it faster? | Slightly. Map update + render PHASE5_MAP_FOV ms vs PHASE5_MAP_UNI ms median per scan. |
| Does it keep near-field detail? | Yes. Sharp curbs kept within 10 m: 50 % vs 42 % of all curb edges (uniform 20 cm). |
| What does it cost? | Far field (25–50 m, 50 cm cells): curbs flattened (8 % vs 19 % kept), ground error 0.096 vs 0.071 m median. |
| Does temporal fusion help? | Strongly. Near coverage 39 % → 96 %; curbs kept 13 % → 41 % (seq 00). Ghost area −55 % with clearing. |
| Does the hazard model help? | Yes over a threshold rule; but the model scores higher on the uniform map than on the foveated map. |
| Real time (10 Hz, 100 ms)? | PHASE5_RT_SUMMARY |

## 2. Problem

Dense 3D point clouds keep everything but are expensive to store and process. 2D maps are cheap but drop
height (curbs, steps, potholes). A uniform 2.5D grid keeps one height surface per cell, but spends the same
cell size everywhere. Foveation spends fine cells where detail matters (near the vehicle) and coarse cells
far away. The project measures whether that trade-off holds on real data, and whether the map still carries
enough information for a downstream hazard classifier.

## 3. Architecture

```
KITTI .bin / calib / poses / times
        │  src/io/kitti_loader.py          pure I/O
        ▼
preprocess (velodyne frame)                src/preprocess/filters.py, transforms.py
  range gate 2–50 m, ego box strip, NaN drop
  T_world_velo_i = Tr⁻¹ · P_i · Tr   (world := velodyne frame at scan 0)
        ▼
streaming foveated fusion (world frame)    src/temporal/fusion.py, terrain.py
  rings 0–10 m @ 10 cm, 10–25 m @ 20 cm, 25–50 m @ 50 cm (F-A, locked)
  per-scan cell stats → weighted fusion, decay, ghost clearing, hand-off between rings
        ▼
render: FoveatedLayers around sensor        src/mapping/foveated_grid.py
        ▼
hazard labeler (0–25 m reaction zone)      src/models/features.py, targets.py, labeler.py
  14 cell features → random forest → traversable / potentially hazardous / obstacle / unknown
  25–50 m: threshold rule
        ▼
visualisation + evaluation                 src/viz/grid_viz.py, scripts/*.py
```

Key design rules (from `AGENTS.md`):

- **Frames are explicit.** Velodyne, camera and world frames are never mixed silently; every grid carries a
  frame tag and rejects points from another frame.
- **Unobserved ≠ flat.** A cell with no points is `NaN` / count 0, never 0.0.
- **Sensor-frame ground is −1.73 m**, not 0. Obstacles are measured against local terrain, not a fixed plane.
- **Pure math kernels.** Binning, statistics and feature extraction take arrays and return arrays (58 unit tests).

### 3.1 Per-cell layers

count, z min, z max, z mean, z variance (roughness), ground (10th percentile of z), plus for the fused map:
decayed confidence weight, obstacle log-odds, last-seen time, and an `inherited` flag for placeholder cells.

### 3.2 Locked parameters (`src/config.py`)

| Parameter | Value | Why |
|---|---|---|
| Ground percentile | 10th | robust low envelope per cell |
| Range gate | 2–50 m | drops ego returns and sparse far points |
| Ego box (velodyne) | x −1.2..1.6, y −0.9..0.9, z −1.8..0 m | KITTI car body |
| Foveated bands (F-A) | 0–10 m @ 10 cm, 10–25 m @ 20 cm, 25–50 m @ 50 cm | best memory/accuracy trade in Phase 2 sweep |
| Decay time constant | 3 s; forget below weight 0.05 (~9 s unseen) | stale observation decay |
| Obstacle height | ≥ 30 cm over local terrain | obstacle evidence + target class |
| Log-odds | hit +0.85, miss −0.4, clamp [−2, 3.5] | OctoMap defaults |
| Local terrain | 2 m cells, p10, 3×3 min filter | reaches under cars |
| Hazard thresholds | step ≥ 8 cm / 40 cm, slope ≥ 15 %, roughness ≥ 8 cm | 5 cm roughness sat inside stacked-scan noise |

## 4. Phase results

### Phase 1 — Uniform baseline

- Loader, filters, frame transforms and a uniform 2.5D grid, verified on synthetic scenes (flat road + box,
  curb, ramp, empty patch).
- Real check on seq 00: near-field (< 15 m) cell ground median **−1.738 m** over 100 scans (expected ≈ −1.73 m).
- Pose chain check: same scan in velodyne vs world frame agrees within 2 cm.
- Dense reference map (100 scans stacked in world frame, 0.1–0.2 m cells): flat-cell ground agreement between
  single scans and the reference: median +2.5 cm, p95 9.8 cm — the poses are good enough to use stacked scans as truth.
- A single scan observes only **7.4 %** of 0.2 m cells within 50 m: beam gaps, the motivation for fusion.

### Phase 2 — Foveated grid (single scan, seq 00)

| Config | Cells | MB | vs U-20 | ms median |
|---|---|---|---|---|
| U-20 (uniform 20 cm) | 250,000 | 12.00 | 1.00× | 24.8 |
| U-10 (uniform 10 cm) | 1,000,000 | 48.00 | 4.00× | 53.0 |
| F-spec 5/20/50 cm | 192,468 | 9.24 | 0.77× | 35.9 |
| **F-A 10/20/50 cm** | 97,856 | **4.70** | **0.39×** | 29.4 |
| F-B 10/25/50 cm | 126,652 | 6.08 | 0.51× | 32.0 |

- Rings store only cells touching their annulus; each point goes to one ring by its own range.
- 5 cm near cells were too sparse for one scan (22 % coverage) → rejected.
- Curb test (8–30 cm steps from the reference, probes 20 cm apart, within 10 m): when both sides are observed,
  10 cm cells keep 52 % of curbs vs 30 % for 20 cm cells (step height kept 0.56 vs 0.11). But 10 cm cells are
  observed less often on one scan, so overall they kept fewer curbs (10 % vs 14 %) → temporal fusion needed.
- Obstacles: recall 100 % at every resolution (a coarse cell keeps the highest point); cost is area bloat.

### Phase 3 — Temporal fusion (seq 00, 100 scans)

- World-aligned lattice per band, so vehicle rotation never resamples a cell; each scan is pose-compensated
  into world frame.
- Weighted fusion (one scan = weight 1), exponential decay, pruning → stale cells return to unknown.
- Ghost clearing: obstacle log-odds per cell; when a scan sees the cell free and evidence drops to ≤ 0, the old
  z max is dropped.
- Hand-off: cells leaving a ring outward merge into the coarser lattice; coarse cells entering a finer ring split
  into flagged placeholder children, replaced by the first real fine observation.

| Seq 00, near ring 2–10 m | single scan | fused, no seeding | **fused + seeding** | fused uniform 20 cm |
|---|---|---|---|---|
| Ground coverage | 39 % | 85 % | **96 %** | 97 % |
| Curbs kept (all edges, 20 cm probes) | 13 % | 32 % | **41 %** | 30 % |
| Ground error p95 (m) | 0.49 | 0.32 | 0.41 | 0.42 |
| Ghost area (m², 10 scans) | — | 172 (386 without clearing) | 173 | 149 |

CPU: one-sort per-cell kernel and known bounding boxes took the fused update from 79 ms to 49 ms per scan.

### Phase 4 — Hazard model (train seq 00, test seq 08)

- **Features** (from the live fused map, what the vehicle has at that moment): mean / max / ground height over
  local terrain, roughness, relief, step across 40 cm, slope over 1 m, range, cell size, observation count,
  confidence, age, obstacle log-odds, placeholder flag.
- **Targets** (from the dense reference built from the whole slice, same thresholds): obstacle if top ≥ 30 cm over
  terrain; potentially hazardous if step ≥ 8 cm, slope ≥ 15 % or roughness ≥ 8 cm; unknown if the reference cell
  has < 5 points; else traversable. Using a denser, independent source for targets avoids the model simply
  re-learning the thresholds from its own inputs.
- **Split:** train seq 00 scans 0–99, test seq 08 scans 1500–1599. Model-size choices were made on a seq 00
  time-block hold-out (scans 10–199 vs 210–299).
- **Data issue found:** KITTI ground-truth poses for seq 08 frames 0–99 climb 16 m over 44 m of travel while the
  scans show flat ground. That window is unusable as truth; the test slice was moved to frames 1500+ and the training
  script now rejects slices whose poses climb more than 10 % of path length.

PHASE4_TABLE

### Phase 5 — Benchmark triad (train seq 00 0–299, test seq 08 1500–1799)

PHASE5_TABLES

## 5. Measured trade-offs

PHASE5_TRADEOFFS

## 6. Limitations

- **2.5D is one surface per cell.** Tree canopy or overhangs above the road count as obstacles; clearance under
  them is not represented.
- **Truth is geometric**, from thresholds on a stacked reference map — not human labels and not collision outcomes.
  The reference also contains observations from later scans, so some "misses" are things the live map could not
  have seen yet.
- **Steep ramps:** local terrain uses a 6 m window; slopes above ~15 % can look like obstacles.
- **Small data:** one training sequence, one test sequence, 31 s each. Semantic labels (SemanticKITTI) were not
  reachable from the development environment, so no moving-object ground truth.
- **Timing** measured in Python/NumPy on a shared VM; not representative of an embedded target.
- The prototype must not control a vehicle or make safety-critical decisions.

## 7. Reproduce

```bash
pip install -r requirements.txt && pytest
python scripts/fetch_kitti_slice.py --seq 00 --start 0 --count 300 --out data/kitti_long
python scripts/fetch_kitti_slice.py --seq 08 --start 1500 --count 300 --out data/kitti_long
python -m scripts.benchmark
```

Per-phase scripts and the short-slice commands are listed in `README.md`.
