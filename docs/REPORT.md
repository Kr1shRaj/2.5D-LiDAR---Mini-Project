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
| Is it faster? | Slightly. Map update + render 52.3 ms vs 54.8 ms median per scan. |
| Does it keep near-field detail? | Yes. Sharp curbs kept within 10 m: 50 % vs 42 % of all curb edges (uniform 20 cm). |
| What does it cost? | Far field (25–50 m, 50 cm cells): curbs flattened (8 % vs 19 % kept), ground error 0.096 vs 0.071 m median. |
| Does temporal fusion help? | Strongly. Near coverage 39 % → 96 %; curbs kept 13 % → 41 % (seq 00). Ghost area −55 % with clearing. |
| Does the hazard model help? | Yes over a threshold rule; but the model scores higher on the uniform map than on the foveated map. |
| Real time (10 Hz, 100 ms)? | Median yes: map + model labeling 77 ms (p95 99.6 ms). 5 % of scans exceed 100 ms. The model only labels 0–25 m. |

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
  into flagged placeholder children, replaced by the first real fine observation. (Map v1 rule. Map v2 below
  changes the edge handling.)

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

Test on seq 08, 540,000 sampled spots (random forest, 8 trees, depth 12):

| Class | Model P / R / F1 | Rule on same map P / R / F1 | Support |
|---|---|---|---|
| traversable | 0.53 / 0.46 / 0.49 | 0.39 / 0.74 / 0.51 | 98,929 |
| potentially hazardous | 0.17 / 0.38 / 0.23 | 0.09 / 0.34 / 0.14 | 30,237 |
| obstacle | 0.64 / 0.60 / 0.62 | 0.54 / 0.51 / 0.52 | 161,389 |
| unknown | 0.73 / 0.68 / 0.71 | 0.54 / 0.18 / 0.27 | 249,445 |
| **macro F1** | **0.513** | 0.361 | |
| true obstacle called traversable | **7.5 %** | 18.9 % | |

Model confusion matrix (rows = truth, columns = predicted):

| truth \ pred | traversable | hazardous | obstacle | unknown |
|---|---|---|---|---|
| traversable | 45,442 | 27,258 | 8,553 | 17,676 |
| potentially hazardous | 8,869 | 11,463 | 4,687 | 5,218 |
| obstacle | 12,044 | 11,886 | 97,301 | 40,158 |
| unknown | 18,920 | 18,411 | 41,409 | 170,705 |

- Macro F1 by ring (model / rule): 2–10 m 0.570 / 0.489, 10–25 m 0.498 / 0.429, 25–50 m 0.450 / 0.289.
- The hazardous class is the weakest (F1 0.23): it is rare (5.6 % of test spots) and partly invisible in a sparse
  live map. Raising the roughness threshold from 5 to 8 cm removed noise stripes on flat road (truth traversable
  14 % → 18 %) but did not raise macro F1 (0.526 → 0.519 with the earlier boosted-tree model).
- Model choice (seq 00 hold-out only): gradient boosted trees 300 rounds 0.537, 15 rounds 0.532, random forest
  8 trees 0.528, single tree 0.517. The forest was kept because boosted trees paid ~1 ms per tree in the live
  loop (thread wake-up) and could not meet the 100 ms budget.

### Phase 5 — Benchmark triad (train seq 00 0–299, test seq 08 1500–1799)

Same test slice for every configuration. Configs 1–2 label hazards with the threshold rule; config 3 runs the
random forest on the 0–25 m reaction zone through an incremental labeler (re-labels a cell only when it is new,
its obstacle state flips, its top moves > 10 cm, or its label is 0.3 s old) and the rule beyond 25 m.

**Cost per scan** (all 300 scans, preprocessing excluded):

| Config | Map ms med / p95 | Label ms med / p95 | Cells labelled / scan | **Total ms med / p95** | Scans > 100 ms | Stored MB | Rendered MB |
|---|---|---|---|---|---|---|---|
| 1 uniform (20 cm) | 54.8 / 76.6 | – | – | 54.8 / 76.6 | 0.0 % | 8.75 | 9.47 |
| 2 foveated (F-A) | 52.3 / 69.0 | – | – | 52.3 / 69.0 | 0.3 % | 6.96 | 4.70 |
| 3 foveated + model | 52.3 / 69.0 | 25.0 / 34.8 | 13,141 | **77.4 / 99.6** | 5.0 % | 6.96 | 4.70 |
| (extra) uniform + model | 54.8 / 76.6 | 19.0 / 29.0 | 7,530 | 74.2 / 104.4 | 7.4 % | 8.75 | 9.47 |

**Height reconstruction vs dense reference** (ground, metres):

| Map | Ring | Coverage | Error median | Error p95 |
|---|---|---|---|---|
| uniform | 2–10 m | 98 % | 0.029 | 0.457 |
| foveated | 2–10 m | 98 % | 0.034 | 0.407 |
| uniform | 10–25 m | 85 % | 0.052 | 0.809 |
| foveated | 10–25 m | 89 % | 0.055 | 0.829 |
| uniform | 25–50 m | 63 % | 0.071 | 1.398 |
| foveated | 25–50 m | 73 % | 0.096 | 1.550 |

**Preservation** (curbs: share of all reference curb edges kept, probes 20 cm apart):

| Map | Curbs 2–10 m | Curbs 10–25 m | Curbs 25–50 m | Obstacle recall | Area bloat | Ghost area |
|---|---|---|---|---|---|---|
| uniform | 42 % | 23 % | 19 % | 100 % | 1.51× | 1,165 m² |
| foveated | **50 %** | 24 % | 8 % | 100 % | 1.47× | 1,386 m² |

**Hazard labels** (P / R / F1):

| Config | Macro F1 | Traversable | Hazardous | Obstacle | Unknown | Obstacle → traversable |
|---|---|---|---|---|---|---|
| 1 uniform, rule | 0.385 | 0.41 / 0.62 / 0.49 | 0.10 / 0.34 / 0.16 | 0.65 / 0.46 / 0.54 | 0.43 / 0.31 / 0.36 | 17.1 % |
| 2 foveated, rule | 0.346 | 0.38 / 0.65 / 0.48 | 0.08 / 0.33 / 0.13 | 0.55 / 0.50 / 0.52 | 0.44 / 0.18 / 0.25 | 19.0 % |
| 3 foveated + model (0–25 m) | 0.358 | 0.38 / 0.58 / 0.46 | 0.09 / 0.38 / 0.14 | 0.56 / 0.54 / 0.55 | 0.45 / 0.20 / 0.28 | 14.8 % |
| (extra) uniform + model (0–25 m) | 0.406 | 0.43 / 0.57 / 0.49 | 0.12 / 0.37 / 0.18 | 0.67 / 0.48 / 0.56 | 0.44 / 0.37 / 0.40 | 13.7 % |
| (offline) foveated + model, all rings | 0.476 | 0.51 / 0.47 / 0.49 | 0.13 / 0.44 / 0.20 | 0.67 / 0.51 / 0.58 | 0.64 / 0.63 / 0.63 | 10.7 % |

**Hazard by ring** (macro F1 / true obstacle called traversable):

| Config | 2–25 m (reaction zone) | 2–10 m | 10–25 m | 25–50 m |
|---|---|---|---|---|
| 1 uniform, rule | 0.437 / 23.5 % | 0.448 / 33.4 % | 0.424 / 20.9 % | 0.333 / 12.3 % |
| 2 foveated, rule | 0.419 / 25.9 % | 0.442 / 42.1 % | 0.406 / 21.8 % | 0.292 / 13.7 % |
| **3 foveated + model** | **0.472 / 16.0 %** | 0.502 / 28.8 % | 0.455 / 12.7 % | 0.292 / 13.7 % |
| (extra) uniform + model | 0.518 / 15.5 % | 0.532 / 26.1 % | 0.500 / 12.8 % | 0.334 / 12.3 % |

The whole-slice macro F1 of config 3 (0.358) is pulled down by the 25–50 m ring, which holds 62 % of the
sampled spots and still uses the rule; inside the reaction zone the model lifts macro F1 from 0.419 to 0.472
and cuts obstacle-called-traversable from 25.9 % to 16.0 %.

### Map v2 — ring seam fix (after Phase 5)

The replay viewer showed thin empty arcs just outside the 10 m and 25 m ring edges.

**Cause:**
- Lookup picks the ring by the query point's own range.
- Map v1 moved a coarse cell into the finer ring as soon as its *centre* crossed the edge.
- So the outer part of an edge cell lost its history.
- Fine cells just past the edge also kept theirs in the fine store, which lookup never reads there.

**Map v2 (`MAP_VERSION = 2`):**
- A cell whose square crosses a ring edge is kept in **both** rings:
  - A coarse cell reaching into a finer ring seeds missing fine children, and leaves only when its whole square is inside.
  - A fine cell reaching past its ring builds its missing coarse parent, and leaves only when its whole square is outside.
- Copies are made only when missing, so no observation is counted twice (tested: cell weight never exceeds the one-ring steady state).
- Each ring also takes points up to one own-cell diagonal past its edges, for cells touching that ring. Edge cells then see their whole square, not just their own side.

Observed share of map lookups by offset from the ring edge (seq 08, 29 scans, `scripts/eval_seams.py`):

| Edge | Map | −0.2..−0.1 m | −0.1..0 m | 0..+0.1 m | +0.1..+0.2 m | +0.3..+0.4 m |
|---|---|---|---|---|---|---|
| 10 m | v1 | 93.3 % | 84.8 % | **64.5 %** | 84.3 % | 93.1 % |
| 10 m | v2 | 93.6 % | 93.6 % | **93.7 %** | 93.5 % | 93.2 % |
| 25 m | v1 | 54.6 % | 50.5 % | **29.4 %** | 37.1 % | 49.1 % |
| 25 m | v2 | 59.2 % | 59.4 % | **59.6 %** | 59.9 % | 59.3 % |

Benchmark, both maps run on the same day (timing varies between days; Phase 5 numbers were another day):

| Measure (seq 08 test) | Map v1 (same-day rerun) | Map v2 |
|---|---|---|
| Foveated map ms, median / p95 | 49.2 / 65.3 | 48.6 / 59.9 |
| Foveated + model total ms, median / p95 | 72.9 / 95.4 | 70.9 / 87.1 |
| Scans over 100 ms (foveated + model) | 1.0 % | 0.0 % |
| 0–25 m macro F1: foveated + model / uniform + model | 0.472 / 0.518 | 0.478 / 0.513 |
| 0–25 m obstacle called traversable: foveated + model | 16.0 % | 15.4 % |
| Curbs kept 2–10 / 10–25 / 25–50 m (foveated) | 50 / 24 / 8 % | 50 / 24 / 8 % |
| Ghost area (foveated) | 1386 m² | 1372 m² |

- The v1 rerun reproduces the Phase 5 hazard numbers exactly (deterministic pipeline).
- The fix narrows the model gap from 0.046 to 0.035. It does not close it.
- Map p95 is within the 65 ms map budget on this run.

### Map v3 / v4 — points layer, ego box fix, candidate corridor check

**Map v3** adds a `points` layer: the mean number of points per scan in each cell, for the model's density feature.

**Map v4 widens the ego box from x_max 1.6 m to 2.8 m.**
- Fixed sensor-frame points at x 2.5–2.6 m, y ±0.5 m, z −0.9 m appear in 93–94 % of scans on both sequences. They are part of the recording car.
- They planted an "obstacle" 2.5 m ahead in every scan. The car then drove over that spot, so no later scan ever saw it free and cleared it.
- Result: a trail of fake obstacle cells along the driven path.
- The lowest beam first reaches the road ~3.7 m away, so the wider box removes nothing real.

Benchmark on map v4 (same scripts as Phase 5; seq 08 test):

| Measure | Map v2 | Map v4 |
|---|---|---|
| Foveated map ms, median / p95 | 48.6 / 59.9 | 49.7 / 61.1 |
| Labeler ms, median / p95 | 22.1 / 28.0 | 22.3 / 27.7 |
| Foveated + model total ms, median / p95 | 70.9 / 87.1 | 72.3 / 85.7 |
| Stored MB max (foveated) | 6.96 | 7.64 (`points` layer) |
| 0–25 m macro F1: foveated + model / uniform + model | 0.478 / 0.513 | 0.484 / 0.509 |
| 0–25 m obstacle called traversable (foveated + model) | 15.4 % | 15.6 % |

- Curbs kept, obstacle recall and ground error are unchanged to the reported precision.
- The seam check is still flat at both edges.
- The hazard benchmark's sampled truth hardly moves, because it rarely samples the narrow driven strip. The corridor check below is where the fix shows.

**Candidate corridor check (`src/planning/corridor.py`).**
- **What it is:** a research check of which driving arcs the map shows as blocked, caution or clear. It steers nothing and predicts no collisions: the KITTI car really drove.
- **Arcs:**
  - 15 constant-curvature arcs, curvature −0.1 … +0.1 m⁻¹ (radius ≥ 10 m), 20 m long.
  - Car footprint 2.2 m wide, probes every 0.2 m along and ~0.18 m across (~20k probes).
  - Arcs are built in the velodyne frame and moved to world with the scan pose.
- **Blocking rule:** an arc is blocked when an obstacle cluster (≥ 4 obstacle probes within 1 m of arc) lies closer than 8 m. An obstacle probe counts only if the model label is obstacle **and** the map's own obstacle evidence agrees (log-odds > 0).
- **Unseen spots are never free:** an arc more than 30 % unseen is never picked.

Why map evidence is required: model obstacle labels alone over-fire on flat road.
- On the path the car actually drove, 80 % of model obstacle probes had height over terrain below 0.30 m (median 0.25 m), negative log-odds and ~1 cm roughness.
- With labels alone, the driven arc came out blocked in 62.3 % of test scans.

**Truth for scoring:**
- A dense 0.1 m stack of scans i−5 … i+5 (~1 s) around each scored scan, with the same thresholds.
- A whole-slice stack smears passing cars along the lane: it blocked 83–97 % of arcs, so it was rejected.
- **Sanity check:** the reference blocks the driven arc in 0.0 % of seq 08 test scans.
- `min_hits` = 4 was chosen on seq 00 (train) only: best verdict agreement (90.9 %) in the sweep.

Seq 08 test, 58 scored scans × 15 arcs (`scripts/eval_corridor.py`):

| Config | Verdict agree | Blocked yes/no agree | Reference-blocked arcs missed | False clear (live CLEAR, ref BLOCKED) | Picked arc ref-blocked | Driven arc blocked |
|---|---|---|---|---|---|---|
| foveated + model + evidence | **85.2 %** | 88.3 % | 30.8 % | **0.4 %** | 3.4 % | **1.9 %** |
| foveated + model, label only | 54.0 % | 55.2 % | 0.0 % | 0.0 % | 0.0 % | 62.3 % |
| foveated + rule + evidence | 80.6 % | 88.0 % | 33.1 % | 0.4 % | 3.4 % | 1.9 % |
| uniform + model + evidence | 85.9 % | 89.0 % | 29.7 % | 0.4 % | 3.4 % | 0.0 % |

- **Corridor cost:** 3.63 ms median, 4.46 ms p95 (budget 5 ms).
- **Missed blocks:** reference-blocked arcs that come out CAUTION live (30.8 %) are missed blocks. They are not false clears: the arc is still flagged.
- The replay viewer draws arcs, the picked candidate, block points and the driven path, with bookmarks for the 25 frames where straight is blocked and another arc is a candidate.

### Map v5 — adaptive refinement + faster hand-off

`AGENTS.md`: ring bands are only the base layer; resolution must refine on hazard cues.

**How refinement works:**
- In the 25–50 m ring (50 cm base cells), a cell with a hazard cue opens a **3 × 3 patch of 20 cm cells**.
- **Cues:**
  - ground step ≥ 8 cm to a 4-neighbour cell
  - z std 8–30 cm inside the cell (rough ground; walls have metre-scale std and are excluded)
  - obstacle evidence (log-odds > 0)
- **Guards:**
  - A cue counts only from cells seen in ≥ 3 recent scans and touched this scan. Far ground noise (median error 0.12 m) otherwise fired the 8 cm step cue on ~40 % of the ring.
  - **Budget:** at most 3000 refined base cells, ranked steps first, then roughness, then obstacle. Without the cap, seq 08 asked for ~6,500–7,300 cells and map p95 rose to 73–81 ms.
  - Cue update runs every 2nd scan; the 1 s hold covers the gap. Points feed the patches every scan.

**History rules (no loss, no double counting):**
- The patch is a second layer over the base ring. The 50 cm base keeps all its own data.
- Fine cells are seeded from their parent (flagged inherited), then replaced by real 20 cm data.
- Fine cells cross the 25 m edge as copies only when missing, both ways, onto the same 20 cm lattice.
- An expired patch drops only its fine cells.
- Lookup serves the fine cell where it is observed, else the base cell.
- Tested: weights never exceed the one-layer bound, expiry leaves the base untouched, and with refinement off the output is bit-identical to map v4.

**Tuning (seq 00, train only, `scripts/eval_refine.py`):**

| Variant | Curbs kept 25–50 m | Map ms median / p95 | Store MB |
|---|---|---|---|
| Uniform 20 cm | 19 % | 33.4 / 41.4 | 6.04 |
| Foveated, no refine | 7 % | 31.3 / 37.1 | 5.53 |
| Refine, no budget, cues every scan | 13 % | 39.5 / 51.6 | 7.80 |
| **Refine, map v5 config** | **10 %** | 40.1 / 47.9 | 7.08 |

**Final, seq 08 test.** Map v4 code and map v5 code were run back to back on the same day (`outputs/phase5_benchmark_mapv4_sameday.txt`, `..._mapv5.txt`):

| Measure | Map v4 | Map v5 |
|---|---|---|
| Foveated map ms, median / p95 | 45.5 / 57.3 | 50.0 / 59.2 |
| Uniform map ms, median / p95 | 51.0 / 64.5 | 41.7 / 50.7 |
| Labeler ms, median / p95 | 20.6 / 25.2 | 24.0 / 29.0 |
| Foveated + model total, median / p95 | 66.2 / 81.7 | 74.3 / 86.2 |
| Store MB max (foveated / uniform) | 7.64 / 9.63 | 9.36 / 9.63 |
| Rendered MB (foveated) | 4.70 | 5.48 |
| **Curbs kept 25–50 m (foveated)** | **8 %** | **11 %** (uniform 19 %) |
| Curbs kept 2–10 / 10–25 m (foveated) | 50 / 24 % | 50 / 24 % |
| Ground cover 25–50 m / err p95 | 73 % / 1.561 m | 74 % / 1.664 m |
| Ghost area (foveated) | 1371 m² | 1137 m² |
| Area bloat (foveated) | 1.47× | 1.44× |
| 0–25 m macro F1, foveated + model / uniform + model | 0.484 / 0.509 | 0.481 / 0.509 |

- **Refinement closes about a quarter of the far-curb gap** (8 → 11 % vs 19 % uniform). The budget, needed for latency, holds it below the 13 % seen uncapped on seq 00.
- **Map p95 stays under the 65 ms budget.** The hand-off speed-ups pay for most of the refinement cost:
  - sorted-merge `add_missing` and `integrate_obs` (no `np.isin` / `union1d` / `argsort`)
  - no copy when nothing is dropped
  - squared distances
  - a single gather index in render
- The uniform map gained most from these speed-ups (−10 / −14 ms), since it has one large store.
- **Labeler p95 is 29.0 ms against its 30 ms budget.** Part of that is lookups visiting the patch layer, part is noise; it needs watching.
- **Corridor unchanged on map v5** (seq 08): verdict agreement 85.6 %, false clear 0.4 %, driven arc blocked 1.9 %, 3.69 / 4.55 ms.
- **Seams flat:** 93.2–93.7 % at 10 m, 59.1–60.1 % at 25 m.

## 5. Measured trade-offs

1. **Memory: foveation wins.** Half the rendered map (4.70 vs 9.47 MB) and 20 % less stored state, with
   similar map latency (52 vs 55 ms).
2. **Near field: foveation wins.** 10 cm cells keep more sharp curbs within 10 m (50 % vs 42 %) and a lower
   near-field ground p95 error (0.41 vs 0.46 m). It needs temporal fusion and coarse-to-fine seeding to pay
   off: on single scans, 10 cm cells were too sparse.
3. **Far field: foveation loses accuracy, gains coverage.** 50 cm cells see more of the far ring (73 % vs 63 %)
   but flatten curbs (8 % vs 19 % kept) and raise ground error (0.096 vs 0.071 m median).
4. **Obstacles are never lost** (100 % recall in every map) — a coarse cell keeps the highest point; the price is
   area bloat (~1.5×). Ghost trails are cut 55 % by clearing (seq 00) but foveated maps still show 19 % more
   ghost area than uniform on seq 08.
5. **Learned labels beat thresholds** on the same map (macro F1 +0.15, obstacle-called-traversable 18.9 % →
   7.5 % in Phase 4). **But the model is better on the uniform map** (0–25 m macro F1 0.518 vs 0.472): coarse
   and placeholder cells make features noisier. Foveation helps the map more than it helps the classifier.
6. **Real time is reached at the median, not with margin.** Map + incremental labeler runs at 77 ms median,
   99.6 ms p95; 5 % of scans exceed 100 ms. Getting there required a sparse one-sort cell kernel, O(1) cell
   lookup, batched feature probes, model labels only in the 0–25 m zone, event-driven re-labeling and a
   thread-free random forest instead of boosted trees (−0.004 hold-out F1).

### Next steps

- Close the foveated vs uniform gap for the classifier (0.025 on map v4) and the model's over-eager obstacle labels on flat road: diagnose by cell size and
  inherited flag first, then neighbour context features. Tune on a separate validation sequence (seq 05).
- Far curbs: refinement reaches 11 % (uniform 19 %). More needs a bigger budget, i.e. a faster base pipeline (compiled kernels).
- Cut p95 latency (render and hand-off still ~20 ms; a compiled implementation would remove most Python overhead).
- Validate on more sequences and with SemanticKITTI labels for moving objects.

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
