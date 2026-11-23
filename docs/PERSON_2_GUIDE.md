# Person 2 guide — Systems, mapping, corridor and viewer lead

You own everything that builds the map, checks corridors and shows the result. This guide is standalone.

## 1. Project in one paragraph

KITTI LiDAR scans (~120k points, 10 Hz) become a **foveated 2.5D elevation map**, fused over time in a world frame:
- **Rings:** 10 cm cells to 10 m, 20 cm to 25 m, 50 cm to 50 m.
- **Fusion:** weighted, with 3 s decay, obstacle log-odds and ghost clearing.

Person 1's model labels cells as candidate hazards (traversable / potentially hazardous / obstacle / unknown). A **candidate corridor check** scores 15 driving arcs on those labels. A browser **replay viewer** shows it all scan by scan.

Research prototype: nothing here steers or controls a vehicle, and nothing predicts collisions (`AGENTS.md`).

## 2. Ownership

| You own | You read only (ask Person 1) |
|---|---|
| `src/mapping/`, `src/temporal/` (grid, fusion, hand-off, snapshots) | `src/models/` (features, targets, labeler, model) |
| `src/planning/corridor.py` | `scripts/train_hazard.py`, model files |
| `scripts/benchmark.py` (timing / map sections), `scripts/eval_seams.py`, `scripts/eval_corridor.py` | hazard P/R/F1 tables |
| `scripts/export_replay.py`, `scripts/export_snapshots.py`, `scripts/find_corridor_events.py` | |
| `viewer/` | |
| `docs/INTERFACES.md` (keep it true) | `src/config.py`: thresholds and bands are joint |

## 3. Latency budget (p95 per scan, seq 08 test, 100 ms total)

| Part | Budget | Last measured (map v4) |
|---|---|---|
| map integrate + render | **65 ms** | 49.7 median / 61.1 p95 |
| hazard labeler (Person 1) | 30 ms | 22.3 median / 27.7 p95 |
| corridor check | **5 ms** | 3.63 median / 4.46 p95 |

- Timing varies run to run: two runs of the same unchanged code differed by up to ~11 ms total p95.
- Compare against a **same-day baseline run** before claiming a change, using a git worktree of the old commit (see report).

## 4. Done so far (map v2 → v4)

| Item | Result |
|---|---|
| Ring seam fix (v2) | Observed share just past 10 m: 64.5 % → 93.7 %; past 25 m: 29.4 % → 59.6 %. Flat across edges. `scripts/eval_seams.py`. |
| `points` layer (v3) | Mean points per scan per cell, for Person 1's density feature. |
| Ego box x_max 1.6 → 2.8 m (v4) | Removes recording-car body points (x 2.5–2.6, y ±0.5, z −0.9 m; 93 % of scans) that left fake obstacles along the driven path. |
| Snapshots | `scripts/export_snapshots.py` + `src/temporal/snapshot.py` (exact round trip, tested). |
| Corridor check | 15 constant-curvature arcs, radius ≥ 10 m, 20 m long, 2.2 m footprint. Obstacle must be a cluster (≥ 4 probes within 1 m) **and** have map evidence (log-odds > 0). Unseen never counts as free. |
| Corridor benchmark | Seq 08 test: verdict agreement 85.2 %, strict false clear 0.4 %, driven arc blocked 1.9 %. `min_hits` = 4 chosen on seq 00 (train) only. |
| Viewer corridor layer | Arcs coloured by verdict, ✕ at block point, cyan picked arc with arrow and footprint band, dashed driven path, HUD box, event bookmarks (N / P keys). |

## 5. Remaining tasks, in order

### 5.1 Keep the corridor honest
- The reference for corridor scoring is a dense stack of scans i−5..i+5 (~1 s). A whole-slice stack smeared moving cars along the lane and blocked 83–97 % of arcs. Keep the window short.
- **Sanity row:** "reference blocks driven arc" must stay ~0 % on test. It was 0.0 % on seq 08 and 22.4 % on seq 00, where the reference is noisier. If it rises, fix the reference before trusting any corridor number.
- **Known weakness:** 30.8 % of reference-blocked arcs come out CAUTION live, not BLOCKED. That is a missed block, but not a false clear. Look at those cases in the viewer before tuning further.

### 5.2 Adaptive refinement (core spec item, not built yet)
- `AGENTS.md`: ring bands are only the base layer; refinement must trigger on height variance, sharp steps, or hazard cues.
- Plan: in the 25–50 m ring, a 50 cm cell with step ≥ 8 cm, high `z_var`, or log-odds > 0 is refined to 20 cm for a local patch.
- The same rule as hand-off applies: copy when missing, never double count, keep history on refine and un-refine.
- **Target metric:** curbs kept 25–50 m (foveated 8 % vs uniform 19 % in Phase 5).
- **Also report:** memory, map ms p95 (≤ 65 ms), ghost area, hazard F1 change. Bump `MAP_VERSION`.

### 5.3 Map latency headroom
- Hand-off and render are still Python-heavy.
- **Profile first** (`cProfile` on `TemporalFoveatedMap.integrate` / `render`).
- Candidates:
  - fewer `np.unique` / `searchsorted` passes in `add_missing` and `merge_children`
  - cache annulus masks per lattice offset

### 5.4 Viewer v2 (after numbers are final)
- Split screen, uniform vs foveated, synced scrub. About doubles replay size (~250 MB).
- Optional: corridor tooltip showing per-arc unknown / hazard share on hover.

### 5.5 Every change
- `pytest` green.
- `python -m scripts.eval_seams` still flat.
- `python -m scripts.benchmark` timing table in the commit message.
- Bump `MAP_VERSION` when `render()` output changes, and update `docs/INTERFACES.md`.
- Re-export snapshots for Person 1.

## 6. How the pieces run

```bash
python -m scripts.benchmark                     # triad: uniform / foveated / foveated + model (train 00, test 08)
python -m scripts.eval_seams                    # observed share around ring edges
python -m scripts.eval_corridor                 # corridor vs short-window reference, seq 08 test
python -m scripts.eval_corridor --seq data/kitti_long/sequences/00 --tune   # min_hits sweep, TRAIN only
python -m scripts.find_corridor_events          # list frames where straight is blocked and another arc is a candidate
python -m scripts.export_replay                 # replay + corridor for viewer (~124 MB)
python -m http.server 8000 --bind 127.0.0.1     # open http://127.0.0.1:8000/viewer/
```

## 7. Rules that bite (`AGENTS.md`)

- **Frames:** arcs live in the velodyne frame and move to world only through `T_world_velo` in `probes_to_world`. The viewer flips y in one function only (`worldToScreen`).
- **Unobserved = NaN / unknown, never 0.0.** Unseen is never free in the corridor check.
- **Wording:**
  - Say "candidate corridor check".
  - Never "obstacle avoidance", "steering", "path planning" or "collision avoidance".
  - The KITTI car really drove; nothing was steered.
- **Train 00 / tune 05 / test 08:** never tune a corridor or map parameter on seq 08.
- **Commits:** caveman style, human author, no tool attribution. Branch first, merge to `main` on request.
