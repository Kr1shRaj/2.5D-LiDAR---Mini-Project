# Person 1 guide — Model / ML lead

You own the hazard model: features, targets, training, the live labeler. This guide is standalone: read it, set up, start working.

## 1. Project in one paragraph

A LiDAR on a car's roof gives ~120,000 3D points per scan, 10 scans per second (KITTI odometry dataset). We turn each scan into a **2.5D elevation map**: a grid of ground cells, each holding a few heights and statistics. Cells are **foveated**: 10 cm cells within 10 m of the car, 20 cm cells to 25 m, 50 cm cells to 50 m. The map is **fused over time** in a world frame, with confidence decay (τ = 3 s), obstacle log-odds and ghost-trail clearing. Your model then labels each cell with a **candidate hazard class**: traversable, potentially hazardous, obstacle, unknown. A corridor check uses those labels to score candidate driving arcs. This is a research prototype: labels are candidate hazard labels, **never** collision or accident predictions (`AGENTS.md`).

## 2. Current state

- **Map version 5** (`src/config.MAP_VERSION`):
  - v2 fixed data gaps at ring edges.
  - v3 added the `points` layer.
  - v4 widened the ego box to x_max 2.8 m. Points from the recording car's own body (x 2.5–2.6 m, y ±0.5 m, z −0.9 m, in 93 % of scans) used to leave a trail of fake obstacles along the driven path.
  - v5 adds adaptive refinement: 20 cm hazard patches in the 25–50 m ring as an **overlay band**. Lookups there return 0.2 m resolution where a patch cell exists. Your labeler p95 rose to 29.0 ms (budget 30), partly from lookups visiting this band.
- **Latest benchmark numbers:** `docs/REPORT.md`, sections "Map v4" and "Map v5".
- **Your two problems:**
  1. **Accuracy gap.** 0–25 m macro F1 on map v5 is **0.481 foveated + model vs 0.509 uniform + model** (gap 0.028; v4 0.484, v1 gap 0.046).
     - Obstacle called traversable: 15.0 % foveated + model, 15.6 % uniform + model.
     - Labeler p95 is 29.0 ms against your 30 ms budget (map v5), so you have ~1 ms of headroom: any new feature must be paid for.
  2. **Over-eager obstacles on flat road.** On the path the car actually drove, 80 % of the model's "obstacle" cells look like plain road to every other signal:
     - height over terrain median 0.25 m, below the 0.30 m line
     - map obstacle evidence (log-odds) negative
     - roughness ~1 cm

     Using model labels alone, the corridor check marked the driven arc blocked in **62.3 %** of test scans. Person 2 now requires map evidence (log-odds > 0) before an obstacle label blocks an arc, which brings it to **1.9 %**. That is a safety net, not a fix: the labels themselves are still wrong.

## 3. Ownership

| You own | You read only (ask Person 2 for changes) |
|---|---|
| `src/models/` (`features.py`, `targets.py`, `labeler.py`) | `src/mapping/`, `src/temporal/` (map, layers, fusion, snapshots) |
| `scripts/train_hazard.py` | `src/planning/` (corridor), `viewer/`, `scripts/export_replay.py` |
| hazard section of `scripts/benchmark.py` (P/R/F1 tables) | `scripts/benchmark.py` timing section |
| model files `outputs/**/model_*.joblib` | `src/config.py`: thresholds and bands are joint decisions |

## 4. Interface contract (from `docs/INTERFACES.md`; do not break)

- **Map → you:**
  - `FoveatedLayers.sample_many(xy, names)` returns float arrays: NaN for unobserved spots, never 0.0.
  - Layers: `count`, `z_min`, `z_max`, `z_mean`, `z_var`, `ground`, `weight`, `logodds`, `last_seen`, `inherited`, `points`.
  - `count` = scans that saw the cell. `points` = mean points per scan, so density per m² = `points / resolution**2`.
- **You → everyone:**
  - `HazardLabeler(model, zone_m).update(fov, terrain, t) -> int`
  - `HazardLabeler.labels_at(fov, xy) -> int64`, with −1 = no model label.
  - Class ids fixed: 0 traversable, 1 potentially hazardous, 2 obstacle, 3 unknown.
- **Model file** (joblib dict) keys: `model`, `features`, `classes`, `version`, `map_version`, `train_slice`, `val_slice`.
  - The loader must refuse a mismatched `features` list or `map_version`. (Adding this check is your first small task.)
- **Latency:** labeler update **≤ 30 ms p95** per scan on test seq 08. Map is 65 ms and corridor 5 ms of the 100 ms total.
- **Data split:**
  - Train: seq 00 scans 0–299.
  - Validation, for all tuning: seq 05 slice.
  - Test: seq 08 scans 1500–1799. Final numbers only. Never tune on it.

## 5. Step by step

### Step 0 — setup (day 1)

```bash
git checkout main && git pull
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # all tests must pass
python scripts/fetch_kitti_slice.py --seq 00 --start 0 --count 300 --out data/kitti_long
python scripts/fetch_kitti_slice.py --seq 08 --start 1500 --count 300 --out data/kitti_long
python scripts/fetch_kitti_slice.py --seq 05 --start 0 --count 300 --out data/kitti_long   # validation
```

- `load_sequence` refuses a slice whose ground-truth poses climb more than 10 % of the path; seq 08 frames 0–99 fail this. If the seq 05 slice fails, pick another start (e.g. 1000) and note it.
- Re-run the baseline and keep the output as your reference: `python -m scripts.benchmark > outputs/p1_baseline.txt`.

### Step 1 — snapshots (frozen current map version)

```bash
python -m scripts.export_snapshots --seq data/kitti_long/sequences/00   # writes outputs/snapshots/mapv5/...
python -m scripts.export_snapshots --seq data/kitti_long/sequences/05
python -m scripts.export_snapshots --seq data/kitti_long/sequences/08
```

- Writes `outputs/snapshots/mapv<N>/<seq>_<first>-<last>/<scan>.npz`, every 5 scans from scan 10. About 150–200 MB per slice.
- `from src.temporal.snapshot import load_snapshot` gives `(fov, terrain, meta)`. Lookups match the live map exactly.
- Feature experiments then run without re-running fusion. When Person 2 bumps `MAP_VERSION`, re-export.

### Step 2 — diagnose the gap (before any new feature)

Split the test rows, using the same targets as `train_hazard.build_rows`. Report macro F1 and "obstacle called traversable" per group:

- cell size: 10 cm vs 20 cm (foveated) vs 20 cm (uniform)
- inherited yes / no
- within 30 cm of a ring edge (10 m, 25 m)
- `points` bins (0–2, 2–5, 5–10, 10+ per scan)
- range: 2–10 m, 10–25 m

Also a separate **flat-road obstacle table**: spots where the model says obstacle but the target says traversable.
- Histograms of `rel_max`, `logodds`, `roughness`, `count`, `points`, `inherited`.
- Expect most of them near `rel_max` 0.2–0.3 m with negative log-odds.

Output: `outputs/p1_diagnosis.txt` + one paragraph conclusion. Pick features from this, not from guesses.

### Step 3 — features, cheapest first, each measured alone

Add to `FEATURES` in `src/models/features.py` one at a time. Record validation (seq 05) macro F1, obstacle precision, and labeler ms for each.

1. **Point density:** `points / resolution**2`. Free; layer already exists.
2. **Obstacle evidence as a gate, not only an input:** `logodds` is already a feature. Test a post-rule "obstacle needs log-odds > 0, else hazardous", and compare with letting the model learn it.
3. **Terrain bias:** `rel_max` is measured against a 6 m minimum-filter terrain, which sits ~0.2 m under sloped road. Try a second reference: the cell's own `ground` vs a 3×3 neighbour median of `ground`. Ask Person 2 first if it needs a map layer.
4. **Scale-pooled roughness:** for 10 cm cells, pool roughness over the surrounding 20 cm block, so both maps measure at the same metric scale.
5. **Inherited handling:** low-trust flag, or rule labels until the first real fine scan.
6. **Neighbour pooling (3×3 max / mean of height and roughness):** biggest likely gain and biggest cost. Keep only if p95 ≤ 30 ms.

### Step 4 — tune the random forest (validation only)

- Grid: trees 4–16, depth 8–14, `min_samples_leaf` 20–100, `class_weight` balanced vs balanced_subsample vs a manual obstacle weight.
- **Class weighting is a suspect for problem 2:** balanced weights reward obstacle recall at the cost of precision.
- **Per-class probability thresholds:** raise the obstacle threshold until obstacle precision goes up without raising "obstacle called traversable" much. Report the curve.
- Keep `n_jobs=1` (thread wake-up cost broke the latency budget with boosted trees before), `random_state=0`.

### Step 5 — final test (once)

- Freeze settings. Train on seq 00. Run `python -m scripts.benchmark` and `python -m scripts.eval_corridor` on seq 08, once.
- Report:
  - per-class precision / recall / F1 and confusion matrix
  - macro F1 0–25 m, foveated vs uniform
  - obstacle called traversable
  - labeler ms median / p95
  - corridor "driven arc blocked" **with model labels only** (target: far below 62.3 %)
- Update the model section of `docs/REPORT.md`. Save the model with full metadata.

### Step 6 — hand-off

- New model file path to Person 2. The replay export picks it up with `--model <path>`.
- Tests:
  - feature shapes
  - NaN never turned into 0
  - loader refuses mismatched files
  - seeded tiny training is reproducible

## 6. Rules that bite (`AGENTS.md`)

- **No leakage:** train 00, tune 05, test 08. Never shuffle frames across splits.
- **Measure, do not assume:** every claim gets a number from a script in `scripts/`.
- **Seeds:** numpy, sklearn `random_state`.
- **Wording:** "candidate hazard labels", never "predicts collisions".
- **Commits:** caveman style, human author, no tool attribution.
- **When unsure** about frames, thresholds or calibration: stop and ask.
