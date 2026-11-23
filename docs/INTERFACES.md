# Team interfaces

Two roles, one boundary. Each side may change its own internals freely as long as these contracts hold.

| Area | Owner | Other side |
|---|---|---|
| Fused map, layer names, `render()` output, hand-off (`src/mapping/`, `src/temporal/`) | Person 2 (systems) | read only, request new layers |
| Snapshots (`src/temporal/snapshot.py`, `scripts/export_snapshots.py`) | Person 2 | read only |
| Corridor check (`src/planning/`), replay export, `viewer/` | Person 2 | read only |
| Features, targets, model, `HazardLabeler` internals (`src/models/`, `scripts/train_hazard.py`) | Person 1 (model) | read only |
| `src/config.py` thresholds and bands | joint: change only with both agreeing | |

## 1. Map → model

- `FoveatedLayers.sample_many(xy, names)`: world-frame xy (metres, KITTI velodyne frame at sequence scan 0).
  - Returns one float array per name.
  - NaN where the spot is unobserved or outside the rings. Never 0.0 for unobserved.
- Band choice at a spot = the ring holding that spot's range from the sensor (`FoveatedLayers.lookup`).
- Layer names:
  - core: `count`, `z_min`, `z_max`, `z_mean`, `z_var`, `ground`
  - extra: `weight`, `logodds`, `last_seen`, `inherited`
  - plus `resolution` through the feature sampler
- **Layers are only added, never renamed or removed.** New layer = new map version.
- `count` in the fused map = scans that saw the cell, not points.
- `TemporalFoveatedMap.terrain`: local terrain of the latest scan, `Terrain.sample(xy)`, NaN where none.
- `src.config.MAP_VERSION`: bumped whenever fusion or hand-off changes what `render()` returns.

| Version | What changed |
|---|---|
| 1 | Phases 3–5: centre-based ring hand-off. Seam gaps just past ring edges. |
| 2 | Edge-strip cells kept in both bands (copy when missing, removed only when wholly on the other side). Each band also ingests points one own-cell diagonal past its edges, for cells touching its ring. No seam gap. |

## 2. Snapshots (frozen map for model work)

- Run: `python -m scripts.export_snapshots --seq <dir> [--start 10 --every 5]`
- Writes `outputs/snapshots/mapv<N>/<seq>_<first>-<last>/<scan>.npz`.
- `load_snapshot(path) -> (FoveatedLayers, Terrain, meta)`. Lookups match the live map exactly (tested).
- `meta`: `map_version`, `sequence`, `scan`, `index`, `t`, `pose` (4×4 T_world_velo), `frame_note`.
- Reference (truth) maps are not in snapshots. Build them from raw scans with `build_reference`.

## 3. Model → everyone

- `HazardLabeler(model, zone_m).update(fov, terrain, t) -> int` (cells run through the model this scan).
- `HazardLabeler.labels_at(fov, xy) -> int64 array`.
  - Class ids from `src.models.targets`: 0 traversable, 1 potentially hazardous, 2 obstacle, 3 unknown.
  - −1 = no model label (outside zone or unobserved). The caller then uses the threshold rule.
- Model file (joblib dict) keys: `model`, `features`, `classes`, `version`, `map_version`, `train_slice`, `val_slice`.
  The loader must refuse a file whose `features` or `map_version` differ from the code.

## 4. Corridor input

- The corridor check takes a `label_at(xy) -> class id` callable only.
- It never imports model code, so it works with rule labels before the model is ready.

## 5. Latency budget (p95 per scan, 100 ms total, 10 Hz)

| Part | Owner | Budget p95 |
|---|---|---|
| map integrate + render | Person 2 | 65 ms |
| hazard labeler update | Person 1 | 30 ms |
| corridor check | Person 2 | 5 ms |

- Measure with `scripts/benchmark.py` on test seq 08 1500–1799.
- Timing varies run to run on the same machine. Compare against a same-day baseline run before claiming a change.

## 6. Data split

- Train: seq 00 scans 0–299.
- Validation (tuning only): seq 05 slice.
- Test (final numbers only, never tuning): seq 08 scans 1500–1799.
