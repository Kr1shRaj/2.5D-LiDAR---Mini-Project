# Replay viewer

Browser replay of the foveated 2.5D map, scan by scan. Static HTML + plain JS modules, no build step, no npm.
Pure consumer: reads replay files written by `scripts/export_replay.py`, never imports or calls `src/`.
Only third-party code: one pinned Three.js file in `vendor/` (see `vendor/README.md`), loaded by the 3D tab only.

## Run

From repo root (KITTI slices fetched as in main README):

```bash
python -m scripts.export_replay          # train seq 00 0-299 model (cached), replay seq 08 1500-1799
# multi-turn demo slice (3 turns, held-out test seq), fetched to its own folder so benchmark slices stay as they are
python scripts/fetch_kitti_slice.py --seq 08 --start 2400 --count 300 --out data/kitti_demo
python -m scripts.export_replay --test data/kitti_demo/sequences/08 --scene "seq 08 2400-2699 · 3 turns (test)"
python -m http.server 8000 --bind 127.0.0.1
# open http://127.0.0.1:8000/viewer/
```

- Each export adds its folder to `outputs/replay/index.json`; the **Scene** menu lists them.
  Default scene: `08_2400-2699` if exported, else `08_1500-1799`.
- `--no-reference` skips the per-scan reference verdicts (faster export, no reference row in the strip).

- First export trains the same F-A random forest as `scripts/benchmark.py` (seed 0, about 2 min) and caches it in
  `outputs/replay/model_fa_seq00.joblib`.
- Then 2 passes over the test slice:
  - Pass 1 is timed: integrate + render, then labeler update, same split as the benchmark.
  - Pass 2 repeats the deterministic pipeline untimed and writes frames.
- Output: `outputs/replay/08_1500-1799/` (git ignored). 300 frames, about 158 MB gzip (map v5 adds refined patch cells).
- Other replay folder: `http://127.0.0.1:8000/viewer/?replay=../outputs/replay/<folder>/`.
- Needs a browser with `DecompressionStream` (current Chrome, Edge, Firefox, Safari).
- Browsers block `file://` data loads, so a local server is required.
- `--bind 127.0.0.1` keeps the repo off your network.

## What it shows

| Fill (keys 1-4) | Meaning |
|---|---|
| Hazard labels | Green traversable, orange potentially hazardous, red obstacle, grey unknown. Model labels inside 25 m, threshold rule beyond. Transparent = unobserved (no data), which is not the unknown class. |
| Height over terrain | z max minus local terrain. Obstacle threshold 30 cm. |
| Ground elevation | Cell ground relative to road near car (median ground within 10 m, from data), ±1 m. |
| Confidence + hand-off | Decayed weight. Purple = inherited placeholder from coarse ring, not yet seen at fine resolution. |

- Overlays: ring borders (10 / 25 / 50 m, cell size inside, 25 m = model zone edge), car path, car box (ego box from `src/config.py`).
- Click a spot to pin it. The inspector shows the cell under that world spot every frame:
  - ring, cell size, label + source
  - z max, ground, terrain, roughness
  - scans seen, weight, log-odds, inherited flag
- Each band is clipped to its own ring. Every spot shows the cell `FoveatedLayers.lookup` returns there.

HUD per scan:
- pipeline ms, measured in the export run
- stored + rendered MB
- cells re-labelled
- occupied cells
- observed cells per ring
- class counts
- sparkline of total ms with the 100 ms budget line (click to seek)

Replay frames/s is browser speed, not pipeline speed.

## Real vs simulation badge

- Green **REAL KITTI REPLAY** badge: recorded KITTI scans, real pipeline, the car drove itself.
- Pink **SIMULATION DEMO (NON-BENCHMARK)** badge + banner: synthetic scene (manifest `"synthetic": true`).
  Never a result; see `AGENTS.md`, simulation demo exception.

## 2D map and 3D chase tabs (key T)

- **2D map:** top-down, as before.
- **3D chase:** every observed cell as a box at its true cell size (10 / 20 / 50 cm, refined 20 cm patches
  replace 50 cm cells where observed), from cell ground up to z max, coloured by the same fill.
  - Current scan points, arcs, picked footprint ribbon, driven path, car box, ghost clears.
  - Camera follows car heading. Drag = orbit, wheel = distance, double click or R = reset.
  - Arcs and rings are drawn at road level near the car (median ground within 10 m, from data).

## Extra layers

| Toggle | Shows |
|---|---|
| Points | Current scan, every 4th point. Grey = road (< 15 cm over local terrain), amber 15-30 cm, red ≥ 30 cm, blue-grey = no terrain there. |
| Ghost clears | Magenta: cells where ghost clearing dropped an old obstacle z max this scan (held obstacle evidence + height ≥ 3 scans in a row from own scans, now seen free). Fade over 10 scans. Ring hand-off placeholders replaced by their first real fine scan are not counted. |
| Refined | Cyan tint on refined 20 cm hazard patches (25-50 m ring). |

## Corridor story

- **Verdict card** (top right of map): straight verdict in big text, the reason (cluster distance, or hazard /
  unseen share), the picked candidate, the arc nearest the path the driver really took, and two tags:
  - candidate vs driver (within one arc = agree)
  - live vs **reference** straight verdict (rule labels on a dense stack of scans i−5..i+5, as `scripts/eval_corridor.py`)
- **Strip** under the map, one column per scan, click to seek:
  1. live straight verdict
  2. reference straight verdict
  3. candidate vs driver: cyan agree, purple differ, grey none
  - white ticks on top = events

## Corridor layer

Toggle **Corridor**:
- 15 candidate arcs (radius ≥ 10 m, 20 m long): green clear, orange caution, red blocked.
- ✕ = first obstacle cluster on a blocked arc.
- Cyan band + arrow = picked candidate.
- Dashed white = path the driver really took next.

HUD: straight verdict, candidate arc, clear length, unknown / hazard share, corridor ms.

**Events** menu (or N / P keys) jumps to frames where straight is blocked and another arc is a candidate, zoomed to the 25 m zone.

Candidate check only: the KITTI car really drove, nothing was steered.

## Honesty notes

- Pipeline ms change run to run on the same machine. Two export runs of the unchanged pipeline gave:

  | Run | Median ms | p95 ms | Scans over 100 ms |
  |---|---|---|---|
  | 1 | 81.2 | 100.0 | 5.0 % |
  | 2 | 83.8 | 110.8 | 12.4 % |

  The report's benchmark run gave 77.4 / 99.6 / 5.0 %. The HUD shows the numbers of the run that wrote the replay.
- Heights are quantized to 1 cm and roughness to 1 mm for display only. Labels are computed in Python before quantizing.
- Labels are candidate hazard labels from geometry plus a small model. Not collision prediction.
- KITTI data is CC BY-NC-SA 3.0. Replay files are derived from it: non-commercial use, attribute KITTI.

## Controls

- Space: play / pause.
- ← / →: step one frame. Shift + arrow: step 10.
- Home / End: first / last frame.
- 1-4: fill.
- T: 2D map / 3D chase.
- N / P: next / previous corridor event.
- V: cycle view (car centred heading up, car centred world axes, world fixed).
- R: reset view.
- Drag: pan. Wheel: zoom around cursor.
- Speed menu: 0.25-4x of real scan time (10 Hz).

## Frame format (`fovreplay-1`)

`manifest.json` holds:
- frame definition: world = KITTI velodyne frame at sequence scan 0, metres, z up
- bands, columns (dtype, scale, NaN sentinel), classes, thresholds, timing note
- per frame: pose `[x, y, z, yaw]`, ring centre, band lattice (`x_min`, `y_min`, `nx`, `ny`, `n_obs`), ms, MB, counts

Additive fields (old viewers ignore them):
- manifest: `scene`, `synthetic`, `points` (format note), `ghosts` (rule), `corridor.reference`, `corridor.driven`
- per frame: `cleared` (ghost-cleared cells, `[dx, dy]` m from sensor xy), `refined` (patch cells), `points` (file),
  `corridor.drv` (driver curvature, null near slice end), `corridor.ref` (reference verdict per arc)
- `frames/NNNNNN.pts.gz` (`fovpoints-1`): `u32 n`, then `i2 dx`, `i2 dy` (cm from sensor xy), `i2 z` (world, cm),
  `i1 h` (height over local terrain, 2 cm, −128 = none), each padded to 4 bytes.

Each `frames/NNNNNN.bin.gz` is gzip of the bands near to far. Each band holds:
1. Observed mask:
   - packed bits over the dense lattice
   - flat id = `ix * ny + iy`, little-endian bit order
   - padded to 4 bytes
2. One column per manifest entry:
   - values for the set bits, in flat order
   - little endian
   - each column padded to 4 bytes

Value = int × scale. NaN = sentinel (signed: dtype min, unsigned: dtype max). An unobserved cell is never written, so it can never read as 0.0. Encoder / decoder + tests: `src/viz/replay.py`, `tests/test_replay.py`.
