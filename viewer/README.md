# Replay viewer

Browser replay of the foveated 2.5D map, scan by scan. Static HTML + plain JS modules, no build step, no npm.
Pure consumer: reads replay files written by `scripts/export_replay.py`, never imports or calls `src/`.

## Run

From repo root (KITTI slices fetched as in main README):

```bash
python -m scripts.export_replay          # train seq 00 0-299 model (cached), replay seq 08 1500-1799
python -m http.server 8000 --bind 127.0.0.1
# open http://127.0.0.1:8000/viewer/
```

- First export trains the same F-A random forest as `scripts/benchmark.py` (seed 0, about 2 min) and caches it in
  `outputs/replay/model_fa_seq00.joblib`.
- Then 2 passes over the test slice:
  - Pass 1 is timed: integrate + render, then labeler update, same split as the benchmark.
  - Pass 2 repeats the deterministic pipeline untimed and writes frames.
- Output: `outputs/replay/08_1500-1799/` (git ignored). 300 frames, about 124 MB gzip.
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
- V: cycle view (car centred heading up, car centred world axes, world fixed).
- R: reset view.
- Drag: pan. Wheel: zoom around cursor.
- Speed menu: 0.25-4x of real scan time (10 Hz).

## Frame format (`fovreplay-1`)

`manifest.json` holds:
- frame definition: world = KITTI velodyne frame at sequence scan 0, metres, z up
- bands, columns (dtype, scale, NaN sentinel), classes, thresholds, timing note
- per frame: pose `[x, y, z, yaw]`, ring centre, band lattice (`x_min`, `y_min`, `nx`, `ny`, `n_obs`), ms, MB, counts

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
