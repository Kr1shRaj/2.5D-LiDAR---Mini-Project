# Simulation demo (NON-BENCHMARK)

A separate, clearly badged mode so a viewer can watch the pipeline react: a virtual car on a **synthetic**
multi-turn street sees obstacles in its lane and steers around them in closed loop.
Rules: `AGENTS.md`, "Simulation demo exception". Nothing here is a result.

```bash
python -m scripts.run_sim            # ~4 min. writes outputs/replay/sim_demo_street/, adds it to the viewer Scene menu
python -m http.server 8000 --bind 127.0.0.1   # viewer: pick "SIM · demo street ..." (pink badge + banner)
```

## What is synthetic and what is real

| Part | Synthetic (sim only) | Real code, unchanged |
|---|---|---|
| Street | flat ground z = 0, 7 m road, 12 cm kerb + 3 m footway, building walls with gaps (`src/sim/world.py`) | |
| Obstacles | parked car, low debris (35 cm), three cones closing the lane, a pedestrian crossing, a stopped car | |
| LiDAR | 64 beams −24.8..+2°, 1024 azimuths, mounted 1.73 m, 2 cm range noise, seeded (`src/sim/lidar.py`) | |
| Map | | `TemporalFoveatedMap` (fusion, decay, ghost clearing, hand-off, refinement), map v5 |
| Labels | | `HazardLabeler` with the KITTI seq 00 model; rule beyond 25 m. The model never sees sim data. |
| Corridor | | `check_corridor`: clear / caution / blocked per arc, obstacle only with map evidence |
| Route intent | road centreline, pure pursuit to the ego lane 8 m+ ahead (`src/sim/drive.py`) | |
| Car | kinematic: curvature rate limit 0.03 /m per scan, accel 2, brake 4 m/s², cruise 8 m/s | |

## Closed loop, per scan (10 Hz)

1. Raycast the scene from the virtual car pose (velodyne frame, flat road at z −1.73 m as on KITTI).
2. Range gate + ego box, then map integrate + render, labeler update, corridor check. Timed like the real export.
3. Arc choice, three modes:
   - **drive**: among arcs the corridor check would pick (not blocked, ≤ 30 % unseen), the one nearest the route
     curvature, penalised for caution and short clear length.
   - **creep** (1.5 m/s): none pickable *only* because the map has not seen enough yet (standing start). Unseen is
     still never called free; rolling slowly fills it.
   - **stop**: every arc blocked.
4. Speed: 8 m/s when the chosen arc and straight are clear, ≤ 4 m/s otherwise, slower when the clear length is short.
5. Move, then repeat.

The corridor check never sees the route or the scene boxes, only the map. In the viewer, cyan = corridor pick,
yellow dashed = arc the sim drove (shown when different), white boxes = sim truth.

## Honesty notes

- Sim scans are cleaner than KITTI (no reflectivity, no beam dropouts, perfect pose, box-shaped objects), so the
  map, labels and corridor look better here than on real data. Real behaviour is in the KITTI replays and
  `scripts/eval_corridor.py`.
- Contact and off-road counts in the manifest are a sim self-check against its own truth, not a safety claim.
- Pipeline ms in the sim manifest exclude raycasting and are labelled "simulation demo, not a benchmark".
- Wording: "demo steering of a virtual car in a synthetic scene". The real system never steers anything.
