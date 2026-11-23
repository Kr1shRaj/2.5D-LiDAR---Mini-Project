from src.mapping.foveated_grid import Band

# phase 1 params. team picked these. all velodyne frame.
GROUND_PERCENTILE = 10.0
EGO_BOX_VELO = (-1.2, 2.8, -0.9, 0.9, -1.8, 0.0)  # x_min, x_max, y_min, y_max, z_min, z_max [m]. x_max 1.6 -> 2.8 (map v4): car body points at x 2.5-2.6, y +-0.5, z -0.9 in 93 % of scans
RANGE_MIN_M = 2.0
RANGE_MAX_M = 50.0

# phase 2 foveated bands (F-A). team picked after seq 00 eval. sensor-centred, velodyne frame.
FOVEATED_BANDS = (
    Band(0.0, 10.0, 0.10),
    Band(10.0, 25.0, 0.20),
    Band(25.0, RANGE_MAX_M, 0.50),
)

# fused map output version. bump when fusion / hand-off changes what render() returns.
# v1 = phases 3-5 (centre-based ring hand-off, seam gaps). v2 = edge strip kept in both bands, no seam.
# v3 = v2 + `points` layer (mean points per scan in cell).
# v4 = v3 + ego box x_max 2.8 m: recording-car body points no longer leave obstacle trail along driven path.
# v5 = v4 + adaptive refinement: hazard-cued 20 cm patches inside the 50 cm far ring.
MAP_VERSION = 5

# phase 3 temporal fusion. first-pick values, tune on data. log-odds = octomap defaults (p 0.7 hit, 0.4 miss).
TEMPORAL_TAU_S = 3.0  # stale decay time constant
TEMPORAL_PRUNE_WEIGHT = 0.05  # one-scan cell forgotten after ~3 tau unseen
OBSTACLE_H_M = 0.30  # z max this far over local terrain = obstacle
LOGODDS_HIT = 0.85
LOGODDS_MISS = -0.4
LOGODDS_MIN = -2.0
LOGODDS_MAX = 3.5
TERRAIN_CELL_M = 2.0  # local terrain cell; 3x3 min window spans 6 m, wider than car

# phase 4 geometric safety thresholds -> target classes. first picks, team approve / tune.
HAZARD_STEP_M = 0.08  # ground change across 40 cm (curb, pothole edge)
HAZARD_SLOPE = 0.15  # ground rise over 1 m (15 %, ~8.5 deg)
HAZARD_ROUGH_M = 0.08  # z std inside cell (gravel, grass, broken surface). 5 cm sat in stacked-scan noise (road p90 4.6 cm)
REF_MIN_POINTS = 5  # reference cell with fewer points over whole slice = unknown

# adaptive refinement (map v5). far ring base cells 50 cm; hazard cue -> local patch at 20 cm.
REFINE_RES_M = 0.2  # equals mid ring cell size, so fine cells copy straight across the 25 m edge
REFINE_STEP_M = HAZARD_STEP_M  # ground change to a 4-neighbour coarse cell
REFINE_STD_M = HAZARD_ROUGH_M  # z std inside coarse cell ...
REFINE_STD_MAX_M = OBSTACLE_H_M  # ... up to obstacle height: rough ground, not walls (metre-scale std)
REFINE_ON_OBSTACLE = True  # obstacle evidence (log-odds > 0) also triggers
REFINE_PATCH_CELLS = 1  # patch = trigger cell + this many coarse cells around it (1 -> 3 x 3)
REFINE_HOLD_S = 1.0  # patch stays refined this long after last trigger
REFINE_BUDGET_CELLS = 3000  # cap on refined 50 cm cells (~13 % of outer ring): bounds latency in clutter
REFINE_EVERY_SCANS = 2  # cue update + patch seeding every 2nd scan; hold time covers gap. points still fed every scan
REFINE_MIN_WEIGHT = 3.0  # cue only from outer cells seen in >= ~3 recent scans: far ground noise (err med 0.12 m) fired 8 cm step everywhere
