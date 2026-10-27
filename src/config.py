from src.mapping.foveated_grid import Band

# phase 1 params. team picked these. all velodyne frame.
GROUND_PERCENTILE = 10.0
EGO_BOX_VELO = (-1.2, 1.6, -0.9, 0.9, -1.8, 0.0)  # x_min, x_max, y_min, y_max, z_min, z_max [m]
RANGE_MIN_M = 2.0
RANGE_MAX_M = 50.0

# phase 2 foveated bands (F-A). team picked after seq 00 eval. sensor-centred, velodyne frame.
FOVEATED_BANDS = (
    Band(0.0, 10.0, 0.10),
    Band(10.0, 25.0, 0.20),
    Band(25.0, RANGE_MAX_M, 0.50),
)

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
HAZARD_ROUGH_M = 0.05  # z std inside cell (gravel, grass, broken surface)
REF_MIN_POINTS = 5  # reference cell with fewer points over whole slice = unknown
