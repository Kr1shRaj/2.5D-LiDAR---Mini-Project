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
