from dataclasses import dataclass

import numpy as np

from src.planning.corridor import BLOCKED, CAUTION, CLEAR, CorridorResult
from src.sim.world import ROAD_HALF_M, Box, World

# SIMULATION DEMO (NON-BENCHMARK). closed loop for a virtual car in a synthetic scene.
# split of roles, kept visible in the export:
#   corridor check (real code, unchanged) = which arcs are clear / caution / blocked, from the map only
#   route intent (sim only)               = where the road goes, from the synthetic centreline (map knows nothing of it)
# the car drives the non-blocked arc nearest the route curvature. it never sees the world boxes directly.

LANE_D_M = -1.75  # right-hand traffic: ego lane centre right of road centreline


@dataclass(frozen=True)
class DriveParams:
    cruise_ms: float = 8.0  # ~ KITTI seq 00 mean speed
    caution_ms: float = 4.0  # chosen arc caution, or straight not clear
    creep_ms: float = 1.5  # no pickable arc only because map has not seen enough yet: roll slowly so it fills
    accel: float = 2.0  # m/s^2
    brake: float = 4.0  # m/s^2
    dk_max: float = 0.03  # curvature change per scan (1/m): steering rate limit
    lookahead_min_m: float = 8.0
    lookahead_s: float = 1.2  # lookahead = max(min, speed * this)
    w_caution: float = 0.3  # arc choice: penalty for caution arc, in units of full curvature span
    w_short: float = 0.5  # ... and for clear length short of full arc


@dataclass
class CarState:
    x: float
    y: float
    yaw: float
    v: float
    k: float = 0.0  # current path curvature


# pure-pursuit curvature toward ego-lane point lookahead metres down the route (velo frame: x fwd, y left)
def route_curvature(world: World, car: CarState, p: DriveParams) -> float:
    s = world.nearest_station(np.array([car.x, car.y]))
    look = max(p.lookahead_min_m, car.v * p.lookahead_s)
    c, h = world.at_station(s + look)
    tx, ty = c + LANE_D_M * np.array([-np.sin(h), np.cos(h)])
    dx, dy = tx - car.x, ty - car.y
    lx = np.cos(car.yaw) * dx + np.sin(car.yaw) * dy
    ly = -np.sin(car.yaw) * dx + np.cos(car.yaw) * dy
    return float(2.0 * ly / max(lx * lx + ly * ly, 1e-6))


DRIVE, CREEP, STOP = "drive", "creep", "stop"


# arc to drive + mode. DRIVE: among arcs the corridor check would let be picked (not blocked, mostly seen), the one
# nearest route curvature, penalised for caution and short clear length. CREEP: none pickable, but some arcs are
# rejected only for being unseen (not blocked): take nearest-route of those at creep speed, so the map fills
# (unseen is still never called free). STOP: every arc blocked.
def choose_arc(r: CorridorResult, k_route: float, length_m: float, p: DriveParams) -> tuple[int, str]:
    span = max(np.ptp(r.curvatures), 1e-9)
    cost = (np.abs(r.curvatures - k_route) / span + p.w_caution * (r.verdict == CAUTION)
            + p.w_short * (1.0 - r.block_m / length_m))
    ok = np.isfinite(r.score)
    if ok.any():
        return int(np.argmin(np.where(ok, cost, np.inf))), DRIVE
    unseen_only = r.verdict != BLOCKED
    if unseen_only.any():
        return int(np.argmin(np.where(unseen_only, np.abs(r.curvatures - k_route), np.inf))), CREEP
    return -1, STOP


# speed target from chosen arc, mode + straight verdict; brake to stop when no arc
def target_speed(r: CorridorResult, choice: int, mode: str, straight: int, p: DriveParams) -> float:
    if mode == STOP:
        return 0.0
    if mode == CREEP:
        return p.creep_ms
    if r.verdict[choice] == CLEAR and r.verdict[straight] == CLEAR:
        return p.cruise_ms
    # slow inside caution / when straight is not clear; clear length short -> slower still
    return min(p.caution_ms, p.cruise_ms * float(r.block_m[choice]) / 20.0 + 1.0)


# one scan of kinematic motion: speed toward target within accel / brake, curvature toward chosen within rate limit
def step(car: CarState, k_cmd: float, v_cmd: float, dt: float, p: DriveParams) -> CarState:
    dv = np.clip(v_cmd - car.v, -p.brake * dt, p.accel * dt)
    v = max(car.v + dv, 0.0)
    k = car.k + float(np.clip(k_cmd - car.k, -p.dk_max, p.dk_max))
    yaw_mid = car.yaw + 0.5 * k * v * dt
    return CarState(car.x + v * dt * np.cos(yaw_mid), car.y + v * dt * np.sin(yaw_mid), car.yaw + k * v * dt, v, k)


# ground-truth contact check: sample car footprint (ego box xy, velo frame) and test inside any tall box.
# kerb strips are drivable-height (12 cm) and not counted; walls and obstacles are.
def contacts(car: CarState, ego_box: tuple, boxes: list[Box]) -> list[str]:
    x0, x1, y0, y1 = ego_box[:4]
    gx, gy = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 5))
    c, s = np.cos(car.yaw), np.sin(car.yaw)
    px = car.x + c * gx.ravel() - s * gy.ravel()
    py = car.y + s * gx.ravel() + c * gy.ravel()
    hit = []
    for b in boxes:
        if b.kind == "kerb" or np.hypot(b.cx - car.x, b.cy - car.y) > 8.0:
            continue
        cb, sb = np.cos(-b.yaw), np.sin(-b.yaw)
        lx = cb * (px - b.cx) - sb * (py - b.cy)
        ly = sb * (px - b.cx) + cb * (py - b.cy)
        if np.any((np.abs(lx) <= b.length / 2) & (np.abs(ly) <= b.width / 2)):
            hit.append(b.kind)
    return hit


# signed lateral offset of car from road centreline (+ left), for off-road accounting
def lateral_offset(world: World, car: CarState) -> float:
    s = world.nearest_station(np.array([car.x, car.y]))
    c, h = world.at_station(s)
    return float(-np.sin(h) * (car.x - c[0]) + np.cos(h) * (car.y - c[1]))


# car outside carriageway (any part on footway)
def off_road(world: World, car: CarState, half_width: float) -> bool:
    return abs(lateral_offset(world, car)) + half_width > ROAD_HALF_M
