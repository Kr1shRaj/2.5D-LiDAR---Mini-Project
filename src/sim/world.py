from dataclasses import dataclass, field

import numpy as np

# SIMULATION DEMO (NON-BENCHMARK). synthetic multi-turn street, all geometry fabricated on purpose.
# world frame: metres, z up, flat ground plane z = 0. boxes = oriented boxes (yaw about z), bottom z0, top z1.

ROAD_HALF_M = 3.5  # carriageway half width: two 3.5 m lanes
KERB_H_M = 0.12  # footway top over road
FOOTWAY_M = 3.0
WALL_GAP_M = 2.0  # verge between footway and building line
WALL_H_M = 6.0
SEG_M = 2.0  # footway / wall pieces this long along route (overlap a bit so bends stay closed)


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    yaw: float
    length: float  # along yaw
    width: float
    z0: float
    z1: float
    kind: str  # kerb / wall / car / pedestrian / debris / cone


@dataclass
class Mover:
    box: Box  # pose before it starts
    vx: float
    vy: float
    duration: float  # walks this long, then stands
    trigger_station: float  # starts when virtual car passes this route station
    t0: float = np.inf  # sim time it started, set by drive loop

    # box at sim time t (holds still before t0 and after t0 + duration)
    def at(self, t: float) -> Box:
        dt = min(max(t - self.t0, 0.0), self.duration) if np.isfinite(self.t0) else 0.0
        b = self.box
        return Box(b.cx + self.vx * dt, b.cy + self.vy * dt, b.yaw, b.length, b.width, b.z0, b.z1, b.kind)


@dataclass
class World:
    route: np.ndarray  # (n, 2) centreline points, 0.5 m apart
    heading: np.ndarray  # (n,) centreline heading, rad
    station: np.ndarray  # (n,) distance along centreline, m
    static: list[Box]
    movers: list[Mover] = field(default_factory=list)

    # every box at sim time t
    def boxes(self, t: float) -> list[Box]:
        return self.static + [m.at(t) for m in self.movers]

    # centreline point + heading at station s (clamped)
    def at_station(self, s: float) -> tuple[np.ndarray, float]:
        i = int(np.clip(np.searchsorted(self.station, s), 0, len(self.station) - 1))
        return self.route[i], float(self.heading[i])

    # station of centreline point nearest xy
    def nearest_station(self, xy: np.ndarray) -> float:
        return float(self.station[np.argmin(np.sum((self.route - xy) ** 2, axis=1))])


# centreline from (kind, value, radius) pieces: ("straight", length) or ("turn", degrees, radius); + = left
def build_route(pieces: list[tuple], step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, h = 0.0, 0.0, 0.0
    pts, hs = [(x, y)], [h]
    for piece in pieces:
        if piece[0] == "straight":
            n = int(round(piece[1] / step))
            k = 0.0
        else:
            ang, radius = np.radians(piece[1]), piece[2]
            n = int(round(abs(ang) * radius / step))
            k = np.sign(ang) / radius
        for _ in range(n):
            h_mid = h + k * step / 2
            x, y = x + step * np.cos(h_mid), y + step * np.sin(h_mid)
            h += k * step
            pts.append((x, y))
            hs.append(h)
    route = np.array(pts)
    return route, np.array(hs), np.arange(len(route)) * step


# box pieces along both sides of route at lateral offset (centre of strip), every SEG_M
def side_strips(route: np.ndarray, heading: np.ndarray, offset: float, width: float, z0: float, z1: float,
                kind: str, rng: np.random.Generator | None = None, gap_p: float = 0.0) -> list[Box]:
    out = []
    every = int(round(SEG_M / 0.5))
    for i in range(0, len(route) - 1, every):
        h = heading[i]
        n = np.array([-np.sin(h), np.cos(h)])
        mid = route[min(i + every // 2, len(route) - 1)]
        for side in (1.0, -1.0):
            if rng is not None and rng.uniform() < gap_p:
                continue  # gap between buildings
            c = mid + side * offset * n
            out.append(Box(float(c[0]), float(c[1]), float(h), SEG_M * 1.15, width, z0, z1, kind))
    return out


# obstacle box at route station s, lateral offset d (+ = left of centreline)
def on_route(route: np.ndarray, heading: np.ndarray, station: np.ndarray, s: float, d: float, length: float,
             width: float, height: float, kind: str, yaw_rel: float = 0.0) -> Box:
    i = int(np.clip(np.searchsorted(station, s), 0, len(station) - 1))
    h = heading[i]
    c = route[i] + d * np.array([-np.sin(h), np.cos(h)])
    return Box(float(c[0]), float(c[1]), float(h + yaw_rel), length, width, 0.0, height, kind)


# default demo street: 3 turns (left 90, right 90, left 60) with obstacles in the driving lane. seeded.
def demo_world(seed: int = 0) -> World:
    rng = np.random.default_rng(seed)
    route, heading, station = build_route([
        ("straight", 70.0), ("turn", 90.0, 25.0), ("straight", 60.0), ("turn", -90.0, 22.0),
        ("straight", 70.0), ("turn", 60.0, 30.0), ("straight", 80.0),
    ])
    kerb = side_strips(route, heading, ROAD_HALF_M + FOOTWAY_M / 2, FOOTWAY_M, 0.0, KERB_H_M, "kerb")
    walls = side_strips(route, heading, ROAD_HALF_M + FOOTWAY_M + WALL_GAP_M + 0.5, 1.0, 0.0, WALL_H_M, "wall",
                        rng, gap_p=0.15)
    # right-hand traffic: ego lane centre at d = -1.75 m (right of centreline)
    obstacles = [
        on_route(route, heading, station, 45.0, -1.6, 4.4, 1.8, 1.5, "car"),  # parked / stopped car in ego lane
        on_route(route, heading, station, 132.0, -1.2, 0.6, 0.6, 0.35, "debris"),  # low debris just past 1st turn
        on_route(route, heading, station, 190.0, -2.6, 0.4, 0.4, 0.7, "cone"),
        on_route(route, heading, station, 194.0, -1.8, 0.4, 0.4, 0.7, "cone"),
        on_route(route, heading, station, 198.0, -1.0, 0.4, 0.4, 0.7, "cone"),  # cones closing ego lane
        on_route(route, heading, station, 300.0, -1.7, 4.6, 1.9, 1.6, "car"),  # stopped car after 2nd turn
    ]
    # pedestrian steps off right footway at station 250, walks left at 1.4 m/s once car is 28 m away
    ped0 = on_route(route, heading, station, 250.0, -ROAD_HALF_M - 1.0, 0.5, 0.5, 1.75, "pedestrian")
    h = heading[np.searchsorted(station, 250.0)]
    speed = 1.4
    mover = Mover(ped0, vx=float(-np.sin(h) * speed), vy=float(np.cos(h) * speed),
                  duration=(2 * ROAD_HALF_M + 2.0) / speed, trigger_station=250.0 - 28.0)
    return World(route=route, heading=heading, station=station, static=kerb + walls + obstacles, movers=[mover])
