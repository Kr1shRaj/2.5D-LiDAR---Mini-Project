from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.models.targets import HAZARDOUS, OBSTACLE, UNKNOWN
from src.preprocess.transforms import velo_to_world

# Candidate corridor check. Research prototype: it reports which constant-curvature arcs ahead of the
# sensor look blocked / caution / clear on the 2.5D map. It does not steer, plan or control anything,
# and it does not predict collisions. Labels are candidate hazard labels from the map.
#
# Arcs live in the velodyne frame of the current scan (x forward, y left), in the sensor's horizontal
# plane (z = 0), and move to world with the scan pose T_world_velo. Only xy is used for map lookup.

CLEAR, CAUTION, BLOCKED = 0, 1, 2
VERDICTS = ["clear", "caution", "blocked"]
NO_LABEL = -1  # label_at answer for unobserved spot / outside map: counts as unknown, never as free

LabelAt = Callable[[np.ndarray], np.ndarray]
OccupiedAt = Callable[[np.ndarray], np.ndarray]  # xy -> bool, map's own obstacle evidence (log-odds > 0)


# knobs for arc fan, footprint and verdicts. lengths in metres, curvature in 1/m.
@dataclass(frozen=True)
class CorridorParams:
    curvatures: tuple[float, ...] = tuple(np.round(np.linspace(-0.1, 0.1, 15), 6))  # min radius 10 m
    length_m: float = 20.0  # look-ahead along arc, inside 0-25 m model zone
    step_m: float = 0.2  # probe spacing along arc
    half_width_m: float = 1.1  # ego box half width 0.9 m + 0.2 m margin
    lateral_step_m: float = 0.2  # probe spacing across footprint
    stop_m: float = 8.0  # obstacle closer than this along arc = blocked
    window_m: float = 1.0  # obstacle must be a cluster: min_hits obstacle probes within this stretch of arc
    min_hits: int = 4  # ~ two 20 cm cells. lone noisy cell does not block (obstacle label precision ~0.56)
    max_hazard: float = 0.10  # hazard share before block over this = caution
    max_unknown: float = 0.30  # unknown share before block over this = caution
    w_hazard: float = 0.5  # score penalties, fixed before any test scoring
    w_unknown: float = 0.3
    w_turn: float = 1.0
    w_change: float = 1.0


# per arc result of one check. arrays line up with params.curvatures.
@dataclass
class CorridorResult:
    curvatures: np.ndarray
    verdict: np.ndarray  # CLEAR / CAUTION / BLOCKED
    block_m: np.ndarray  # distance along arc to first obstacle cluster, length_m if none
    hazard: np.ndarray  # share of probes before block labelled potentially hazardous
    unknown: np.ndarray  # share of probes before block unobserved or labelled unknown
    score: np.ndarray  # -inf for arcs that cannot be picked
    pick: int  # best arc not blocked and mostly seen, -1 = no clear candidate
    extra: dict = field(default_factory=dict)


# station distances along arc and lateral offsets across footprint
def _samples(p: CorridorParams) -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(0.0, p.length_m + 1e-9, p.step_m)
    n_lat = 2 * int(np.ceil(p.half_width_m / p.lateral_step_m)) + 1  # odd: one probe on centreline
    return s, np.linspace(-p.half_width_m, p.half_width_m, n_lat)


# probe xy of every arc in velodyne frame, shape (arcs, stations, lateral, 2).
# centreline: x = sin(k s) / k, y = (1 - cos(k s)) / k (straight line when k = 0); heading = k s.
# footprint probe = centreline + l * (-sin heading, cos heading), l across car width.
def arc_probes_velo(p: CorridorParams) -> np.ndarray:
    s, lat = _samples(p)
    k = np.array(p.curvatures)[:, None]
    ks = k * s[None, :]
    safe = np.where(np.abs(k) < 1e-9, 1.0, k)
    x = np.where(np.abs(k) < 1e-9, s[None, :], np.sin(ks) / safe)
    y = np.where(np.abs(k) < 1e-9, 0.0, (1 - np.cos(ks)) / safe)
    nx, ny = -np.sin(ks), np.cos(ks)
    px = x[:, :, None] + lat[None, None, :] * nx[:, :, None]
    py = y[:, :, None] + lat[None, None, :] * ny[:, :, None]
    return np.stack([px, py], axis=-1)


# velodyne-frame probe xy -> world xy with scan pose (sensor horizontal plane, z = 0 in velodyne frame)
def probes_to_world(probes_velo: np.ndarray, t_world_velo: np.ndarray) -> np.ndarray:
    flat = probes_velo.reshape(-1, 2)
    pts = np.column_stack([flat, np.zeros(len(flat))])
    return velo_to_world(pts, t_world_velo)[:, :2].reshape(probes_velo.shape)


# score arcs from labels at their probes. labels shape (arcs, stations, lateral), class ids or NO_LABEL.
def score_arcs(labels: np.ndarray, p: CorridorParams, prev_curvature: float | None = None) -> CorridorResult:
    s, _ = _samples(p)
    k = np.array(p.curvatures)
    # obstacle probes per station, summed over window ahead of each station; first window over min_hits
    per = (labels == OBSTACLE).sum(axis=2)
    w = max(int(round(p.window_m / p.step_m)), 1)
    c = np.concatenate([np.zeros((len(k), 1)), np.cumsum(per, axis=1)], axis=1)
    win = c[:, np.minimum(np.arange(len(s)) + w, len(s))] - c[:, :len(s)]
    hit = win >= p.min_hits
    start = np.where(hit.any(axis=1), hit.argmax(axis=1), len(s))
    # block distance = first obstacle probe inside that window
    nz = (per > 0) & (np.arange(len(s))[None, :] >= start[:, None])
    first = np.where(start < len(s), nz.argmax(axis=1), len(s))
    block = np.where(first < len(s), s[np.minimum(first, len(s) - 1)], p.length_m)
    before = np.arange(len(s))[None, :] < first[:, None]
    n = np.maximum(before.sum(axis=1) * labels.shape[2], 1)
    hazard = ((labels == HAZARDOUS) & before[:, :, None]).sum(axis=(1, 2)) / n
    unknown = (((labels == UNKNOWN) | (labels == NO_LABEL)) & before[:, :, None]).sum(axis=(1, 2)) / n
    verdict = np.full(len(k), CLEAR)
    verdict[(hazard > p.max_hazard) | (unknown > p.max_unknown) | (block < p.length_m)] = CAUTION
    verdict[block < p.stop_m] = BLOCKED
    kmax = max(np.abs(k).max(), 1e-9)
    score = block * (1 - p.w_hazard * hazard - p.w_unknown * unknown) - p.w_turn * np.abs(k) / kmax
    if prev_curvature is not None:
        score = score - p.w_change * np.abs(k - prev_curvature) / kmax
    # blocked arcs and arcs the map mostly has not seen are never picked: unseen is not free
    score = np.where((verdict == BLOCKED) | (unknown > p.max_unknown), -np.inf, score)
    pick = int(np.argmax(score)) if np.isfinite(score).any() else -1
    return CorridorResult(curvatures=k, verdict=verdict, block_m=block, hazard=hazard, unknown=unknown,
                          score=score, pick=pick)


# obstacle label without map evidence -> hazardous (caution, not block). label alone over-fires on flat road.
def confirm_obstacles(labels: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    return np.where((labels == OBSTACLE) & ~occupied, HAZARDOUS, labels)


# full check for one scan: arcs -> world -> labels via label_at (+ evidence via occupied_at) -> verdicts + pick
def check_corridor(label_at: LabelAt, t_world_velo: np.ndarray, p: CorridorParams,
                   prev_curvature: float | None = None, occupied_at: OccupiedAt | None = None) -> CorridorResult:
    probes = probes_to_world(arc_probes_velo(p), t_world_velo)
    flat = probes.reshape(-1, 2)
    labels = label_at(flat)
    if occupied_at is not None:
        # evidence only needed where label says obstacle (few % of probes): one small lookup
        ob = np.flatnonzero(labels == OBSTACLE)
        occ = np.ones(len(flat), dtype=bool)
        occ[ob] = occupied_at(flat[ob])
        labels = confirm_obstacles(labels, occ)
    return score_arcs(labels.reshape(probes.shape[:3]), p, prev_curvature)
