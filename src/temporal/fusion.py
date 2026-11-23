from dataclasses import dataclass, fields, replace

import numpy as np

from src.mapping.foveated_grid import Band, BandLayers, FoveatedLayers, annulus_mask, validate_bands
from src.mapping.uniform_grid import cell_stats_sparse, point_to_cell, spec_from_bounds
from src.preprocess.transforms import velo_to_world
from src.temporal.terrain import Terrain, local_terrain

# Streaming foveated map in WORLD frame. Cells sit on world-aligned lattices (one per band
# resolution), so vehicle rotation never resamples a cell. Rings follow the sensor.
# Each band takes points in its ring plus a margin of one own-cell diagonal past each edge, but keeps
# only cells whose square touches its ring. Edge cells then see every point inside their square
# (a touching cell's points all lie within one diagonal of the ring), not just their own side.
# Each scan counts as one observation per cell (weight 1); old weight decays with time.
# ground = decayed weighted mean of per-scan cell ground (per-scan ground = p-th percentile, as phase 1).
# Obstacle evidence = log-odds per cell: hit if scan z max rises obstacle_h over local terrain.
# Lookup picks band by query point range, so a cell whose square crosses a ring edge answers for its
# own side of the edge. Both bands keep such edge cells (with history) until wholly on the other side.
# Hand-off out: fine cell reaching past its ring builds missing coarse parent (merged copy).
# Hand-off in: coarse cell reaching into finer ring seeds missing fine children that copy parent
# stats, flagged `inherited`. First real fine scan of child replaces copied stats.
# Adaptive refinement (refine_res > 0): outer ring cell with hazard cue (step, z std, obstacle evidence)
# opens a local patch on a finer `refine` lattice. Outer ring keeps ALL its own data (base layer never
# loses history); patch cells are a second, finer layer: seeded from parent (inherited), then fed real
# points. Patch expires after refine_hold_s without cue: fine cells dropped, coarse base still complete.
# Patch cells crossing into the mid ring copy across when missing (same lattice), and back.

KEY_SHIFT = 2**31


# pack world cell index (ix, iy) into one sortable int64 key
def encode_keys(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return ix.astype(np.int64) * 2**32 + (iy.astype(np.int64) + KEY_SHIFT)


# unpack int64 key into world cell index (ix, iy)
def decode_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = keys & 0xFFFFFFFF
    return (keys - low) >> 32, low - KEY_SHIFT


# world xy of cell centres for keys
def key_centers(keys: np.ndarray, resolution: float) -> np.ndarray:
    ix, iy = decode_keys(keys)
    return np.column_stack([(ix + 0.5) * resolution, (iy + 0.5) * resolution])


# tuning knobs for fusion. times in seconds, heights in metres.
@dataclass(frozen=True)
class FusionParams:
    tau_s: float  # stale decay: weight and log-odds shrink by exp(-dt / tau)
    prune_weight: float  # forget cell (back to unknown) when decayed weight drops below
    obstacle_h: float  # scan z max this far over local terrain = obstacle hit
    l_hit: float
    l_miss: float
    l_min: float
    l_max: float
    terrain_cell: float
    ground_percentile: float
    clear_ghosts: bool = True  # False = naive max fusion, keeps every old z max
    seed_children: bool = True  # False = coarse cell entering finer ring stays hidden, fine cell starts empty
    refine_res: float = 0.0  # adaptive refinement cell size in outer ring; 0 = off
    refine_step: float = 0.08  # trigger: ground change to a 4-neighbour outer cell
    refine_std: float = 0.08  # trigger: z std inside outer cell ...
    refine_std_max: float = np.inf  # ... below this (walls / poles have metre-scale std: not a ground hazard)
    refine_on_obstacle: bool = True  # trigger: obstacle evidence log-odds > 0
    refine_patch: int = 1  # patch = trigger cell + this many outer cells around it
    refine_hold_s: float = 1.0  # patch stays refined this long after last trigger
    refine_min_weight: float = 0.0  # cue only from outer cells this confident (decayed scans), noise guard
    refine_budget: int = 0  # max refined outer cells (0 = no cap). steps first, then rough, then obstacle
    refine_every: int = 1  # run cue update + patch seeding every n-th scan (hold time covers the gap)


# sparse cell layers for one band lattice. arrays line up with sorted unique keys.
@dataclass
class CellStore:
    resolution: float
    keys: np.ndarray
    weight: np.ndarray
    mean: np.ndarray
    m2: np.ndarray  # weight * variance
    z_min: np.ndarray
    z_max: np.ndarray
    ground: np.ndarray
    logodds: np.ndarray
    last_seen: np.ndarray
    n_obs: np.ndarray
    points: np.ndarray  # decayed weighted mean of points per scan in cell (point density, not history length)
    inherited: np.ndarray  # True = placeholder copied from coarse parent, no own fine scan yet

    @property
    def size(self) -> int:
        return int(self.keys.size)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f.name).nbytes for f in fields(self) if f.name != "resolution")


ARRAY_FIELDS = [f.name for f in fields(CellStore) if f.name != "resolution"]
FILL = {"keys": 0, "weight": 0.0, "mean": np.nan, "m2": 0.0, "z_min": np.nan, "z_max": np.nan,
        "ground": np.nan, "logodds": 0.0, "last_seen": -np.inf, "n_obs": 0,
        "points": 0.0, "inherited": False}
DTYPES = {"keys": np.int64, "n_obs": np.int64, "inherited": bool}


# empty store for one lattice
def empty_store(resolution: float) -> CellStore:
    arrays = {n: np.empty(0, dtype=DTYPES.get(n, np.float64)) for n in ARRAY_FIELDS}
    return CellStore(resolution=resolution, **arrays)


# keep only rows where mask true
def store_take(store: CellStore, mask: np.ndarray) -> CellStore:
    if mask.all():  # nothing dropped: no copy of 12 arrays (arrays are never edited in place)
        return store
    return replace(store, **{n: getattr(store, n)[mask] for n in ARRAY_FIELDS})


# shrink weight and log-odds by elapsed time, drop cells too faint to trust
def decay_store(store: CellStore, dt: float, params: FusionParams) -> CellStore:
    if dt < 0:
        raise ValueError(f"time went backwards: dt = {dt}")
    lam = np.exp(-dt / params.tau_s)
    out = replace(store, weight=store.weight * lam, m2=store.m2 * lam, logodds=store.logodds * lam)
    return store_take(out, out.weight >= params.prune_weight)


# one scan worth of cell stats on world lattice: keys + per cell n, mean, var, min, max, ground
@dataclass
class ScanObs:
    keys: np.ndarray
    n: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    ground: np.ndarray


# bin world points of one scan on world lattice of `resolution`. bounds = known xy box, skips min/max pass.
def scan_observation(
    points_world: np.ndarray,
    resolution: float,
    ground_percentile: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> ScanObs:
    if len(points_world) == 0:
        e = np.empty(0)
        return ScanObs(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), e, e, e, e, e)
    if bounds is None:
        lo, hi = points_world[:, :2].min(axis=0), points_world[:, :2].max(axis=0)
        bounds = (lo[0], hi[0] + 1e-9, lo[1], hi[1] + 1e-9)
    spec = spec_from_bounds(*bounds, resolution, frame="world")
    ix, iy, valid = point_to_cell(points_world[:, :2], spec)
    flat = ix[valid] * spec.ny + iy[valid]
    ids, n, mean, var, z_min, z_max, ground = cell_stats_sparse(flat, points_world[valid, 2].astype(np.float64),
                                                                 ground_percentile)
    # flat ids sorted ix-major -> world keys come out sorted too
    lix, liy = np.divmod(ids, spec.ny)
    keys = encode_keys(lix + int(round(spec.x_min / resolution)), liy + int(round(spec.y_min / resolution)))
    return ScanObs(keys=keys, n=n, mean=mean, var=var, z_min=z_min, z_max=z_max, ground=ground)


# mask of cells (centres xy, square side res) whose square touches ring [r_min, r_max) around sensor
def touches_ring(centers: np.ndarray, res: float, sensor: np.ndarray, band: Band) -> np.ndarray:
    d = np.abs(centers - sensor)
    near = np.maximum(d - res / 2, 0.0)
    far = d + res / 2
    return (np.hypot(near[:, 0], near[:, 1]) < band.r_max) & (np.hypot(far[:, 0], far[:, 1]) > band.r_min)


# keep selected cells of scan observation
def obs_take(obs: ScanObs, mask: np.ndarray) -> ScanObs:
    return ScanObs(**{f.name: getattr(obs, f.name)[mask] for f in fields(ScanObs)})


# fold one scan observation (weight 1 per cell) into store. terrain_z = local terrain under each obs cell.
def integrate_obs(store: CellStore, obs: ScanObs, terrain_z: np.ndarray, t: float, params: FusionParams) -> CellStore:
    # both key lists sorted: insert missing obs keys by searchsorted (no union sort). fresh arrays, input untouched.
    pos = np.searchsorted(store.keys, obs.keys)
    have = (pos < store.size) & (store.keys[np.minimum(pos, max(store.size - 1, 0))] == obs.keys) if store.size \
        else np.zeros(obs.keys.size, dtype=bool)
    at = pos[~have]
    total = store.size + at.size
    slot_new = at + np.arange(at.size)
    old = np.ones(total, dtype=bool)
    old[slot_new] = False
    new = {}
    for n in ARRAY_FIELDS:
        a = getattr(store, n)
        out = np.empty(total, dtype=a.dtype)
        out[old] = a
        out[slot_new] = obs.keys[~have] if n == "keys" else FILL[n]
        new[n] = out
    keys = new["keys"]
    oi = np.searchsorted(keys, obs.keys)

    # inherited placeholder meets first real fine scan: forget copied stats, start fresh
    seen = (new["n_obs"][oi] > 0) & ~new["inherited"][oi]
    w0 = np.where(seen, new["weight"][oi], 0.0)
    m0 = np.where(seen, new["mean"][oi], 0.0)
    g0 = np.where(seen, new["ground"][oi], 0.0)
    p0 = np.where(seen, new["points"][oi], 0.0)
    m2_0 = np.where(seen, new["m2"][oi], 0.0)
    w1 = w0 + 1.0
    # weighted merge of old cell (weight w0) with this scan (weight 1)
    new["mean"][oi] = (w0 * m0 + obs.mean) / w1
    new["m2"][oi] = m2_0 + obs.var + (obs.mean - m0) ** 2 * w0 / w1
    new["ground"][oi] = (w0 * g0 + obs.ground) / w1
    new["points"][oi] = (w0 * p0 + obs.n) / w1
    new["z_min"][oi] = np.fmin(np.where(seen, new["z_min"][oi], np.nan), obs.z_min)

    known = np.isfinite(terrain_z)
    hit = known & (obs.z_max - terrain_z >= params.obstacle_h)
    miss = known & ~hit
    l1 = np.clip(np.where(seen, new["logodds"][oi], 0.0) + np.where(hit, params.l_hit, 0.0) + np.where(miss, params.l_miss, 0.0),
                 params.l_min, params.l_max)
    zmax_keep = np.fmax(np.where(seen, new["z_max"][oi], np.nan), obs.z_max)
    if params.clear_ghosts:
        # cell now seen free and no longer believed occupied: drop old z max, trust this scan
        new["z_max"][oi] = np.where(miss & (l1 <= 0.0), obs.z_max, zmax_keep)
    else:
        new["z_max"][oi] = zmax_keep
    new["logodds"][oi] = l1
    new["weight"][oi] = w1
    new["last_seen"][oi] = t
    new["n_obs"][oi] = np.where(seen, new["n_obs"][oi], 0) + 1
    new["inherited"][oi] = False
    return CellStore(resolution=store.resolution, **new)


# merge fine cells into coarse store. parent = coarse cell holding child centre.
def merge_children(coarse: CellStore, fine: CellStore) -> CellStore:
    if fine.size == 0:
        return coarse
    c = key_centers(fine.keys, fine.resolution)
    res = coarse.resolution
    parent = encode_keys(np.floor(c[:, 0] / res).astype(np.int64), np.floor(c[:, 1] / res).astype(np.int64))
    allk = np.concatenate([coarse.keys, parent])
    cat = {n: np.concatenate([getattr(coarse, n), getattr(fine, n)]) for n in ARRAY_FIELDS if n != "keys"}
    keys, inv = np.unique(allk, return_inverse=True)
    w = cat["weight"]
    wsum = np.bincount(inv, weights=w, minlength=keys.size)
    mean = np.bincount(inv, weights=w * cat["mean"], minlength=keys.size) / wsum
    var = np.bincount(inv, weights=cat["m2"] + w * (cat["mean"] - mean[inv]) ** 2, minlength=keys.size) / wsum
    ground = np.bincount(inv, weights=w * cat["ground"], minlength=keys.size) / wsum
    out = {"keys": keys, "mean": mean, "ground": ground}
    # confidence-like fields: max over group, not sum (children share same scans)
    for n, ufunc, start in (("weight", np.maximum, -np.inf), ("z_max", np.fmax, np.nan), ("z_min", np.fmin, np.nan),
                            ("logodds", np.maximum, -np.inf), ("last_seen", np.maximum, -np.inf)):
        acc = np.full(keys.size, start)
        ufunc.at(acc, inv, cat[n])
        out[n] = acc
    n_obs = np.zeros(keys.size, dtype=np.int64)
    np.maximum.at(n_obs, inv, cat["n_obs"])
    out["n_obs"] = n_obs
    out["m2"] = out["weight"] * var
    # parent points per scan = sum over its children, at least what parent already had
    nc = coarse.size
    kids = np.bincount(inv[nc:], weights=cat["points"][nc:], minlength=keys.size)
    own = np.zeros(keys.size)
    own[inv[:nc]] = cat["points"][:nc]
    out["points"] = np.maximum(own, kids)
    # merged cell is placeholder only if every part was placeholder
    out["inherited"] = np.bincount(inv, weights=(~cat["inherited"]).astype(np.float64), minlength=keys.size) == 0
    return CellStore(resolution=res, **out)


# split coarse cells into fine children (centres inside parent). children copy parent stats, flagged inherited.
def split_into_children(coarse: CellStore, fine_res: float) -> CellStore:
    if coarse.size == 0:
        return empty_store(fine_res)
    res = coarse.resolution
    pix, piy = decode_keys(coarse.keys)
    # child centre (i + 0.5) * f must sit in [parent_lo, parent_lo + res); eps keeps edge centre with upper parent
    lo_x = np.ceil(pix * res / fine_res - 0.5 - 1e-9).astype(np.int64)
    hi_x = np.ceil((pix + 1) * res / fine_res - 0.5 - 1e-9).astype(np.int64)
    lo_y = np.ceil(piy * res / fine_res - 0.5 - 1e-9).astype(np.int64)
    hi_y = np.ceil((piy + 1) * res / fine_res - 0.5 - 1e-9).astype(np.int64)
    m = int(np.ceil(res / fine_res)) + 1
    off = np.arange(m)
    cx = lo_x[:, None, None] + off[None, :, None]
    cy = lo_y[:, None, None] + off[None, None, :]
    valid = (cx < hi_x[:, None, None]) & (cy < hi_y[:, None, None])
    parent = np.broadcast_to(np.arange(coarse.size)[:, None, None], valid.shape)[valid]
    keys = encode_keys(np.broadcast_to(cx, valid.shape)[valid], np.broadcast_to(cy, valid.shape)[valid])
    out = {n: getattr(coarse, n)[parent] for n in ARRAY_FIELDS if n != "keys"}
    out["inherited"] = np.ones(keys.size, dtype=bool)
    out["points"] = out["points"] * (fine_res / res) ** 2  # share of parent points by area
    order = np.argsort(keys)
    return CellStore(resolution=fine_res, keys=keys[order], **{n: a[order] for n, a in out.items()})


# add cells to store; key already in store keeps existing row (real data beats placeholder).
# both key arrays sorted: membership + merge by searchsorted, no isin / argsort.
def add_missing(store: CellStore, extra: CellStore) -> CellStore:
    if extra.size == 0:
        return store
    if store.size == 0:
        return extra
    pos = np.searchsorted(store.keys, extra.keys)
    present = store.keys[np.minimum(pos, store.size - 1)] == extra.keys
    if present.all():
        return store
    new, at = store_take(extra, ~present), pos[~present]
    # slot of each new row in merged order: its insertion point + new rows before it
    slot_new = at + np.arange(at.size)
    keep = np.ones(store.size + new.size, dtype=bool)
    keep[slot_new] = False
    merged = {}
    for n in ARRAY_FIELDS:
        a, b = getattr(store, n), getattr(new, n)
        out = np.empty(store.size + new.size, dtype=a.dtype)
        out[keep] = a
        out[slot_new] = b
        merged[n] = out
    return CellStore(resolution=store.resolution, **merged)


# streaming foveated map. feed scans in time order with pose; render rings around latest sensor.
class TemporalFoveatedMap:
    def __init__(self, bands: tuple[Band, ...], params: FusionParams) -> None:
        validate_bands(bands)
        self.bands = tuple(bands)
        self.params = params
        self.stores = [empty_store(b.resolution) for b in bands]
        self.t: float | None = None
        self.sensor_xy = np.zeros(2)
        self.terrain: Terrain | None = None  # local terrain of latest scan, reused by hazard labeler
        far = self.bands[-1]
        self.refine_on = params.refine_res > 0 and len(self.bands) > 1 and far.resolution > params.refine_res
        self.fine = empty_store(params.refine_res if self.refine_on else 1.0)  # refined patch cells
        self.ref_keys = np.empty(0, dtype=np.int64)  # refined outer cells (sorted keys) + until when
        self.ref_until = np.empty(0)
        self.ref_score = np.empty(0)
        self._mid_out: CellStore | None = None

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.stores) + self.fine.nbytes + self.ref_keys.nbytes + self.ref_until.nbytes + self.ref_score.nbytes

    @property
    def n_cells(self) -> int:
        return sum(s.size for s in self.stores) + self.fine.size

    # keys of outer-ring parent cell holding each fine cell centre
    def _parent_keys(self, s: CellStore) -> np.ndarray:
        c = key_centers(s.keys, s.resolution)
        res = self.bands[-1].resolution
        return encode_keys(np.floor(c[:, 0] / res).astype(np.int64), np.floor(c[:, 1] / res).astype(np.int64))

    # mask: fine cell's parent is a refined outer cell
    def _in_patch(self, s: CellStore) -> np.ndarray:
        if not s.size or not self.ref_keys.size:
            return np.zeros(s.size, dtype=bool)
        pk = self._parent_keys(s)
        pos = np.minimum(np.searchsorted(self.ref_keys, pk), self.ref_keys.size - 1)
        return self.ref_keys[pos] == pk

    # patch hand-off: expire old patches; fine cells cross 25 m edge as copies (when missing); drop cells
    # out of outer ring or out of any patch. outer base store untouched.
    def _refine_handoff(self, t: float) -> None:
        alive = self.ref_until >= t
        self.ref_keys, self.ref_until, self.ref_score = self.ref_keys[alive], self.ref_until[alive], self.ref_score[alive]
        mid, far = self.bands[-2], self.bands[-1]
        same = np.isclose(mid.resolution, self.fine.resolution)
        f = self.fine
        if f.size:
            near, fard = self._square_dist2(f)
            if same:
                self.stores[-2] = add_missing(self.stores[-2], store_take(f, near < mid.r_max ** 2))
            self.fine = store_take(f, (fard >= mid.r_max ** 2) & (near < far.r_max ** 2) & self._in_patch(f))
        cand = self._mid_out
        if same and cand is not None and cand.size and self.ref_keys.size:
            self.fine = add_missing(self.fine, store_take(cand, self._in_patch(cand)))

    # hazard cues on outer ring -> refresh / open patches (3 x 3 around cue), seed new patches from parent
    def _refine_update(self, t: float) -> None:
        c, p = self.stores[-1], self.params
        far = self.bands[-1]
        if not c.size:
            return
        # only cells this scan touched: untouched cells have no new evidence, their patch ages out by hold time
        sure = (c.weight >= p.refine_min_weight) & ~c.inherited & (c.last_seen == t)
        # cue score: 3 = ground step (curb, pothole edge), 2 = rough cell, 1 = obstacle evidence. budget keeps best.
        score = np.zeros(c.size)
        if p.refine_on_obstacle:
            score[c.logodds > 0] = 1.0
        std = np.sqrt(c.m2 / np.maximum(c.weight, 1e-12))
        score[(std >= p.refine_std) & (std < p.refine_std_max)] = 2.0
        ix, iy = decode_keys(c.keys)
        si = np.flatnonzero(sure)  # step check only between touched, confident cells
        for dx, dy in ((1, 0), (0, 1)):
            nk = encode_keys(ix[si] + dx, iy[si] + dy)
            pos = np.minimum(np.searchsorted(c.keys, nk), c.size - 1)
            nb = (c.keys[pos] == nk) & sure[pos]
            a, b = si[nb], pos[nb]
            jump = np.abs(c.ground[a] - c.ground[b]) >= p.refine_step
            score[a[jump]] = 3.0
            score[b[jump]] = 3.0  # both sides of the step
        score[~sure] = 0.0
        cue = score > 0
        if not cue.any():
            return
        r = np.arange(-p.refine_patch, p.refine_patch + 1)
        dx, dy = np.meshgrid(r, r, indexing="ij")
        px = (ix[cue][:, None] + dx.ravel()[None, :]).ravel()
        py = (iy[cue][:, None] + dy.ravel()[None, :]).ravel()
        pscore = np.repeat(score[cue], dx.size)
        # merge with live patches: per key best score, latest until
        allk = np.concatenate([self.ref_keys, encode_keys(px, py)])
        alls = np.concatenate([self.ref_score, pscore])
        allu = np.concatenate([self.ref_until, np.full(px.size, t + p.refine_hold_s)])
        order = np.argsort(allk, kind="stable")
        ks = allk[order]
        starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
        keys = ks[starts]
        sc = np.maximum.reduceat(alls[order], starts)
        until = np.maximum.reduceat(allu[order], starts)
        ring = touches_ring(key_centers(keys, far.resolution), far.resolution, self.sensor_xy, far)
        keys, sc, until = keys[ring], sc[ring], until[ring]
        if p.refine_budget and keys.size > p.refine_budget:
            # hard cap on patch area bounds latency in cluttered scenes: keep highest score, then freshest
            best = np.lexsort((-until, -sc))[: p.refine_budget]
            best.sort()
            keys, sc, until = keys[best], sc[best], until[best]
        old = np.zeros(keys.size, dtype=bool)
        if self.ref_keys.size:
            pos = np.minimum(np.searchsorted(self.ref_keys, keys), self.ref_keys.size - 1)
            old = self.ref_keys[pos] == keys
        self.ref_keys, self.ref_score, self.ref_until = keys, sc, until
        new = keys[~old]
        if not new.size:
            return
        pos = np.minimum(np.searchsorted(c.keys, new), c.size - 1)
        parents = store_take(c, np.isin(c.keys, new[c.keys[pos] == new]))
        kids = split_into_children(parents, p.refine_res)
        kids = store_take(kids, touches_ring(key_centers(kids.keys, p.refine_res), p.refine_res, self.sensor_xy, far))
        self.fine = add_missing(self.fine, kids)

    # SQUARED near and far distance of each store cell square to sensor (compare with r**2, no sqrt)
    def _square_dist2(self, s: CellStore) -> tuple[np.ndarray, np.ndarray]:
        ix, iy = decode_keys(s.keys)
        h = s.resolution / 2
        dx = np.abs((ix + 0.5) * s.resolution - self.sensor_xy[0])
        dy = np.abs((iy + 0.5) * s.resolution - self.sensor_xy[1])
        nx, ny = np.maximum(dx - h, 0.0), np.maximum(dy - h, 0.0)
        return nx * nx + ny * ny, (dx + h) ** 2 + (dy + h) ** 2

    # edge strip = cells whose square crosses a ring edge. both bands keep history there:
    # out: fine cell reaching past its ring builds its coarse parent if parent missing (copy, no double count);
    #      fine cell leaves once whole square is outside. last ring: just drop (out of local map).
    # in: coarse cell reaching into finer ring seeds missing fine children there (copy, flagged inherited);
    #     coarse cell leaves once whole square is inside finer ring.
    def _handoff(self) -> None:
        n = len(self.bands)
        self._mid_out = None
        for k, band in enumerate(self.bands):
            s = self.stores[k]
            if not s.size:
                continue
            near, far = self._square_dist2(s)
            r2 = band.r_max ** 2
            out = far >= r2
            if k == n - 2:
                self._mid_out = store_take(s, out)  # reused by patch hand-off: mid cells reaching past mid ring
            if k + 1 < n and out.any():
                parents = merge_children(empty_store(self.bands[k + 1].resolution), store_take(s, out))
                self.stores[k + 1] = add_missing(self.stores[k + 1], parents)
            self.stores[k] = store_take(s, near < r2)
        for k in range(n - 1, 0, -1) if self.params.seed_children else ():
            s, fine = self.stores[k], self.bands[k - 1]
            if not s.size:
                continue
            near, far = self._square_dist2(s)
            reach = near < fine.r_max ** 2
            if not reach.any():
                continue
            kids = split_into_children(store_take(s, reach), fine.resolution)
            kids = store_take(kids, touches_ring(key_centers(kids.keys, fine.resolution), fine.resolution,
                                                 self.sensor_xy, fine))
            self.stores[k - 1] = add_missing(self.stores[k - 1], kids)
            self.stores[k] = store_take(s, far >= fine.r_max ** 2)

    # add one filtered velodyne scan taken at time t with pose T_world_velo
    def integrate(self, points_velo: np.ndarray, t_world_velo: np.ndarray, t: float) -> None:
        if self.t is not None:
            self.stores = [decay_store(s, t - self.t, self.params) for s in self.stores]
            self.fine = decay_store(self.fine, t - self.t, self.params)
        self.t = t
        self.sensor_xy = t_world_velo[:2, 3].copy()
        self._handoff()
        if self.refine_on:
            self._refine_handoff(t)
        if len(points_velo) == 0:
            return
        world = velo_to_world(points_velo[:, :3], t_world_velo)
        r = np.hypot(points_velo[:, 0], points_velo[:, 1])
        cx, cy = self.sensor_xy
        # every point sits within r of sensor, so xy box is known: no min/max pass
        r_out = self.bands[-1].r_max
        terrain = local_terrain(world[r < r_out], "world", self.params.terrain_cell, self.params.ground_percentile,
                                bounds=(cx - r_out, cx + r_out, cy - r_out, cy + r_out))
        self.terrain = terrain
        for k, band in enumerate(self.bands):
            margin = band.resolution * np.sqrt(2.0)
            pts = world[(r >= band.r_min - margin) & (r < band.r_max + margin)]
            if k == len(self.bands) - 1:
                far_pts = pts
            if len(pts) == 0:
                continue
            reach = band.r_max + margin
            box = (cx - reach, cx + reach, cy - reach, cy + reach)
            obs = scan_observation(pts, band.resolution, self.params.ground_percentile, bounds=box)
            centers = key_centers(obs.keys, band.resolution)
            keep = touches_ring(centers, band.resolution, self.sensor_xy, band)
            obs = obs_take(obs, keep)
            tz = terrain.sample(centers[keep])
            self.stores[k] = integrate_obs(self.stores[k], obs, tz, t, self.params)
        if self.refine_on:
            self._scan_no = getattr(self, "_scan_no", -1) + 1
            if self._scan_no % self.params.refine_every == 0:
                self._refine_update(t)
            self._refine_ingest(far_pts, terrain, t)

    # feed outer-ring points that fall in refined patches into fine patch lattice (outer store already has them)
    # far_pts = outer ring points incl. its own (wider) edge margin, already picked by integrate
    def _refine_ingest(self, pts: np.ndarray, terrain: Terrain, t: float) -> None:
        if not self.ref_keys.size or not len(pts):
            return
        far, res = self.bands[-1], self.params.refine_res
        margin = res * np.sqrt(2.0)
        pk = encode_keys(np.floor(pts[:, 0] / far.resolution).astype(np.int64),
                         np.floor(pts[:, 1] / far.resolution).astype(np.int64))
        pos = np.minimum(np.searchsorted(self.ref_keys, pk), self.ref_keys.size - 1)
        pts = pts[self.ref_keys[pos] == pk]
        if not len(pts):
            return
        cx, cy = self.sensor_xy
        reach = far.r_max + margin
        obs = scan_observation(pts, res, self.params.ground_percentile, bounds=(cx - reach, cx + reach, cy - reach, cy + reach))
        centers = key_centers(obs.keys, res)
        keep = touches_ring(centers, res, self.sensor_xy, far)
        self.fine = integrate_obs(self.fine, obs_take(obs, keep), terrain.sample(centers[keep]), t, self.params)

    # rings around latest sensor as FoveatedLayers in world frame. unknown cells = NaN, count 0.
    def render(self) -> FoveatedLayers:
        cx, cy = self.sensor_xy
        out = []
        for band, s in zip(self.bands, self.stores):
            res = band.resolution
            spec = spec_from_bounds(cx - band.r_max, cx + band.r_max, cy - band.r_max, cy + band.r_max, res, "world")
            owned = annulus_mask(spec, band, (cx, cy))
            flat_ids = np.flatnonzero(owned.ravel())
            ix, iy = np.divmod(flat_ids, spec.ny)
            gkeys = encode_keys(ix + int(round(spec.x_min / res)), iy + int(round(spec.y_min / res)))
            pos = np.minimum(np.searchsorted(s.keys, gkeys), max(s.size - 1, 0))
            hit = (s.keys[pos] == gkeys) if s.size else np.zeros(gkeys.size, dtype=bool)
            dst, src = np.flatnonzero(hit), pos[hit]  # gather index computed once for all layers

            def pick(name: str, fill: float) -> np.ndarray:
                a = np.full(gkeys.size, fill)
                a[dst] = getattr(s, name)[src]
                return a

            weight = pick("weight", 0.0)
            count = np.zeros(gkeys.size, dtype=np.int64)
            count[dst] = s.n_obs[src]
            z_var = np.full(gkeys.size, np.nan)
            z_var[dst] = s.m2[src] / s.weight[src]
            out.append(BandLayers(
                band=band, spec=spec, flat_ids=flat_ids, count=count,
                z_min=pick("z_min", np.nan), z_max=pick("z_max", np.nan), z_mean=pick("mean", np.nan),
                z_var=z_var, ground=pick("ground", np.nan),
                extra={"weight": weight, "logodds": pick("logodds", 0.0), "last_seen": pick("last_seen", np.nan),
                       "inherited": pick("inherited", 0.0), "points": pick("points", np.nan)},
            ))
        if self.refine_on:
            out.append(self._render_patches())
        return FoveatedLayers(bands=out, frame="world", ground_percentile=self.params.ground_percentile,
                              center=(float(cx), float(cy)))

    # refined patch cells as overlay band on outer ring (compact: only stored patch cells)
    def _render_patches(self) -> BandLayers:
        far, res, f = self.bands[-1], self.params.refine_res, self.fine
        cx, cy = self.sensor_xy
        spec = spec_from_bounds(cx - far.r_max, cx + far.r_max, cy - far.r_max, cy + far.r_max, res, "world")
        gx, gy = decode_keys(f.keys)
        ix, iy = gx - int(round(spec.x_min / res)), gy - int(round(spec.y_min / res))
        inside = (ix >= 0) & (ix < spec.nx) & (iy >= 0) & (iy < spec.ny)
        s = store_take(f, inside)
        flat = ix[inside] * spec.ny + iy[inside]
        return BandLayers(
            band=Band(far.r_min, far.r_max, res), spec=spec, flat_ids=flat, count=s.n_obs.copy(),
            z_min=s.z_min, z_max=s.z_max, z_mean=s.mean, z_var=s.m2 / np.maximum(s.weight, 1e-12), ground=s.ground,
            extra={"weight": s.weight, "logodds": s.logodds, "last_seen": s.last_seen,
                   "inherited": s.inherited.astype(np.float64), "points": s.points},
            overlay=True,
        )
