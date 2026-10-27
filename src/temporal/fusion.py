from dataclasses import dataclass, fields, replace

import numpy as np

from src.mapping.foveated_grid import Band, BandLayers, FoveatedLayers, annulus_mask, validate_bands
from src.mapping.uniform_grid import build_uniform_grid, spec_from_bounds
from src.preprocess.transforms import velo_to_world
from src.temporal.terrain import local_terrain

# Streaming foveated map in WORLD frame. Cells sit on world-aligned lattices (one per band
# resolution), so vehicle rotation never resamples a cell. Rings follow the sensor.
# Each point goes to the band picked by its range from the sensor at capture time.
# Each scan counts as one observation per cell (weight 1); old weight decays with time.
# ground = decayed weighted mean of per-scan cell ground (per-scan ground = p-th percentile, as phase 1).
# Obstacle evidence = log-odds per cell: hit if scan z max rises obstacle_h over local terrain.
# Hand-off out: cell whose centre leaves its ring outward merges into next coarser lattice.
# Hand-off in: coarse cell whose centre enters finer ring splits into fine children that copy
# parent stats, flagged `inherited`. First real fine scan of child replaces copied stats.

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
        "inherited": False}
DTYPES = {"keys": np.int64, "n_obs": np.int64, "inherited": bool}


# empty store for one lattice
def empty_store(resolution: float) -> CellStore:
    arrays = {n: np.empty(0, dtype=DTYPES.get(n, np.float64)) for n in ARRAY_FIELDS}
    return CellStore(resolution=resolution, **arrays)


# keep only rows where mask true
def store_take(store: CellStore, mask: np.ndarray) -> CellStore:
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
    g = build_uniform_grid(points_world, spec, points_frame="world", ground_percentile=ground_percentile)
    ix, iy = np.nonzero(g.observed)
    keys = encode_keys(ix + int(round(spec.x_min / resolution)), iy + int(round(spec.y_min / resolution)))
    order = np.argsort(keys)
    ix, iy = ix[order], iy[order]
    return ScanObs(keys=keys[order], n=g.count[ix, iy], mean=g.z_mean[ix, iy], var=g.z_var[ix, iy],
                   z_min=g.z_min[ix, iy], z_max=g.z_max[ix, iy], ground=g.ground[ix, iy])


# fold one scan observation (weight 1 per cell) into store. terrain_z = local terrain under each obs cell.
def integrate_obs(store: CellStore, obs: ScanObs, terrain_z: np.ndarray, t: float, params: FusionParams) -> CellStore:
    keys = np.union1d(store.keys, obs.keys)
    new = {n: np.full(keys.size, FILL[n], dtype=getattr(store, n).dtype) for n in ARRAY_FIELDS}
    si = np.searchsorted(keys, store.keys)
    for n in ARRAY_FIELDS:
        new[n][si] = getattr(store, n)
    new["keys"] = keys
    oi = np.searchsorted(keys, obs.keys)

    # inherited placeholder meets first real fine scan: forget copied stats, start fresh
    seen = (new["n_obs"][oi] > 0) & ~new["inherited"][oi]
    w0 = np.where(seen, new["weight"][oi], 0.0)
    m0 = np.where(seen, new["mean"][oi], 0.0)
    g0 = np.where(seen, new["ground"][oi], 0.0)
    m2_0 = np.where(seen, new["m2"][oi], 0.0)
    w1 = w0 + 1.0
    # weighted merge of old cell (weight w0) with this scan (weight 1)
    new["mean"][oi] = (w0 * m0 + obs.mean) / w1
    new["m2"][oi] = m2_0 + obs.var + (obs.mean - m0) ** 2 * w0 / w1
    new["ground"][oi] = (w0 * g0 + obs.ground) / w1
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
    order = np.argsort(keys)
    return CellStore(resolution=fine_res, keys=keys[order], **{n: a[order] for n, a in out.items()})


# add cells to store; key already in store keeps existing row (real data beats placeholder)
def add_missing(store: CellStore, extra: CellStore) -> CellStore:
    new = store_take(extra, ~np.isin(extra.keys, store.keys))
    if new.size == 0:
        return store
    keys = np.concatenate([store.keys, new.keys])
    order = np.argsort(keys)
    return CellStore(resolution=store.resolution,
                     **{n: np.concatenate([getattr(store, n), getattr(new, n)])[order] for n in ARRAY_FIELDS})


# streaming foveated map. feed scans in time order with pose; render rings around latest sensor.
class TemporalFoveatedMap:
    def __init__(self, bands: tuple[Band, ...], params: FusionParams) -> None:
        validate_bands(bands)
        self.bands = tuple(bands)
        self.params = params
        self.stores = [empty_store(b.resolution) for b in bands]
        self.t: float | None = None
        self.sensor_xy = np.zeros(2)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.stores)

    @property
    def n_cells(self) -> int:
        return sum(s.size for s in self.stores)

    # centre distance of each store cell to sensor
    def _dist(self, s: CellStore) -> np.ndarray:
        return np.hypot(*(key_centers(s.keys, s.resolution) - self.sensor_xy).T) if s.size else np.empty(0)

    # out: cells past ring edge (+ one coarse cell slack, stops split/merge flip-flop) merge into coarser store.
    # last ring: drop (out of local map). in: coarse cells inside finer ring split into inherited children.
    def _handoff(self) -> None:
        n = len(self.bands)
        for k, band in enumerate(self.bands):
            s = self.stores[k]
            slack = self.bands[k + 1].resolution if k + 1 < n else 0.0
            leave = self._dist(s) >= band.r_max + slack
            if not leave.any():
                continue
            self.stores[k] = store_take(s, ~leave)
            if k + 1 < n:
                self.stores[k + 1] = merge_children(self.stores[k + 1], store_take(s, leave))
        for k in range(n - 1, 0, -1) if self.params.seed_children else ():
            s = self.stores[k]
            enter = self._dist(s) < self.bands[k - 1].r_max
            if not enter.any():
                continue
            self.stores[k] = store_take(s, ~enter)
            kids = split_into_children(store_take(s, enter), self.bands[k - 1].resolution)
            self.stores[k - 1] = add_missing(self.stores[k - 1], kids)

    # add one filtered velodyne scan taken at time t with pose T_world_velo
    def integrate(self, points_velo: np.ndarray, t_world_velo: np.ndarray, t: float) -> None:
        if self.t is not None:
            self.stores = [decay_store(s, t - self.t, self.params) for s in self.stores]
        self.t = t
        self.sensor_xy = t_world_velo[:2, 3].copy()
        self._handoff()
        if len(points_velo) == 0:
            return
        world = velo_to_world(points_velo[:, :3], t_world_velo)
        r = np.hypot(points_velo[:, 0], points_velo[:, 1])
        cx, cy = self.sensor_xy
        # every point sits within r of sensor, so xy box is known: no min/max pass
        r_out = self.bands[-1].r_max
        terrain = local_terrain(world[r < r_out], "world", self.params.terrain_cell, self.params.ground_percentile,
                                bounds=(cx - r_out, cx + r_out, cy - r_out, cy + r_out))
        for k, band in enumerate(self.bands):
            pts = world[(r >= band.r_min) & (r < band.r_max)]
            if len(pts) == 0:
                continue
            box = (cx - band.r_max, cx + band.r_max, cy - band.r_max, cy + band.r_max)
            obs = scan_observation(pts, band.resolution, self.params.ground_percentile, bounds=box)
            tz = terrain.sample(key_centers(obs.keys, band.resolution))
            self.stores[k] = integrate_obs(self.stores[k], obs, tz, t, self.params)

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

            def pick(name: str, fill: float) -> np.ndarray:
                a = np.full(gkeys.size, fill)
                a[hit] = getattr(s, name)[pos[hit]]
                return a

            weight = pick("weight", 0.0)
            count = np.zeros(gkeys.size, dtype=np.int64)
            count[hit] = s.n_obs[pos[hit]]
            z_var = np.full(gkeys.size, np.nan)
            z_var[hit] = s.m2[pos[hit]] / s.weight[pos[hit]]
            out.append(BandLayers(
                band=band, spec=spec, flat_ids=flat_ids, count=count,
                z_min=pick("z_min", np.nan), z_max=pick("z_max", np.nan), z_mean=pick("mean", np.nan),
                z_var=z_var, ground=pick("ground", np.nan),
                extra={"weight": weight, "logodds": pick("logodds", 0.0), "last_seen": pick("last_seen", np.nan),
                       "inherited": pick("inherited", 0.0)},
            ))
        return FoveatedLayers(bands=out, frame="world", ground_percentile=self.params.ground_percentile,
                              center=(float(cx), float(cy)))
