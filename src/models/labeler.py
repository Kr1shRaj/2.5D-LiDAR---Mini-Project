import numpy as np

from src.mapping.foveated_grid import BandLayers, FoveatedLayers
from src.models.features import cell_features, feature_matrix
from src.temporal.fusion import encode_keys
from src.temporal.terrain import Terrain


# world keys + centre xy of selected cells of one rendered band (world-aligned lattice)
def band_cells(bl: BandLayers, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = bl.spec.resolution
    ix, iy = np.divmod(bl.flat_ids[sel], bl.spec.ny)
    gx = ix + int(round(bl.spec.x_min / res))
    gy = iy + int(round(bl.spec.y_min / res))
    return encode_keys(gx, gy), np.column_stack([(gx + 0.5) * res, (gy + 0.5) * res])


# incremental hazard labeler for streaming map. model labels reaction zone (r < zone_m) only.
# re-label cell when: no label yet (new, handed off, seeded), or this scan touched it and
#   - obstacle evidence flipped sign, or z max moved more than dz_m since last label (event), or
#   - its label is refresh_s old (periodic).
# other cells keep last label. outside zone: no model label, caller falls back to threshold rule.
class HazardLabeler:
    def __init__(self, model, zone_m: float, refresh_s: float = 0.3, dz_m: float = 0.1) -> None:
        self.model = model
        self.zone_m = zone_m
        self.refresh_s = refresh_s
        self.dz_m = dz_m
        # band index -> sorted keys, labels, label time, z max at label, obstacle sign at label
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    # refresh labels after map integrate + render. returns number of cells run through model.
    def update(self, fov: FoveatedLayers, terrain: Terrain, t: float) -> int:
        cx, cy = fov.center
        todo = []
        for k, bl in enumerate(fov.bands):
            if bl.band.r_min >= self.zone_m:
                self.cache.pop(k, None)
                continue
            obs = np.flatnonzero(bl.observed)
            keys, xy = band_cells(bl, obs)
            inzone = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < self.zone_m
            obs, keys, xy = obs[inzone], keys[inzone], xy[inzone]
            zmax, occ = bl.z_max[obs], bl.layer("logodds")[obs] > 0
            old = self.cache.get(k)
            cur = {"keys": keys, "label": np.full(keys.size, -1, dtype=np.int64), "t": np.full(keys.size, t),
                   "zmax": zmax.copy(), "occ": occ.copy()}
            known = np.zeros(keys.size, dtype=bool)
            if old is not None and old["keys"].size:
                pos = np.minimum(np.searchsorted(old["keys"], keys), old["keys"].size - 1)
                known = old["keys"][pos] == keys
                for n in ("label", "t", "zmax", "occ"):
                    cur[n][known] = old[n][pos[known]]
            touched = bl.layer("last_seen")[obs] == t
            event = (np.abs(zmax - cur["zmax"]) > self.dz_m) | (occ != cur["occ"])
            stale = t - cur["t"] >= self.refresh_s - 1e-6
            need = ~known | (touched & (event | stale))
            cur["t"][need], cur["zmax"][need], cur["occ"][need] = t, zmax[need], occ[need]
            todo.append((k, cur, need, xy[need]))
        xy_all = np.vstack([d[3] for d in todo]) if todo else np.empty((0, 2))
        if len(xy_all):
            f = cell_features(fov.sample_many, xy_all, terrain.sample(xy_all), np.array([cx, cy]), t)
            pred = self.model.predict(feature_matrix(f))
            at = 0
            for k, cur, need, xy in todo:
                cur["label"][need] = pred[at : at + len(xy)]
                at += len(xy)
        for k, cur, _, _ in todo:
            self.cache[k] = cur
        return len(xy_all)

    # cached model label at query xy. -1 = no model label (outside zone or cell unobserved).
    def labels_at(self, fov: FoveatedLayers, xy: np.ndarray) -> np.ndarray:
        band_idx, cell_idx = fov.lookup(xy)
        out = np.full(len(xy), -1, dtype=np.int64)
        for k, bl in enumerate(fov.bands):
            if k not in self.cache:
                continue
            sel = np.flatnonzero(band_idx == k)
            keys, _ = band_cells(bl, cell_idx[sel])
            ck, cl = self.cache[k]["keys"], self.cache[k]["label"]
            if ck.size == 0:
                continue
            pos = np.minimum(np.searchsorted(ck, keys), ck.size - 1)
            hit = ck[pos] == keys
            out[sel[hit]] = cl[pos[hit]]
        return out
