from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.mapping.foveated_grid import Band, BandLayers, FoveatedLayers
from src.mapping.uniform_grid import GridSpec
from src.temporal.terrain import Terrain

# Frozen copy of one rendered fused map + its local terrain, so model work can run on a fixed map version
# while fusion code keeps changing. One .npz per scan. Arrays saved as is: NaN stays NaN, never 0.0.

CORE = ("count", "z_min", "z_max", "z_mean", "z_var", "ground")


# GridSpec -> flat dict of plain numbers
def _spec(prefix: str, s: GridSpec) -> dict:
    return {f"{prefix}{k}": np.array(v) for k, v in asdict(s).items()}


# flat dict -> GridSpec
def _unspec(z, prefix: str) -> GridSpec:
    return GridSpec(x_min=float(z[f"{prefix}x_min"]), y_min=float(z[f"{prefix}y_min"]),
                    resolution=float(z[f"{prefix}resolution"]), nx=int(z[f"{prefix}nx"]), ny=int(z[f"{prefix}ny"]),
                    frame=str(z[f"{prefix}frame"]))


# write rendered map + terrain + meta (map version, scan id, time, pose) to one npz
def save_snapshot(path: Path, fov: FoveatedLayers, terrain: Terrain, meta: dict) -> None:
    out = {"n_bands": np.array(len(fov.bands)), "frame": np.array(fov.frame),
           "ground_percentile": np.array(fov.ground_percentile), "center": np.array(fov.center),
           **_spec("terrain_", terrain.spec), "terrain_z": terrain.z,
           **{f"meta_{k}": np.array(v) for k, v in meta.items()}}
    for k, bl in enumerate(fov.bands):
        p = f"b{k}_"
        out |= {f"{p}r_min": np.array(bl.band.r_min), f"{p}r_max": np.array(bl.band.r_max),
                f"{p}band_res": np.array(bl.band.resolution), f"{p}flat_ids": bl.flat_ids, **_spec(f"{p}spec_", bl.spec),
                f"{p}overlay": np.array(bl.overlay)}
        out |= {f"{p}{n}": getattr(bl, n) for n in CORE}
        out |= {f"{p}extra_{n}": a for n, a in bl.extra.items()}
    np.savez_compressed(path, **out)


# npz -> (FoveatedLayers, Terrain, meta). same lookups and layers as live map.
def load_snapshot(path: Path) -> tuple[FoveatedLayers, Terrain, dict]:
    with np.load(path) as z:
        bands = []
        for k in range(int(z["n_bands"])):
            p = f"b{k}_"
            band = Band(float(z[f"{p}r_min"]), float(z[f"{p}r_max"]), float(z[f"{p}band_res"]))
            extra = {n[len(p) + 6:]: z[n] for n in z.files if n.startswith(f"{p}extra_")}
            overlay = bool(z[f"{p}overlay"]) if f"{p}overlay" in z.files else False
            bands.append(BandLayers(band=band, spec=_unspec(z, f"{p}spec_"), flat_ids=z[f"{p}flat_ids"],
                                    **{n: z[f"{p}{n}"] for n in CORE}, extra=extra, overlay=overlay))
        fov = FoveatedLayers(bands=bands, frame=str(z["frame"]), ground_percentile=float(z["ground_percentile"]),
                             center=tuple(float(v) for v in z["center"]))
        terrain = Terrain(spec=_unspec(z, "terrain_"), z=z["terrain_z"])
        meta = {n[5:]: z[n].item() if z[n].ndim == 0 else z[n] for n in z.files if n.startswith("meta_")}
    return fov, terrain, meta
