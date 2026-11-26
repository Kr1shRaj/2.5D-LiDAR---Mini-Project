import numpy as np

from src.mapping.foveated_grid import BandLayers

# Replay frame format for the web viewer (viewer/). One frame = bands near to far, each band blob:
#   observed mask: packbits over dense lattice, flat = ix * ny + iy, bit order little, padded to 4 bytes
#   then one column per COLUMNS entry, values for set mask bits in flat order, each padded to 4 bytes
# All little endian. Quantized ints: value = int * scale. NaN = sentinel (signed: dtype min, unsigned: dtype max).
# Unobserved cell is never written: no mask bit, never a 0.0 height.

FORMAT = "fovreplay-1"
COLUMNS: tuple[tuple[str, str, float], ...] = (
    ("z_max", "<i2", 0.01),  # world z, m
    ("ground", "<i2", 0.01),  # world z, m
    ("terrain", "<i2", 0.01),  # local terrain under cell centre, world z, m
    ("rough", "<i2", 0.001),  # z std inside cell, m
    ("weight", "<u2", 0.01),  # decayed confidence
    ("n_obs", "<u2", 1.0),  # scans that saw cell
    ("logodds", "<i1", 0.05),  # obstacle evidence
    ("label", "<u1", 1.0),  # class id (src.models.targets)
    ("flags", "<u1", 1.0),  # FLAG_* bits
)
FLAG_INHERITED = 1  # placeholder child seeded from coarse parent
FLAG_MODEL = 2  # label from model (reaction zone), else threshold rule


# int sentinel that stands for NaN in this dtype
def nan_sentinel(dtype: str) -> int:
    info = np.iinfo(np.dtype(dtype))
    return int(info.min) if info.min < 0 else int(info.max)


# float array -> ints of dtype at scale. NaN -> sentinel. clip keeps sentinel free.
def quantize(a: np.ndarray, dtype: str, scale: float) -> np.ndarray:
    info, nan = np.iinfo(np.dtype(dtype)), nan_sentinel(dtype)
    lo, hi = (info.min + 1, info.max) if nan == info.min else (info.min, info.max - 1)
    q = np.clip(np.round(np.nan_to_num(a / scale, nan=0.0)), lo, hi).astype(dtype)
    q[np.isnan(a)] = nan
    return q


# ints of dtype at scale -> float, sentinel -> NaN
def dequantize(q: np.ndarray, dtype: str, scale: float) -> np.ndarray:
    out = q.astype(np.float64) * scale
    out[q == nan_sentinel(dtype)] = np.nan
    return out


# bytes rounded up to next multiple of 4
def pad4(n: int) -> int:
    return (n + 3) // 4 * 4


# blob size of one band
def band_nbytes(nx: int, ny: int, n_obs: int) -> int:
    return pad4((nx * ny + 7) // 8) + sum(pad4(n_obs * np.dtype(d).itemsize) for _, d, _ in COLUMNS)


# observed cells of band + one probe point per cell, inside both cell square and band annulus.
# FoveatedLayers.lookup(probe) returns this same cell, so features + labels read there = what map answers there.
# edge cell with centre outside annulus: probe = its square point nearest (centre past r_max) or
# farthest (centre inside r_min) from sensor, nudged into square. cells with no such point dropped.
def export_cells(bl: BandLayers, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sel = np.flatnonzero(bl.observed)
    c = cell_xy(bl, sel)
    lo = c - bl.spec.resolution / 2
    hi = lo + bl.spec.resolution
    s = np.array(center)
    near = np.clip(s, lo, hi)
    far = np.where(np.abs(lo - s) > np.abs(hi - s), lo, hi)
    r = np.hypot(*(c - s).T)[:, None]
    p = np.where(r < bl.band.r_min, far, np.where(r >= bl.band.r_max, near, c))
    p = p + (c - p) * 1e-3
    rp = np.hypot(*(p - s).T)
    ok = (rp >= bl.band.r_min) & (rp < bl.band.r_max)
    return sel[ok], p[ok]


# centre xy of compact cells, same frame as band lattice
def cell_xy(bl: BandLayers, sel: np.ndarray) -> np.ndarray:
    ix, iy = np.divmod(bl.flat_ids[sel], bl.spec.ny)
    res = bl.spec.resolution
    return np.column_stack([bl.spec.x_min + (ix + 0.5) * res, bl.spec.y_min + (iy + 0.5) * res])


# one band -> bytes. flat = dense lattice ids of written cells (ascending), cols = name -> float array
def encode_band(nx: int, ny: int, flat: np.ndarray, cols: dict[str, np.ndarray]) -> bytes:
    if flat.size and np.any(np.diff(flat) <= 0):
        raise ValueError("flat ids must be strictly ascending")
    mask = np.zeros(nx * ny, dtype=bool)
    mask[flat] = True
    parts = [np.packbits(mask, bitorder="little").tobytes()]
    for name, dtype, scale in COLUMNS:
        if len(cols[name]) != flat.size:
            raise ValueError(f"column {name}: {len(cols[name])} values for {flat.size} cells")
        parts.append(quantize(np.asarray(cols[name], dtype=np.float64), dtype, scale).tobytes())
    return b"".join(p + b"\0" * (pad4(len(p)) - len(p)) for p in parts)


# frame bytes -> per band dict: flat ids + dequantized columns. shapes = (nx, ny, n_obs) per band.
def decode_frame(buf: bytes, shapes: list[tuple[int, int, int]]) -> list[dict[str, np.ndarray]]:
    out, at = [], 0
    for nx, ny, n in shapes:
        nb = (nx * ny + 7) // 8
        mask = np.unpackbits(np.frombuffer(buf, np.uint8, nb, at), count=nx * ny, bitorder="little").astype(bool)
        at += pad4(nb)
        band = {"flat": np.flatnonzero(mask)}
        if band["flat"].size != n:
            raise ValueError(f"mask has {band['flat'].size} cells, header says {n}")
        for name, dtype, scale in COLUMNS:
            q = np.frombuffer(buf, np.dtype(dtype), n, at)
            band[name] = dequantize(q, dtype, scale)
            at += pad4(n * np.dtype(dtype).itemsize)
        out.append(band)
    if at != len(buf):
        raise ValueError(f"frame has {len(buf)} bytes, layout says {at}")
    return out


# Point blob (frames/NNNNNN.pts.gz), current scan only, every POINT_STRIDE-th filtered point, world frame:
#   u32 n, then i2 dx[n], i2 dy[n] (cm from sensor xy), i2 z[n] (world z, cm), i1 h[n] (height over local terrain,
#   2 cm, NaN sentinel -128 = no terrain there). each array padded to 4 bytes. little endian.
POINTS_FORMAT = "fovpoints-1"
POINT_STRIDE = 4
POINT_COLUMNS: tuple[tuple[str, str, float], ...] = (
    ("dx", "<i2", 0.01),
    ("dy", "<i2", 0.01),
    ("z", "<i2", 0.01),
    ("h", "<i1", 0.02),
)


# world points + height over terrain -> point blob. xy stored relative to sensor so int16 cm covers +-327 m
def encode_points(world: np.ndarray, sensor_xy: np.ndarray, h: np.ndarray) -> bytes:
    cols = {"dx": world[:, 0] - sensor_xy[0], "dy": world[:, 1] - sensor_xy[1], "z": world[:, 2], "h": h}
    parts = [np.array([len(world)], dtype="<u4").tobytes()]
    for name, dtype, scale in POINT_COLUMNS:
        parts.append(quantize(np.asarray(cols[name], dtype=np.float64), dtype, scale).tobytes())
    return b"".join(p + b"\0" * (pad4(len(p)) - len(p)) for p in parts)


# point blob -> name -> float array (dx, dy, z, h). h NaN where no terrain
def decode_points(buf: bytes) -> dict[str, np.ndarray]:
    n = int(np.frombuffer(buf, "<u4", 1, 0)[0])
    out, at = {}, 4
    for name, dtype, scale in POINT_COLUMNS:
        out[name] = dequantize(np.frombuffer(buf, np.dtype(dtype), n, at), dtype, scale)
        at += pad4(n * np.dtype(dtype).itemsize)
    if at != len(buf):
        raise ValueError(f"point blob has {len(buf)} bytes, layout says {at}")
    return out
