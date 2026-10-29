import numpy as np
import pytest

from src.mapping.foveated_grid import Band, build_foveated_grid
from src.viz.replay import (
    COLUMNS,
    band_nbytes,
    decode_frame,
    dequantize,
    encode_band,
    export_cells,
    nan_sentinel,
    quantize,
)

GROUND_Z = -1.73
BANDS = (Band(0.0, 4.0, 0.1), Band(4.0, 10.0, 0.25))


# seeded road disk to 8 m: outer band only partly seen, so unobserved cells exist
@pytest.fixture
def fov():
    rng = np.random.default_rng(seed=11)
    n = 30000
    r = 8.0 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([r * np.cos(a), r * np.sin(a), GROUND_Z + rng.normal(0, 0.01, n)])
    return build_foveated_grid(pts, BANDS, "velo", "velo", ground_percentile=10.0)


def test_quantize_keeps_nan_apart_from_zero() -> None:
    for _, dtype, scale in COLUMNS:
        a = np.array([np.nan, 0.0, scale, -scale if np.iinfo(np.dtype(dtype)).min < 0 else 2 * scale])
        q = quantize(a, dtype, scale)
        assert q[0] == nan_sentinel(dtype)
        back = dequantize(q, dtype, scale)
        assert np.isnan(back[0]) and not np.isnan(back[1:]).any()
        assert back[1] == 0.0


def test_quantize_clip_never_hits_sentinel() -> None:
    q = quantize(np.array([-1e9, 1e9]), "<i2", 0.01)
    assert not np.any(q == nan_sentinel("<i2"))
    q = quantize(np.array([-1e9, 1e9]), "<u2", 0.01)
    assert not np.any(q == nan_sentinel("<u2"))


def test_band_round_trip_and_layout() -> None:
    rng = np.random.default_rng(seed=5)
    shapes, blobs, truth = [], [], []
    for nx, ny, n in ((7, 9, 20), (5, 5, 0), (13, 3, 39)):
        flat = np.sort(rng.choice(nx * ny, n, replace=False))
        cols = {"z_max": rng.uniform(-3, 3, n), "ground": rng.uniform(-3, 3, n), "terrain": rng.uniform(-3, 3, n),
                "rough": rng.uniform(0, 0.3, n), "weight": rng.uniform(0, 30, n), "n_obs": rng.integers(1, 300, n),
                "logodds": rng.uniform(-2, 3.5, n), "label": rng.integers(0, 4, n), "flags": rng.integers(0, 4, n)}
        if n:
            cols["z_max"][0] = np.nan  # observed cell with no height must stay NaN
        blob = encode_band(nx, ny, flat, cols)
        assert len(blob) == band_nbytes(nx, ny, n) and len(blob) % 4 == 0
        shapes.append((nx, ny, n))
        blobs.append(blob)
        truth.append((flat, cols))
    out = decode_frame(b"".join(blobs), shapes)
    for band, (flat, cols) in zip(out, truth):
        np.testing.assert_array_equal(band["flat"], flat)
        for name, _, scale in COLUMNS:
            np.testing.assert_allclose(band[name], cols[name], atol=scale / 2 + 1e-9)
    assert np.isnan(out[0]["z_max"][0])


def test_encode_rejects_bad_input() -> None:
    cols = {name: np.zeros(2) for name, _, _ in COLUMNS}
    with pytest.raises(ValueError):
        encode_band(4, 4, np.array([3, 1]), cols)
    with pytest.raises(ValueError):
        encode_band(4, 4, np.array([1]), cols)


# probe point: lookup returns same band + same cell. edge cells with centre outside ring still exported.
def test_export_cells_probe_hits_own_cell(fov) -> None:
    edge = 0
    for k, bl in enumerate(fov.bands):
        sel, probe = export_cells(bl, fov.center)
        assert bl.observed[sel].all()
        assert sel.size >= 0.99 * bl.observed.sum()  # almost no observed cell lost
        band_idx, cell_idx = fov.lookup(probe)
        np.testing.assert_array_equal(band_idx, k)
        np.testing.assert_array_equal(cell_idx, sel)
        ix, iy = np.divmod(bl.flat_ids[sel], bl.spec.ny)
        c = np.column_stack([bl.spec.x_min + (ix + 0.5) * bl.spec.resolution, bl.spec.y_min + (iy + 0.5) * bl.spec.resolution])
        r = np.hypot(*c.T)
        edge += np.sum((r < bl.band.r_min) | (r >= bl.band.r_max))
    assert edge > 0  # ring edge cells with centre outside own ring are covered too
    # outer band: disk ends at 8 m, ring 8-10 m stays unobserved, never exported
    sel, probe = export_cells(fov.bands[1], fov.center)
    assert np.hypot(*probe.T).max() < 8.3
