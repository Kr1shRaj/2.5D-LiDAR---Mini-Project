from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.mapping.foveated_grid import FoveatedLayers
from src.mapping.uniform_grid import GridLayers

UNOBSERVED_COLOR = "#bdbdbd"


# draw one layer top down. x right, y up. nan cells = grey unobserved.
def _draw_layer(
    ax: plt.Axes,
    layer: np.ndarray,
    layers: GridLayers,
    title: str,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    spec = layers.spec
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(UNOBSERVED_COLOR)
    # layer is [ix, iy]; imshow wants [row=y, col=x], so transpose and put origin low
    img = ax.imshow(
        np.ma.masked_invalid(layer.T),
        origin="lower",
        extent=(spec.x_min, spec.x_max, spec.y_min, spec.y_max),
        cmap=cm,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(f"x [m] ({spec.frame})")
    ax.set_ylabel(f"y [m] ({spec.frame})")
    ax.set_aspect("equal")
    ax.figure.colorbar(img, ax=ax, label=label, shrink=0.8)


# top down elevation + roughness panels. optional sensor path in grid frame. save png if path given.
def plot_elevation_and_roughness(
    layers: GridLayers,
    out_path: str | Path | None = None,
    path_xy: np.ndarray | None = None,
) -> Figure:
    fig, (ax_elev, ax_rough) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    # clip colour at p1..p99 so stray multipath points do not stretch scale
    lo, hi = (np.nanpercentile(layers.z_max, [1, 99]) if layers.observed.any() else (None, None))
    _draw_layer(ax_elev, layers.z_max, layers, "Max elevation (clip p1..p99)", "viridis", "z max [m]", vmin=lo, vmax=hi)
    # clip colour at p95 so few tall poles do not wash out curbs
    rough = layers.roughness
    rough_vmax = float(np.nanpercentile(rough, 95)) if layers.observed.any() else None
    _draw_layer(ax_rough, rough, layers, "Roughness (z std, clip p95)", "magma", "z std [m]", vmax=rough_vmax)
    if path_xy is not None:
        for ax in (ax_elev, ax_rough):
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="red", lw=1.2, label="sensor path")
        ax_elev.legend(loc="upper left", fontsize=8)
    observed = int(layers.observed.sum())
    total = layers.count.size
    fig.suptitle(
        f"Uniform 2.5D grid  |  {layers.spec.resolution:.2f} m cells  |  "
        f"observed {observed}/{total}  |  grey = unobserved"
    )
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig


# top down foveated map: z max per band (far drawn first) + cell size map. rings mark band edges.
def plot_foveated(fov: FoveatedLayers, out_path: str | Path | None = None, title: str = "") -> Figure:
    fig, (ax_elev, ax_res) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    all_zmax = np.concatenate([b.z_max for b in fov.bands])
    lo, hi = np.nanpercentile(all_zmax, [1, 99]) if np.isfinite(all_zmax).any() else (None, None)
    elev_cm = plt.get_cmap("viridis").copy()
    elev_cm.set_bad((0, 0, 0, 0))
    res_cm = plt.get_cmap("cividis", len(fov.bands))
    r_out = fov.bands[-1].band.r_max
    ax_elev.set_facecolor(UNOBSERVED_COLOR)
    ax_res.set_facecolor("white")
    img = None
    for k, bl in reversed(list(enumerate(fov.bands))):
        s = bl.spec
        ext = (s.x_min, s.x_max, s.y_min, s.y_max)
        # mask cells outside annulus so inner band shows through
        inside = np.isfinite(bl.dense("count"))
        zmax = np.where(inside, bl.dense("z_max"), np.nan)
        img = ax_elev.imshow(np.ma.masked_invalid(zmax.T), origin="lower", extent=ext, cmap=elev_cm,
                             interpolation="nearest", vmin=lo, vmax=hi)
        band_id = np.where(inside, float(k), np.nan)
        ax_res.imshow(np.ma.masked_invalid(band_id.T), origin="lower", extent=ext, cmap=res_cm,
                      interpolation="nearest", vmin=-0.5, vmax=len(fov.bands) - 0.5)
    cx, cy = fov.center
    for ax in (ax_elev, ax_res):
        for bl in fov.bands:
            ax.add_patch(plt.Circle((cx, cy), bl.band.r_max, fill=False, color="red", lw=0.8))
        ax.set_xlim(cx - r_out, cx + r_out)
        ax.set_ylim(cy - r_out, cy + r_out)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x [m] ({fov.frame})")
        ax.set_ylabel(f"y [m] ({fov.frame})")
    fig.colorbar(img, ax=ax_elev, label="z max [m]", shrink=0.8)
    ax_elev.set_title("Max elevation per band (grey = unobserved)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=res_cm(k)) for k in range(len(fov.bands))]
    labels = [f"{b.band.r_min:g}-{b.band.r_max:g} m @ {b.band.resolution * 100:g} cm" for b in fov.bands]
    ax_res.legend(handles, labels, loc="upper left", fontsize=8)
    ax_res.set_title("Cell size by band")
    fig.suptitle(title or f"Foveated 2.5D grid  |  {fov.n_cells} cells  |  {fov.nbytes / 1e6:.2f} MB")
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig


# side by side class maps. grids indexed [ix, iy]; -1 = never seen (white).
def plot_class_maps(
    grids: list[np.ndarray],
    titles: list[str],
    extent: tuple[float, float, float, float],
    class_names: list[str],
    out_path: str | Path | None = None,
    title: str = "",
) -> Figure:
    colors = ["#4daf4a", "#ff7f00", "#e41a1c", "#9e9e9e"]  # traversable, hazardous, obstacle, unknown
    cm = matplotlib.colors.ListedColormap(colors[: len(class_names)])
    cm.set_bad("white")
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 6), constrained_layout=True)
    for ax, g, t in zip(np.atleast_1d(axes), grids, titles):
        ax.imshow(np.ma.masked_less(g.T.astype(float), 0), origin="lower", extent=extent, cmap=cm,
                  vmin=-0.5, vmax=len(class_names) - 0.5, interpolation="nearest")
        ax.set_title(t)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m] (world)")
        ax.set_ylabel("y [m] (world)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors[: len(class_names)]]
    fig.legend(handles, class_names, loc="outside lower center", ncol=len(class_names))
    fig.suptitle(title)
    if out_path is not None:
        fig.savefig(out_path, dpi=110)
    return fig
