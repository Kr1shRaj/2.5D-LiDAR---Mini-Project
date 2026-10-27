from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
