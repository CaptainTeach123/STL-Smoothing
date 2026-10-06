"""Before / after pictures of what a slicer would do with a model.

Faces that were flattened are coloured by the layer they end on; every slicer
layer edge (the contour lines you see on a printed surface) is drawn on top.
A strangely offset set of layer lines shows up as many colours and a tangle of
orange lines; a fixed surface is one colour with no lines through it.

Requires matplotlib (``pip install stl-smoothing[report]``).
"""

from __future__ import annotations

import numpy as np

from .layers import LayerGrid
from .mesh import Mesh
from .slicing import contour_segments


def _import_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Rendering reports needs matplotlib: pip install 'stl-smoothing[report]'"
        ) from exc
    return plt, LineCollection, PolyCollection, Patch, Line2D


def region_layers(mesh: Mesh, grid: LayerGrid, faces: np.ndarray, min_frac: float = 0.002):
    """Layer ends (sorted) that the selected faces occupy, ignoring slivers."""
    if not faces.any():
        return np.zeros(0)
    _, area = mesh.face_normals_areas()
    zc = mesh.face_centroids()[:, 2]
    ends = np.round(grid.layer_end(zc[faces]), 6)
    ids, inv = np.unique(ends, return_inverse=True)
    w = np.bincount(inv, weights=area[faces])
    return ids[w > min_frac * w.sum()]


def _describe(levels: np.ndarray, what: str) -> str:
    if len(levels) == 0:
        return f"No {what} found"
    if len(levels) == 1:
        return f"{what} end on one layer ({levels[0]:.2f} mm)"
    return f"{what} end on {len(levels)} different layers ({levels[0]:.2f}–{levels[-1]:.2f} mm)"


def render_comparison(
    before: Mesh,
    after: Mesh,
    grid: LayerGrid,
    region: np.ndarray,
    out_path: str,
    *,
    view: str = "top",
    what: str = "Smoothed surfaces",
    footnote: str | None = None,
    dpi: int = 110,
) -> dict:
    """Render ``before`` and ``after`` side by side and save a PNG.

    ``region`` is a boolean face mask (same faces in both meshes) selecting the
    surfaces to colour.  ``view`` is ``"top"`` (up-facing) or ``"bottom"``.
    Returns a small dict with the layer ends found before and after.
    """
    plt, LineCollection, PolyCollection, Patch, Line2D = _import_mpl()
    sign = 1.0 if view == "top" else -1.0
    lv_before = region_layers(before, grid, region)
    lv_after = region_layers(after, grid, region)
    all_levels = np.unique(np.concatenate([lv_before, lv_after]))
    cmap = plt.get_cmap("coolwarm")
    colour_of = {
        lv: cmap(0.5 if len(all_levels) == 1 else i / (len(all_levels) - 1))
        for i, lv in enumerate(all_levels)
    }
    lo, hi = before.verts[:, :2].min(0), before.verts[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-9)
    fig_h = 7.0
    fig_w = 2 * fig_h * span[0] / span[1]
    fig, axes = plt.subplots(1, 2, figsize=(max(fig_w, 8) + 0.5, fig_h + 1.2), dpi=dpi)
    for ax, mesh, title, levels in (
        (axes[0], before, "Before", lv_before),
        (axes[1], after, "After", lv_after),
    ):
        n, _ = mesh.face_normals_areas()
        facing = sign * n[:, 2] > 0.02
        idx = np.flatnonzero(facing)
        zc = mesh.face_centroids()[:, 2]
        order = idx[np.argsort(sign * zc[idx])]  # far to near
        shade = 0.40 + 0.35 * np.clip(sign * n[order, 2], 0, 1)
        colours = np.stack([shade, shade, shade, np.ones_like(shade)], axis=1)
        in_region = region[order]
        if in_region.any():
            ends = np.round(grid.layer_end(zc[order][in_region]), 6)
            colours[in_region] = [colour_of.get(e, (0.5, 0.5, 0.5, 1.0)) for e in ends]
        xy = mesh.verts[mesh.faces[order]][:, :, :2]
        ax.add_collection(PolyCollection(xy, facecolors=colours, edgecolors="none", linewidths=0))
        a, b, _ = contour_segments(mesh, grid, facing)
        if len(a):
            segs = np.stack([a[:, :2], b[:, :2]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#f97316", linewidths=0.35))
        ax.set_xlim(lo[0] - 0.02 * span[0], hi[0] + 0.02 * span[0])
        ax.set_ylim(lo[1] - 0.02 * span[1], hi[1] + 0.02 * span[1])
        ax.set_aspect("equal")
        if view == "bottom":
            ax.invert_xaxis()
        ax.axis("off")
        ax.set_title(f"{title}\n{_describe(levels, what)}", loc="left", fontsize=11)
    handles = [
        Patch(facecolor=colour_of[lv], label=f"{lv:.2f} mm") for lv in all_levels
    ]
    handles.append(Line2D([0], [0], color="#f97316", lw=1.2, label="layer edge (a contour line in the slicer)"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 8),
        frameon=False,
        title=f"Layer the surface prints on, at {grid.layer_height:g} mm layers",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035 if footnote else 0.0),
    )
    if footnote:
        fig.text(0.5, 0.012, footnote, ha="center", fontsize=8, color="#666666")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14 if footnote else 0.12, wspace=0.04)
    fig.savefig(out_path)
    plt.close(fig)
    return {"before_layers": lv_before.tolist(), "after_layers": lv_after.tolist()}
