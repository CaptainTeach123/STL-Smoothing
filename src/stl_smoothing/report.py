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
from .preview import EDGE_COLOUR, describe_levels, level_colours, nearest_level, region_levels
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
    before_text: str | None = None,
    after_text: str | None = None,
) -> dict:
    """Render ``before`` and ``after`` side by side and save a PNG.

    ``region`` is a boolean face mask (same faces in both meshes) selecting the
    surfaces to colour.  ``view`` is ``"top"`` (up-facing) or ``"bottom"``.
    Returns a small dict with the layer ends found before and after.
    """
    plt, LineCollection, PolyCollection, Patch, Line2D = _import_mpl()
    sign = 1.0 if view == "top" else -1.0
    lv_before = region_levels(before, grid, region)
    lv_after = region_levels(after, grid, region)
    all_levels = np.unique(np.concatenate([lv_before, lv_after]))
    from matplotlib.colors import to_rgba

    colour_of = {lv: to_rgba(c) for lv, c in level_colours(all_levels, lv_after).items()}
    lo, hi = before.verts[:, :2].min(0), before.verts[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-9)
    fig_h = 7.0
    fig_w = 2 * fig_h * span[0] / span[1]
    # very long thin parts would give an absurdly wide picture: cap the aspect ratio
    fig_w = float(np.clip(fig_w, 8.0, 3.0 * fig_h * 2))
    fig, axes = plt.subplots(1, 2, figsize=(fig_w + 0.5, fig_h + 1.2), dpi=dpi)
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
            keys = list(colour_of)
            colours[in_region] = [colour_of[keys[k]] for k in nearest_level(np.array(keys), ends)]
        xy = mesh.verts[mesh.faces[order]][:, :, :2]
        ax.add_collection(PolyCollection(xy, facecolors=colours, edgecolors="none", linewidths=0))
        a, b, _ = contour_segments(mesh, grid, facing)
        if len(a):
            segs = np.stack([a[:, :2], b[:, :2]], axis=1)
            ax.add_collection(LineCollection(segs, colors=EDGE_COLOUR, linewidths=0.35))
        ax.set_xlim(lo[0] - 0.02 * span[0], hi[0] + 0.02 * span[0])
        ax.set_ylim(lo[1] - 0.02 * span[1], hi[1] + 0.02 * span[1])
        ax.set_aspect("equal")
        if view == "bottom":
            ax.invert_xaxis()
        ax.axis("off")
        text = before_text if title == "Before" else after_text
        ax.set_title(f"{title}\n{text or describe_levels(levels, what)}", loc="left", fontsize=11)
    handles = [
        Patch(facecolor=colour_of[lv], label=f"{lv:.2f} mm") for lv in all_levels
    ]
    handles.append(Line2D([0], [0], color=EDGE_COLOUR, lw=1.2, label="layer edge (a contour line in the slicer)"))
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
