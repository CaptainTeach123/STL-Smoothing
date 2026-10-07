"""Data for the before/after picture, without any plotting library.

The command line draws it with matplotlib (:mod:`report`); the web page draws the same
picture on a canvas from the arrays built here.  The colour scheme is shared, so both
look the same: the layer a surface ends on is blue (lower) -> pale blue (the level it was
moved to) -> peach -> red (higher), and every slicer layer edge is an orange line.
"""

from __future__ import annotations

import numpy as np

from .flatten import FlattenResult
from .layers import LayerGrid
from .mesh import Mesh
from .slicing import contour_segments
from .summary import flattened_indices

PALETTE = [
    (0.0, "#4b63d1"),
    (0.2, "#8aa6f2"),
    (0.4, "#c4d3f5"),
    (0.55, "#f2d3c4"),
    (0.75, "#e8826a"),
    (1.0, "#b8232f"),
]
EDGE_COLOUR = "#f97316"
MAX_PREVIEW_FACES = 1_500_000


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def palette_hex(t: float) -> str:
    """Colour at position ``t`` (0..1) of the palette."""
    xs = [p[0] for p in PALETTE]
    cols = np.stack([_rgb(p[1]) for p in PALETTE])
    c = [np.interp(t, xs, cols[:, k]) for k in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def level_colours(levels: np.ndarray, final: np.ndarray) -> dict:
    """``{level: "#rrggbb"}``: the level the surfaces end up on (``final``) is pale blue."""
    levels = np.asarray(levels, dtype=np.float64)
    n = len(levels)
    if n == 0:
        return {}
    ref = int(np.argmin(np.abs(levels - (float(np.median(final)) if len(final) else float(np.median(levels))))))
    out = {}
    for i, lv in enumerate(levels):
        if i == ref:
            t = 0.4
        elif i < ref:
            t = 0.4 * i / ref
        else:
            t = 0.4 + 0.6 * (i - ref) / max(n - 1 - ref, 1)
        out[float(lv)] = palette_hex(t)
    return out


def region_levels(mesh: Mesh, grid: LayerGrid, faces: np.ndarray, min_frac: float = 0.005) -> np.ndarray:
    """Layer ends (sorted) that the selected faces occupy, ignoring slivers."""
    if not faces.any():
        return np.zeros(0)
    _, area = mesh.face_normals_areas()
    zc = mesh.face_centroids()[:, 2]
    ends = np.round(grid.layer_end(zc[faces]), 6)
    ids, inv = np.unique(ends, return_inverse=True)
    w = np.bincount(inv, weights=area[faces])
    return ids[w > min_frac * w.sum()]


def nearest_level(levels: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the closest entry of the sorted ``levels`` for every value of ``ends``."""
    pos = np.clip(np.searchsorted(levels, ends), 0, len(levels) - 1)
    left = np.clip(pos - 1, 0, len(levels) - 1)
    use_left = np.abs(levels[left] - ends) < np.abs(levels[pos] - ends)
    return np.where(use_left, left, pos)


def describe_levels(levels: np.ndarray, what: str) -> str:
    if len(levels) == 0:
        return f"No {what} found"
    if len(levels) == 1:
        return f"{what} end on one layer ({levels[0]:.2f} mm)"
    return f"{what} end on {len(levels)} different layers ({levels[0]:.2f}–{levels[-1]:.2f} mm)"


VIEWS = (
    (1, "top", "Flattened top surfaces"),
    (-1, "bottom", "Flattened ceilings (seen from below)"),
)


def build_preview(before: Mesh, after: Mesh, grid: LayerGrid, res: FlattenResult):
    """Everything a canvas needs to draw the before/after picture.

    Returns ``(meta, arrays)``: ``meta`` is JSON-serialisable, ``arrays`` maps names to
    numpy arrays.  ``xy`` holds the (x, y) corners of every face (the same before and
    after); per view and per mesh there are the draw ``order`` (far to near), a grey
    ``shade`` and a layer-``level`` index per face, and the layer-edge ``segs``.
    """
    changed = flattened_indices(res)
    F = before.n_faces
    meta: dict = {"views": [], "n_faces": int(F)}
    arrays: dict = {}
    if not changed:
        return meta, arrays
    region_all = np.isin(res.face_plateau, changed)
    lo, hi = before.verts[:, :2].min(0), before.verts[:, :2].max(0)
    meta["bbox"] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    arrays["xy"] = after.verts[after.faces][:, :, :2].astype(np.float32)

    zc_of = {}
    n_of = {}
    for name, mesh in (("before", before), ("after", after)):
        n_of[name], _ = mesh.face_normals_areas()
        zc_of[name] = mesh.face_centroids()[:, 2]

    for sign, key, what in VIEWS:
        mask = region_all & (res.facing == sign)
        if not mask.any():
            continue
        lv_before = region_levels(before, grid, mask)
        lv_after = region_levels(after, grid, mask)
        levels = np.unique(np.concatenate([lv_before, lv_after]))
        colours = level_colours(levels, lv_after)
        view = {
            "key": key,
            "what": what,
            "levels": [float(v) for v in levels],
            "colours": [colours[float(v)] for v in levels],
            "before_text": describe_levels(lv_before, what),
            "after_text": describe_levels(lv_after, what),
            "before_layers": int(len(lv_before)),
            "after_layers": int(len(lv_after)),
            "mirror_x": sign < 0,
        }
        for name, mesh in (("before", before), ("after", after)):
            nz = n_of[name][:, 2]
            zc = zc_of[name]
            facing = sign * nz > 0.02
            idx = np.flatnonzero(facing)
            order = idx[np.argsort(sign * zc[idx], kind="stable")].astype(np.uint32)  # far to near
            shade = (255.0 * (0.40 + 0.35 * np.clip(sign * nz, 0, 1))).astype(np.uint8)
            level = np.full(F, 255, dtype=np.uint8)
            level[mask] = nearest_level(levels, np.round(grid.layer_end(zc[mask]), 6)).astype(np.uint8)
            a, b, _ = contour_segments(mesh, grid, facing)
            segs = np.concatenate([a[:, :2], b[:, :2]], axis=1).astype(np.float32)
            arrays[f"{key}_{name}_order"] = order
            arrays[f"{key}_{name}_shade"] = shade
            arrays[f"{key}_{name}_level"] = level
            arrays[f"{key}_{name}_segs"] = segs
        meta["views"].append(view)
    return meta, arrays
