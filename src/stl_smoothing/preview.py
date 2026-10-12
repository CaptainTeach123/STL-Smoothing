"""Data for the before/after picture, without any plotting library.

The command line draws it with matplotlib (:mod:`report`); the web page draws the same
picture on a canvas from the arrays built here.  The colour scheme is shared, so both
look the same: the layer a surface ends on is blue (lower) -> pale blue (the level it was
moved to) -> peach -> red (higher), and every slicer layer edge is an orange line.
"""

from __future__ import annotations

import numpy as np

from ._compat import bincount
from .flatten import FlattenResult
from .layers import LayerGrid
from .mesh import Mesh
from .slicing import contour_segment_count, contour_segments
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
    w = bincount(inv, weights=area[faces])
    return ids[w > min_frac * w.sum()]


def nearest_level(levels: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the closest entry of the sorted ``levels`` for every value of ``ends``."""
    pos = np.clip(np.searchsorted(levels, ends), 0, len(levels) - 1)
    left = np.clip(pos - 1, 0, len(levels) - 1)
    use_left = np.abs(levels[left] - ends) < np.abs(levels[pos] - ends)
    return np.where(use_left, left, pos)


def _join(items) -> str:
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def view_texts(before: Mesh, after: Mesh, grid: LayerGrid, labels: np.ndarray, mask: np.ndarray,
               noun: str = "surface") -> tuple[str, str, list, list]:
    """Captions for the Before and After pictures, per flattened surface (``labels`` = plateau index
    per face, ``mask`` = the faces shown).  Several surfaces at different heights each ending on one
    layer must read as success, not as "3 different layers".  Also returns the layer counts."""
    ids = sorted({int(i) for i in labels[mask]})
    lv_b = [region_levels(before, grid, mask & (labels == i)) for i in ids]
    lv_a = [region_levels(after, grid, mask & (labels == i)) for i in ids]
    nb, na = [len(v) for v in lv_b], [len(v) for v in lv_a]
    if len(ids) == 1:
        b, a = lv_b[0], lv_a[0]
        before_text = (f"The flattened {noun} ended on {nb[0]} layers ({b[0]:.2f}–{b[-1]:.2f} mm)" if nb[0] > 1
                       else f"The flattened {noun} ended on one layer ({b[0]:.2f} mm)" if nb[0] else f"No {noun} found")
        after_text = (f"It now ends on one layer ({a[0]:.2f} mm)" if na[0] == 1
                      else f"It now ends on {na[0]} layers ({a[0]:.2f}–{a[-1]:.2f} mm)" if na[0] else "")
    else:
        before_text = f"{len(ids)} flattened {noun}s; they ended on {_join(nb)} layers"
        if all(k == 1 for k in na):
            after_text = "Each now ends on one layer: " + _join(f"{v[0]:.2f}" for v in sorted(lv_a, key=lambda v: v[0])) + " mm"
        else:
            after_text = f"They now end on {_join(na)} layers"
    return before_text, after_text, nb, na


def cut_limit(before: Mesh, after: Mesh, grid: LayerGrid, mask: np.ndarray, sign: int):
    """Height beyond which geometry can hide the flattened surfaces from the viewer (above them for a top
    view, below them for a ceiling view), and the margin used; :func:`hiding_faces` decides which of
    the faces beyond it really are in the way.  A tall model standing on a flat base
    would otherwise show only its own top in a picture meant to show the base."""
    margin = max(1.0, 3 * grid.layer_height)
    z = np.concatenate([m.verts[m.faces[mask]][:, :, 2].ravel() for m in (before, after)])
    return (float(z.max()) + margin if sign > 0 else float(z.min()) - margin), margin


def _footprint(mesh: Mesh, faces: np.ndarray, cell: float, origin: np.ndarray, shape) -> np.ndarray:
    """Boolean xy grid of the cells the selected triangles cover (sampled at about the cell size)."""
    grid = np.zeros(shape, dtype=bool)
    tri = mesh.verts[mesh.faces[faces]][:, :, :2]
    if len(tri) == 0:
        return grid
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    k_all = np.clip(np.ceil(edge / cell).astype(int), 1, 64)
    for k in np.unique(k_all):
        t = tri[k_all == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        u, v = i[keep] / k, j[keep] / k  # barycentric samples, about one per cell
        pts = (t[:, None, 0] + u[None, :, None] * (t[:, None, 1] - t[:, None, 0])
               + v[None, :, None] * (t[:, None, 2] - t[:, None, 0])).reshape(-1, 2)
        ij = np.floor((pts - origin) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        grid[ij[ok, 0], ij[ok, 1]] = True
    return grid


def hiding_faces(mesh: Mesh, region: np.ndarray, candidates: np.ndarray, min_share: float = 0.02) -> np.ndarray:
    """Which ``candidates`` lie over (in x and y) the ``region`` faces, so a picture taken along z would
    show them instead of the region.  Nothing is reported unless they would hide at least ``min_share``
    of the region: a dome beside a flattened panel (touching only its edge) does not hide it; a lamp
    standing on a flattened base does."""
    out = np.zeros(len(candidates), dtype=bool)
    if not candidates.any() or not region.any():
        return out
    lo, hi = mesh.verts[:, :2].min(0), mesh.verts[:, :2].max(0)
    cell = max(0.5, float((hi - lo).max()) / 1024)
    shape = tuple((np.floor((hi - lo) / cell).astype(int) + 2).tolist())
    cover = _footprint(mesh, region, cell, lo, shape)
    over = _footprint(mesh, candidates, cell, lo, shape)
    if (cover & over).sum() < min_share * cover.sum():
        return out
    idx = np.flatnonzero(candidates)
    tri = mesh.verts[mesh.faces[idx]][:, :, :2]
    pts = np.concatenate([tri, tri.mean(1, keepdims=True)], axis=1)  # 3 corners and the centroid
    ij = np.clip(np.floor((pts - lo) / cell).astype(int), 0, np.array(shape) - 1)
    out[idx] = cover[ij[..., 0], ij[..., 1]].any(axis=1)
    return out


def region_bbox(mesh: Mesh, mask: np.ndarray) -> list:
    """xy box ``[x0, y0, x1, y1]`` a picture should show: the flattened faces with some room around them,
    inside the model's own box.  A small surface on a big model is then large enough to see."""
    lo, hi = mesh.verts[:, :2].min(0), mesh.verts[:, :2].max(0)
    xy = mesh.verts[mesh.faces[mask]][:, :, :2].reshape(-1, 2)
    if len(xy) == 0:
        return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    rlo, rhi = xy.min(0), xy.max(0)
    pad = np.maximum(5.0, 0.15 * (rhi - rlo))
    blo, bhi = np.maximum(rlo - pad, lo), np.minimum(rhi + pad, hi)
    return [float(blo[0]), float(blo[1]), float(bhi[0]), float(bhi[1])]


def describe_levels(levels: np.ndarray, what: str) -> str:
    if len(levels) == 0:
        return f"No {what} found"
    if len(levels) == 1:
        return f"{what} end on one layer ({levels[0]:.2f} mm)"
    return f"{what} end on {len(levels)} different layers ({levels[0]:.2f}–{levels[-1]:.2f} mm)"


# Layer-edge segments drawn per picture.  Their number grows as 1 / layer height (a 200k-triangle model at
# 0.01 mm layers would be 3 million), and building them costs about 300 bytes each.
MAX_EDGE_SEGMENTS = 800_000

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
        before_text, after_text, _, _ = view_texts(before, after, grid, res.face_plateau, mask,
                                                   "surface" if sign > 0 else "ceiling")
        limit, margin = cut_limit(before, after, grid, mask, sign)
        view = {
            "key": key,
            "what": what,
            "levels": [float(v) for v in levels],
            "colours": [colours[float(v)] for v in levels],
            "before_text": before_text,
            "after_text": after_text,
            "before_layers": int(len(lv_before)),
            "after_layers": int(len(lv_after)),
            "mirror_x": sign < 0,
            "bbox": region_bbox(before, mask),
        }
        for name, mesh in (("before", before), ("after", after)):
            nz = n_of[name][:, 2]
            zc = zc_of[name]
            facing = sign * nz > 0.02
            beyond = hiding_faces(mesh, mask, facing & (sign * zc > sign * limit))
            if beyond.any():  # cut away what hides the flattened surfaces (a model standing on its base)
                facing = facing & ~beyond
                view["cut_note"] = (f"Parts of the model more than {margin:g} mm "
                                    f"{'above the flattened surfaces' if sign > 0 else 'below the flattened ceilings'} "
                                    "are left out of this picture so the surfaces can be seen.")
            idx = np.flatnonzero(facing)
            order = idx[np.argsort(sign * zc[idx], kind="stable")].astype(np.uint32)  # far to near
            shade = (255.0 * (0.40 + 0.35 * np.clip(sign * nz, 0, 1))).astype(np.uint8)
            level = np.full(F, 255, dtype=np.uint8)
            level[mask] = nearest_level(levels, np.round(grid.layer_end(zc[mask]), 6)).astype(np.uint8)
            edge_faces = facing
            if contour_segment_count(mesh, grid, edge_faces) > MAX_EDGE_SEGMENTS:
                edge_faces = facing & mask  # too many at this layer height: only the flattened surfaces
                view["edge_note"] = ("At this layer height there are too many layer edges to draw them all, "
                                     "so only the ones on the flattened surfaces are shown.")
            if contour_segment_count(mesh, grid, edge_faces) > MAX_EDGE_SEGMENTS:
                edge_faces = np.zeros_like(facing)
                view["edge_note"] = "At this layer height there are too many layer edges to draw them."
            a, b, _ = contour_segments(mesh, grid, edge_faces)
            segs = np.concatenate([a[:, :2], b[:, :2]], axis=1).astype(np.float32)
            arrays[f"{key}_{name}_order"] = order
            arrays[f"{key}_{name}_shade"] = shade
            arrays[f"{key}_{name}_level"] = level
            arrays[f"{key}_{name}_segs"] = segs
        meta["views"].append(view)
    return meta, arrays
