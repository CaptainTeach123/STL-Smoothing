"""Emulate where a slicer draws layer edges on a surface."""

from __future__ import annotations

import numpy as np

from .layers import LayerGrid
from .mesh import Mesh


def slice_planes(grid: LayerGrid, zmin: float, zmax: float) -> np.ndarray:
    """Mid-layer sampling heights between zmin and zmax."""
    first = grid.first
    h = grid.layer_height
    planes = [first / 2.0]
    z = first + 0.5 * h
    while z <= zmax + h:
        planes.append(z)
        z += h
    p = np.asarray(planes)
    return p[(p >= zmin - h) & (p <= zmax + h)]


def _plane_counts(t: np.ndarray, grid: LayerGrid):
    """For triangles ``t`` (F,3,3): the sampling planes and, per face, the first plane
    strictly inside its z range and how many planes that is."""
    z = t[:, :, 2]
    planes = slice_planes(grid, float(z.min()), float(z.max()))
    zmin, zmax = z.min(1), z.max(1)
    lo = np.searchsorted(planes, zmin, side="right")
    hi = np.searchsorted(planes, zmax, side="left")
    return planes, lo, np.maximum(hi - lo, 0)


def contour_segment_count(mesh: Mesh, grid: LayerGrid, face_mask: np.ndarray | None = None) -> int:
    """How many segments :func:`contour_segments` would return (an upper bound), without building them."""
    t = mesh.verts[mesh.faces]
    if face_mask is not None:
        t = t[face_mask]
    if len(t) == 0:
        return 0
    return int(_plane_counts(t, grid)[2].sum())


def contour_segments(mesh: Mesh, grid: LayerGrid, face_mask: np.ndarray | None = None,
                     verts: np.ndarray | None = None):
    """Exact slicer contour segments of the surface at every mid-layer plane.

    Returns ``(seg_a, seg_b, face_idx)``: endpoints (S, 3) each and the face
    each segment lies in.  These are the "layer edges" a slicer draws on the
    surface.
    """
    v = mesh.verts if verts is None else verts
    t = v[mesh.faces]  # (F,3,3)
    fidx = np.arange(mesh.n_faces)
    if face_mask is not None:
        t = t[face_mask]
        fidx = fidx[face_mask]
    if len(t) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    planes, lo, cnt = _plane_counts(t, grid)  # for each face, the planes strictly inside (zmin, zmax)
    if cnt.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    fi = np.repeat(np.arange(len(t)), cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    p = planes[lo[fi] + offs]
    tf = t[fi]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = tf[:, a, 2], tf[:, b, 2]
        crosses = (za - p) * (zb - p) < 0
        s = np.divide(p - za, zb - za, out=np.zeros_like(za), where=crosses)
        pts.append((crosses, tf[:, a] + (tf[:, b] - tf[:, a]) * s[:, None]))
    # exactly two of the three edges are crossed for a generic triangle
    seg_a = np.zeros((len(fi), 3))
    seg_b = np.zeros((len(fi), 3))
    got = np.zeros(len(fi), int)
    for crosses, q in pts:
        first = crosses & (got == 0)
        second = crosses & (got == 1)
        seg_a[first] = q[first]
        seg_b[second] = q[second]
        got = got + crosses
    ok = got == 2
    return seg_a[ok], seg_b[ok], fidx[fi[ok]]


def contour_length(mesh: Mesh, grid: LayerGrid, face_mask: np.ndarray | None = None,
                   verts: np.ndarray | None = None) -> float:
    """Total xy length (mm) of slicer layer edges on the selected faces."""
    a, b, _ = contour_segments(mesh, grid, face_mask, verts)
    return float(np.linalg.norm((a - b)[:, :2], axis=1).sum())
