"""Moving the vertices: pin plateaus to their level, blend into the neighbours.

Only z changes.  The plateau vertices are pinned to the snapped level; every other
vertex receives a share of the move of its nearest pinned vertex that decays
smoothly with distance (the "feather"), so no crease appears where a plateau
meets a skirt or a slanted wall.  Walls are kept from folding over, and any face
that would still flip is damped back.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from ._compat import bincount, reduceat
from ._detect import Analysis
from .layers import LayerGrid
from .mesh import Mesh
from .params import Params


def vertex_targets(mesh: Mesh, grid: LayerGrid, A: Analysis, P: Params) -> np.ndarray:
    """Target height of every pinned vertex (NaN elsewhere).

    Vertices of accepted plateau faces go to the level of their patch (a vertex
    shared by two patches goes with the larger one).  Vertices of exactly-flat
    patches go to the layer boundary nearest their level and win any conflict.
    """
    faces = mesh.faces
    V = mesh.n_verts
    area, exact, topo, zc = A.area, A.exact, A.topo, A.zc
    pc, clevel = A.plateau_comp, A.clevel
    tgt = np.full(V, np.nan)
    f_acc = np.flatnonzero(pc >= 0)
    if len(f_acc):
        G = len(clevel)
        vv = faces[f_acc].ravel()
        gg = np.repeat(pc[f_acc], 3)
        ww = np.repeat(area[f_acc], 3)
        uk, inv = np.unique(vv * np.int64(G) + gg, return_inverse=True)
        wsum = bincount(inv, weights=ww)
        uv, ug = uk // G, uk % G
        order = np.lexsort((wsum, uv))  # per vertex, the heaviest patch last
        uv_s, ug_s = uv[order], ug[order]
        last = np.r_[uv_s[1:] != uv_s[:-1], True]
        tgt[uv_s[last]] = clevel[ug_s[last]]
    tgt = _absorb_spikes(mesh, tgt, P)
    if P.snap_exact_flat and len(A.exact_area):
        need = exact_needs_snap(A, grid, P)
        sel = (A.exact_label >= 0) & need[np.maximum(A.exact_label, 0)]
        fi = np.flatnonzero(sel)
        lvl = grid.snap(A.exact_z)
        tgt[faces[fi].ravel()] = np.repeat(lvl[A.exact_label[fi]], 3)
    return tgt


def exact_needs_snap(A: Analysis, grid: LayerGrid, P: Params) -> np.ndarray:
    """Which exactly-flat patches should be nudged onto the layer grid.

    A patch that already prints on one layer well clear of the slicer's sampling
    planes is left exactly as it is; one that sits on or near a plane (it could
    print on either layer) moves to the nearest layer boundary.
    """
    z = A.exact_z
    m = P.plane_margin * grid.layer_height + 1e-6  # inclusive: a plateau exactly one margin away still counts
    good = (A.exact_area >= P.min_flat_area) & (z > 0.5 * grid.first)
    straddles = grid.layer_number(z - m) != grid.layer_number(z + m)
    return good & straddles


def _absorb_spikes(mesh: Mesh, tgt: np.ndarray, P: Params) -> np.ndarray:
    """Pull a lone vertex whose whole neighbourhood is ONE plateau onto that plateau.

    Single-vertex spikes and pits ("hot pixels") belong only to steep faces, so
    they are never part of a plateau, and would stand out of the flattened
    surface.
    """
    pinned = ~np.isnan(tgt)
    if not pinned.any() or P.spike_max <= 0:
        return tgt
    g = mesh.vertex_graph(weighted=False)
    indptr, indices = g.indptr, g.indices
    deg = np.diff(indptr)
    ne = deg > 0
    if not ne.any():
        return tgt
    V = mesh.n_verts
    starts = indptr[:-1][ne]
    nb_min = np.full(V, np.inf)
    nb_max = np.full(V, -np.inf)
    nb_free = np.zeros(V)
    nb_min[ne] = reduceat(np.minimum, np.where(pinned, tgt, np.inf)[indices], starts)
    nb_max[ne] = reduceat(np.maximum, np.where(pinned, tgt, -np.inf)[indices], starts)
    nb_free[ne] = reduceat(np.add, (~pinned)[indices].astype(np.float64), starts)
    z = mesh.verts[:, 2]
    spike = (
        ~pinned & ne & (nb_free == 0) & (nb_min == nb_max) & (np.abs(z - nb_min) <= P.spike_max)
    )
    if not spike.any():
        return tgt
    out = tgt.copy()
    out[spike] = nb_min[spike]
    return out


def feather(mesh: Mesh, A: Analysis, dz: np.ndarray, pinned: np.ndarray, bed_v: np.ndarray, P: Params) -> np.ndarray:
    """Spread the move of the pinned vertices into their neighbourhood.

    Every vertex that is not pinned gets a share of the move of its nearest
    pinned vertex, decaying smoothly (C1) with the distance travelled over the
    mesh edges.  Across soft faces (continuations of the surface) the decay
    length is set by a tolerated slope change; across walls it is a small
    multiple of the move, which is just enough to keep wall faces from folding
    over when their top row is moved.
    """
    faces = mesh.faces
    V = mesh.n_verts
    delta = dz.copy()
    src = pinned & (np.abs(dz) > 1e-9)
    if not src.any():
        return delta
    face_soft = A.valid & (A.soft[+1] | A.soft[-1])
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    fsoft = np.tile(face_soft, 3)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    usable = (~pinned & ~bed_v) | src
    m = usable[lo] & usable[hi] & ~(src[lo] & src[hi])
    lo, hi, fsoft = lo[m], hi[m], fsoft[m]
    if len(lo) == 0:
        return delta
    key = lo * np.int64(V) + hi
    order = np.argsort(key, kind="stable")
    key, lo, hi, fsoft = key[order], lo[order], hi[order], fsoft[order]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    soft_edge = reduceat(np.logical_or, fsoft, starts)
    lo, hi = lo[starts], hi[starts]
    L = np.linalg.norm(mesh.verts[lo] - mesh.verts[hi], axis=1)
    L = np.maximum(L, 1e-9) * np.where(soft_edge, 1.0, P.steep_cost)
    G = sparse.coo_matrix((np.r_[L, L], (np.r_[lo, hi], np.r_[hi, lo])), shape=(V, V)).tocsr()
    srcs = np.flatnonzero(src)
    dist, _, origin = csgraph.dijkstra(
        G, directed=False, indices=srcs, min_only=True, return_predecessors=True, limit=P.feather_max
    )
    elig = ~pinned & ~bed_v
    reach = elig & np.isfinite(dist) & (origin >= 0)
    d0 = np.where(reach, dz[np.maximum(origin, 0)], 0.0)
    W = np.clip(np.abs(d0) / np.tan(np.radians(P.feather_slope_deg)) * 1.5, P.feather_min, P.feather_max)
    t = np.clip(dist / W, 0, 1)
    wgt = np.where(reach, (1 - t) ** 2 * (1 + 2 * t), 0.0)
    f = d0 * wgt
    # smooth the blended correction a few times (ring vertices only)
    if P.feather_smooth_iters > 0 and reach.any():
        Gu = G.copy()
        Gu.data = np.ones_like(Gu.data)
        deg = np.asarray(Gu.sum(axis=1)).ravel()
        full = np.where(src, dz, f)
        for _ in range(P.feather_smooth_iters):
            avg = (Gu @ full) / np.maximum(deg, 1)
            full = np.where(reach & (deg > 0), 0.5 * full + 0.5 * avg, full)
        f = np.where(reach, full, 0.0)
    delta[reach] = f[reach]
    return delta


def unfold(mesh: Mesh, A: Analysis, z0: np.ndarray, z: np.ndarray, fixed: np.ndarray, P: Params) -> np.ndarray:
    """Keep wall faces from folding over.

    Moving the top row of a wall by more than the height of the neighbouring
    vertices turns the face inside out.  For every edge of a steep face the order
    of its end heights is kept (with at least ``unfold_keep`` of the original
    height difference) by pushing the free vertex along; pinned vertices never
    move.  Violations are relaxed a few times, Jacobi style.
    """
    iters = P.unfold_iters
    if iters <= 0:
        return z
    faces = mesh.faces
    steep = A.valid & ~(A.soft[+1] | A.soft[-1])
    fs = faces[steep]
    if len(fs) == 0:
        return z
    V = mesh.n_verts
    a = np.concatenate([fs[:, 0], fs[:, 1], fs[:, 2]])
    b = np.concatenate([fs[:, 1], fs[:, 2], fs[:, 0]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * np.int64(V) + hi)
    lo, hi = key // V, key % V
    d0 = z0[lo] - z0[hi]
    m = (np.abs(d0) > 1e-9) & ~(fixed[lo] & fixed[hi])
    moved = np.abs(z - z0) > 1e-9  # only edges touching a vertex that moved matter
    m &= moved[lo] | moved[hi]
    lo, hi, d0 = lo[m], hi[m], d0[m]
    if len(lo) == 0:
        return z
    s0 = np.sign(d0)
    need = P.unfold_keep * np.abs(d0)
    z = z.copy()
    for _ in range(iters):
        d1 = z[lo] - z[hi]
        viol = need - s0 * d1  # > 0: the pair has been squeezed / reversed
        bad = viol > 1e-9
        if not bad.any():
            break
        l, h, v, s = lo[bad], hi[bad], viol[bad], s0[bad]
        fl, fh = ~fixed[l], ~fixed[h]
        wl = np.where(fl & fh, 0.5, np.where(fl, 1.0, 0.0))
        wh = np.where(fl & fh, 0.5, np.where(fh, 1.0, 0.0))
        # the "upper" end (sign s) is pushed up, the "lower" end down
        shift = np.zeros(V)
        cnt = np.zeros(V)
        np.add.at(shift, l, s * wl * v)
        np.add.at(cnt, l, (wl > 0).astype(float))
        np.add.at(shift, h, -s * wh * v)
        np.add.at(cnt, h, (wh > 0).astype(float))
        z += np.where(cnt > 0, shift / np.maximum(cnt, 1), 0.0)
    return z


def repair_flips(mesh: Mesh, new_z: np.ndarray, z0: np.ndarray, delta: np.ndarray, P: Params) -> np.ndarray:
    """Damp the move at the corners of any face that would still flip.

    Incremental version: the reference normals are computed once, and after the
    first full pass only faces that touch a vertex whose damping changed are re-tested.
    """
    verts = mesh.verts
    faces = mesh.faces
    V = mesh.n_verts
    lam = np.ones(V)
    c0 = mesh.face_cross(verts)
    big = 0.5 * np.linalg.norm(c0, axis=1) > 1e-9

    def test(cur, fidx=None):
        nv = verts.copy()
        nv[:, 2] = cur
        if fidx is None:
            c1 = mesh.face_cross(nv)
            return big & ((c0 * c1).sum(axis=1) < 0)
        t = nv[faces[fidx]]
        c1 = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return big[fidx] & ((c0[fidx] * c1).sum(axis=1) < 0)

    cur = new_z
    fl = test(cur)
    for _ in range(P.damp_iters):
        if not fl.any():
            return cur
        vs = np.unique(faces[fl])
        lam[vs] *= 0.5
        cur = z0 + lam * delta
        touched = np.zeros(V, bool)
        touched[vs] = True
        fidx = np.flatnonzero(touched[faces].any(axis=1))
        fl[fidx] = test(cur, fidx)
    if fl.any():
        vs = np.unique(faces[fl])
        lam[vs] = 0.0
        cur = z0 + lam * delta
    return cur
