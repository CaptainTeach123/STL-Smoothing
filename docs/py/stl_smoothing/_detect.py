"""Finding the plateaus: which faces belong to a surface that was meant to be flat.

Method (global height-histogram mode seeking, then connectivity clean-up)
------------------------------------------------------------------------
1. **Classify faces** with edge-preserving *smoothed* normals so facet noise does
   not matter.  Near-flat faces are candidates; steeper ones (but not walls) are
   "soft" continuations; everything else is a wall.  Up- and down-facing surfaces
   are handled separately.
2. **Exactly flat** faces are intentional unless noisy faces continue them
   smoothly; they are never merged with anything and only snapped to the grid.
   Exactly *planar* tilted patches (a clean ramp) are left alone.
3. **Histogram** of candidate heights over the WHOLE mesh, area weighted and
   without connectivity.  Its peaks are plateau levels: disconnected panels at the
   same level add up to the same peak, which is what puts them on the same layer.
   A shallow valley merges two peaks, a deep valley (a real step) keeps them apart.
   Each peak gets a window trimmed where the density falls back to the pedestal
   that a ramp or dome contributes.
4. **Connectivity.**  Candidates are assigned to the window that contains them.
   A component whose boundary is mostly a *window cut* (the surface goes on
   smoothly beyond it) is a clipped ramp, skirt or dome apex and is rejected; a
   real plateau is bounded by steep walls.  Small holes inside a plateau are
   filled, tiny fragments are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csgraph

from .layers import LayerGrid
from .mesh import Mesh
from .params import Params


# --------------------------------------------------------------------- helpers
def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted median."""
    if len(x) == 0:
        return 0.0
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    k = int(np.searchsorted(cw, 0.5 * cw[-1]))
    return float(x[o][min(k, len(x) - 1)])


class Topo:
    """Face adjacency with shared-edge lengths (xy), computed once."""

    def __init__(self, mesh: Mesh):
        f = mesh.faces
        F = len(f)
        V = mesh.n_verts
        a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = lo * np.int64(V) + hi
        order = np.argsort(key, kind="stable")
        sk = key[order]
        same = sk[1:] == sk[:-1]
        i0 = order[:-1][same]
        i1 = order[1:][same]
        self.fa = i0 % F
        self.fb = i1 % F
        d = mesh.verts[lo[i0], :2] - mesh.verts[hi[i0], :2]
        self.elen = np.hypot(d[:, 0], d[:, 1])
        # open (boundary) edges: appear exactly once
        # sk is already sorted: run starts / lengths by differencing (np.unique would sort the 3F keys again)
        first = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        counts = np.diff(np.r_[first, len(sk)])
        op = order[first[counts == 1]]
        d = mesh.verts[lo[op], :2] - mesh.verts[hi[op], :2]
        self.open_face = op % F
        self.open_len = np.hypot(d[:, 0], d[:, 1])
        self.F = F


def components(topo: Topo, member: np.ndarray, same: np.ndarray | None = None):
    """Connected components over the faces in ``member``.

    ``same`` (optional, per-face int) restricts connections to equal values.
    Returns ``(count, labels)`` with labels ``-1`` outside ``member``.
    """
    fa, fb = topo.fa, topo.fb
    keep = member[fa] & member[fb]
    if same is not None:
        keep &= same[fa] == same[fb]
    n = topo.F
    g = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (fa[keep], fb[keep])), shape=(n, n)
    )
    _, lab = csgraph.connected_components(g, directed=False)
    out = -np.ones(n, dtype=np.int64)
    if member.any():
        u, inv = np.unique(lab[member], return_inverse=True)
        out[member] = inv
        return len(u), out
    return 0, out


def grouped_wmedian(labels: np.ndarray, values: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """Weighted median of ``values`` for every label 0..n-1 (NaN for empty labels)."""
    out = np.full(n, np.nan)
    if len(labels) == 0:
        return out
    order = np.lexsort((values, labels))
    l, v, w = labels[order], values[order], weights[order]
    cw = np.cumsum(w)
    ids = np.arange(n)
    starts = np.searchsorted(l, ids, side="left")
    ends = np.searchsorted(l, ids, side="right")
    has = ends > starts
    base = np.where(starts > 0, cw[np.maximum(starts - 1, 0)], 0.0)
    tot = np.where(has, cw[np.maximum(ends - 1, 0)] - base, 0.0)
    idx = np.searchsorted(cw, base + 0.5 * tot)
    idx = np.minimum(np.maximum(idx, starts), np.maximum(ends - 1, 0))
    out[has] = v[idx[has]]
    return out


def plateau_levels(mesh, cl, accepted, comp_group, zc, area, soft, valid, glevel_snap, grid, P):
    """Level of every accepted plateau component.

    All components of a window group share the group level, except across a
    *step*: two components of one group that meet at a wall and whose median
    heights differ by at least ``step_layers`` layers are a deliberate step (an
    embossed pad, an engraved panel, a ledge).  Such a component keeps its height
    relative to its neighbour, rounded to whole layers.
    """
    ncomp = len(accepted)
    clevel = np.full(ncomp, np.nan)
    sel = (cl >= 0) & accepted[np.maximum(cl, 0)]
    if not sel.any():
        return clevel
    carea = np.bincount(cl[sel], weights=area[sel], minlength=ncomp)
    cmed = grouped_wmedian(cl[sel], zc[sel], area[sel], ncomp)
    clevel[accepted] = glevel_snap[comp_group[accepted]]  # default: the group level
    h = grid.layer_height
    s_min = P.step_layers * h

    # component adjacency across walls
    vcomp = -np.ones(mesh.n_verts, dtype=np.int64)
    fi = np.flatnonzero(sel)
    vcomp[mesh.faces[fi].ravel()] = np.repeat(cl[fi], 3)
    steep = valid & ~(soft[+1] | soft[-1])
    c3 = vcomp[mesh.faces[steep]]
    pairs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        a, b = c3[:, i], c3[:, j]
        ok = (a >= 0) & (b >= 0) & (a != b)
        pairs.append(np.stack([np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])], axis=1))
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    if len(pairs) == 0:
        return clevel
    key = pairs[:, 0] * np.int64(ncomp) + pairs[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    a, b = uk // ncomp, uk % ncomp
    # only substantial patches can be a deliberate step; a sliver attached to the edge of a
    # plateau is noise and just follows the group level
    big = carea >= P.min_area
    keep = (cnt >= P.step_min_faces) & (comp_group[a] == comp_group[b]) & accepted[a] & accepted[b] & big[a] & big[b]
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return clevel
    nbrs: dict = {}
    for x, y in zip(a.tolist(), b.tolist()):
        nbrs.setdefault(x, []).append(y)
        nbrs.setdefault(y, []).append(x)
    # pass 1: height of every component relative to the root of its cluster (0 for roots / lone comps)
    rel = np.zeros(ncomp)
    parent: dict = {}
    order: list = []
    seen = set()
    for start in sorted(nbrs, key=lambda c: -carea[c]):  # biggest component of a cluster is its root
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        while stack:
            p = stack.pop()
            for c in nbrs[p]:
                if c in seen:
                    continue
                seen.add(c)
                d = cmed[c] - cmed[p]
                rel[c] = rel[p] if abs(d) < s_min else rel[p] + d
                parent[c] = p
                order.append(c)
                stack.append(c)
    # pass 2: the level of a group that has steps is the median of its faces with the steps taken out
    # (otherwise the root of a stepped cluster is put on the median of the union of all its levels)
    fsel = np.flatnonzero(sel)
    gf = comp_group[cl[fsel]]
    relf = rel[cl[fsel]]
    zf = zc[fsel] - relf
    af = area[fsel]
    base = np.array(glevel_snap, dtype=float)
    for g in np.unique(gf[relf != 0]):
        mg = gf == g
        base[g] = float(grid.snap(wmedian(zf[mg], af[mg])))
    clevel[accepted] = base[comp_group[accepted]]
    for c in order:
        p = parent[c]
        d = cmed[c] - cmed[p]
        clevel[c] = clevel[p] if abs(d) < s_min else float(grid.snap(clevel[p] + d))
    return clevel


# ---------------------------------------------------------- smoothed normals
def smoothed_normals(mesh: Mesh, n: np.ndarray, area: np.ndarray, valid: np.ndarray, P: Params):
    """Face normals averaged over a small neighbourhood of *near-flat* faces.

    Per-vertex jitter (marching cubes, remeshing, noisy generators) tilts single
    faces by many degrees although the surface is flat.  Averaging the normals
    of neighbouring faces removes that noise.  Only faces within
    ``family_slope_deg`` of horizontal (same orientation) take part, so walls do
    not bleed into the plateau edges.  Faces outside the family keep their own
    normal.
    """
    F, V = mesh.n_faces, mesh.n_verts
    out = n.copy()
    if not valid.any():
        return out
    l_med = float(np.sqrt(4.0 * np.median(area[valid]) / np.sqrt(3.0)))
    rounds = int(np.clip(round(P.smooth_radius / max(l_med, 1e-9)), 1, P.smooth_max_rounds))
    cross = n * (2.0 * area)[:, None]
    nz = n[:, 2]
    fam = valid & (np.abs(nz) >= np.cos(np.radians(P.family_slope_deg)))
    for s in (+1, -1):
        mem = fam & (s * nz > 0)
        idx = np.flatnonzero(mem)
        if len(idx) == 0:
            continue
        rows = mesh.faces[idx].ravel()
        cols = np.repeat(np.arange(len(idx)), 3)
        inc = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, len(idx)))
        # face-face adjacency (weight = shared corners), cut at the smoothing reach: a ring
        # average must not pool faces that are further apart than smooth_radius (an apex fan,
        # a ridge between coarse triangles), otherwise a cone or a gable looks flat
        adj = (inc.T @ inc).tocoo()
        cen = mesh.verts[mesh.faces[idx]].mean(axis=1)
        dist = np.linalg.norm(cen[adj.row] - cen[adj.col], axis=1)
        keep = (adj.row == adj.col) | (dist <= 2.0 * P.smooth_radius)
        adj = sparse.csr_matrix((adj.data[keep], (adj.row[keep], adj.col[keep])), shape=adj.shape)
        c = cross[idx]
        for _ in range(rounds):
            c = adj @ c
        ln = np.linalg.norm(c, axis=1, keepdims=True)
        out[idx] = np.divide(c, ln, out=n[idx].copy(), where=ln > 0)
    return out


# ------------------------------------------------------------- mode seeking
def find_windows(z: np.ndarray, w: np.ndarray, h: float, P: Params) -> list[dict]:
    """Peaks of the area-weighted height histogram and their windows.

    Returns dicts with ``lo``, ``hi`` (z window), ``rep`` (mode z), ``dens``,
    ``ped`` (pedestal density), ``mass`` (area inside the window) and ``span``
    (robust height extent of the plateau).
    """
    if len(z) == 0:
        return []
    b = P.bin_frac * h
    sig = max(P.hist_sigma / b, 0.5)
    cap = int(np.ceil(P.max_half_range / b))
    pad = int(np.ceil(4 * sig)) + cap + 2
    z0 = float(z.min()) - pad * b
    nb = int(np.ceil((float(z.max()) - z0) / b)) + pad + 2
    idx = np.floor((z - z0) / b).astype(np.int64)
    H = np.bincount(idx, weights=w, minlength=nb)
    S = gaussian_filter1d(H, sig, mode="constant")
    inner = np.arange(1, nb - 1)
    pk = inner[(S[inner] > S[inner - 1]) & (S[inner] >= S[inner + 1]) & (S[inner] > 1e-6 * S.max())]
    if len(pk) == 0:
        return []
    # Merge peaks separated by a shallow valley: one noisy plateau with a
    # multi-modal height distribution.  A pedestal wiggle never extends a window.
    groups = [dict(mem=[int(p)], rep=int(p)) for p in pk]
    while len(groups) > 1:
        best, br = -1, -1.0
        for j in range(len(groups) - 1):
            a, c = groups[j]["rep"], groups[j + 1]["rep"]
            r = S[a : c + 1].min() / min(S[a], S[c])
            if r > br:
                best, br = j, r
        if br < P.merge_ratio:
            break
        g1, g2 = groups[best], groups[best + 1]
        groups[best : best + 2] = [
            dict(mem=g1["mem"] + g2["mem"], rep=g1["rep"] if S[g1["rep"]] >= S[g2["rep"]] else g2["rep"])
        ]
    for g in groups:
        major = [m for m in g["mem"] if S[m] >= P.member_frac * S[g["rep"]]]
        g["lo"], g["hi"] = min(major), max(major)
    out = []
    for j, g in enumerate(groups):
        D = S[g["rep"]]
        if j == 0:
            lim_l = max(0, g["lo"] - cap)
        else:
            a, c = groups[j - 1]["rep"], g["rep"]
            lim_l = max(a + int(np.argmin(S[a : c + 1])), g["lo"] - cap)
        if j == len(groups) - 1:
            lim_r = min(nb - 1, g["hi"] + cap)
        else:
            a, c = g["rep"], groups[j + 1]["rep"]
            lim_r = min(a + int(np.argmin(S[a : c + 1])), g["hi"] + cap)
        base_l = float(S[lim_l : g["lo"] + 1].min())
        base_r = float(S[g["hi"] : lim_r + 1].min())

        # Where there is no pedestal the window must reach out to the last noise
        # tail (edge_frac small).  A pedestal means a ramp / skirt leaves the
        # plateau on that side: there the window stops where the bump merges into
        # it (edge_frac_ped), which keeps a clipped ramp stripe short.
        def thr(base):
            t = np.clip(base / D / P.ped_scale, 0.0, 1.0)
            return base + (P.edge_frac + (P.edge_frac_ped - P.edge_frac) * t) * (D - base)

        tl, tr = thr(base_l), thr(base_r)
        il = g["lo"]
        while il > lim_l and S[il] > tl:
            il -= 1
        ir = g["hi"]
        while ir < lim_r and S[ir] > tr:
            ir += 1
        zl = z0 + il * b
        zr = z0 + (ir + 1) * b
        seg = H[il : ir + 1]
        cum = np.cumsum(seg)
        tot = float(cum[-1])
        # robust height span: the sparse far tail of the noise does not count
        q0 = int(np.searchsorted(cum, P.span_trim * tot))
        q1 = int(np.searchsorted(cum, (1.0 - P.span_trim) * tot))
        out.append(
            dict(
                lo=zl, hi=zr, rep=z0 + (g["rep"] + 0.5) * b, dens=float(D),
                ped=float(max(base_l, base_r)), mass=tot,
                span=float((max(q1, q0) - q0 + 1) * b),
                # peak density over the mean density across the window: ~1 for heights spread
                # evenly (ramp, crown), 2 or more for a wobbly panel
                peaky=float(D / max(tot / max(len(seg), 1), 1e-12)),
            )
        )
    return out


# ------------------------------------------------------------- face-level tests
def planar_ramps(topo, n, area, valid, P: Params) -> np.ndarray:
    """Faces of exactly planar, tilted patches (intentional ramps / tilted planes).

    Smooth noise has neighbouring normals that differ by hundredths of a degree,
    so "neighbours agree" would chain across a whole noisy panel.  Instead the
    normals are binned on a lattice of ``planar_tol_deg`` and only neighbours in
    the SAME bin connect (two offset lattices so a plane on a bin edge is not
    split).  A real plane stays in one bin, noise leaves it after a millimetre.
    A patch needs ``planar_min_faces`` faces and ``planar_min_area`` mm^2: one
    huge triangle of a coarse mesh is not a ramp.
    """
    nz = n[:, 2]
    c = np.abs(nz)
    member = valid & (c >= np.cos(np.radians(P.max_slope_deg))) & (c <= np.cos(np.radians(P.planar_ramp_deg)))
    out = np.zeros(len(nz), bool)
    if not member.any():
        return out
    cell = np.radians(P.planar_tol_deg)
    fa, fb = topo.fa, topo.fb
    Fn = topo.F
    for off in (0.0, 0.5):
        kx = np.floor(n[:, 0] / cell + off).astype(np.int64)
        ky = np.floor(n[:, 1] / cell + off).astype(np.int64)
        key = (kx * 2000003 + ky) * 2 + (nz > 0)
        keep = member[fa] & member[fb] & (key[fa] == key[fb])
        g = sparse.coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (fa[keep], fb[keep])), shape=(Fn, Fn))
        _, lab = csgraph.connected_components(g, directed=False)
        parea = np.bincount(lab[member], weights=area[member], minlength=int(lab.max()) + 1)
        pcount = np.bincount(lab[member], minlength=int(lab.max()) + 1)
        out |= member & (parea[lab] >= P.planar_min_area) & (pcount[lab] >= P.planar_min_faces)
    return out


def boundary_stats(topo, area, valid, soft, grp, cl, ncomp, gsign, ignore=None, same_level=None, min_width=0.0):
    """Area and boundary composition of every component.

    Boundary edges whose outside neighbour is a wall (steeper than the wall slope,
    or the open edge of the mesh) are ``hard``.  Edges against any other face (a
    soft face outside the window, another level) are a ``cut``: the surface goes on
    smoothly beyond the plateau there.  ``same_level(face_idx, group)`` (optional)
    marks neighbours that are part of the same plateau (an exactly-flat patch the
    noise fades into): they count as ``hard`` too, but only for components whose mean
    width (2 x area / perimeter) is at least ``min_width``: a thin strip along a
    plateau is the shoulder of a ramp or skirt, not noise fading into the plateau.
    """
    sel = cl >= 0
    carea = np.bincount(cl[sel], weights=area[sel], minlength=ncomp)
    comp_g = np.zeros(ncomp, dtype=np.int64)
    comp_g[cl[sel]] = grp[sel]
    sgn_c = gsign[comp_g]
    cut = np.zeros(ncomp)
    hard = np.zeros(ncomp)
    same = np.zeros(ncomp)
    fa, fb, el = topo.fa, topo.fb, topo.elen
    for A, B in ((fa, fb), (fb, fa)):
        m = (cl[A] >= 0) & (cl[B] != cl[A]) & valid[B]
        if ignore is not None:
            m &= ~ignore[B]
        cA = cl[A[m]]
        sB = sgn_c[cA]
        is_soft = np.where(sB > 0, soft[+1][B[m]], soft[-1][B[m]])
        ln = el[m]
        is_same = np.zeros(len(ln), bool)
        if same_level is not None:
            is_same = is_soft & same_level(B[m], comp_g[cA])
        is_cut = is_soft & ~is_same
        is_hard = ~is_soft
        np.add.at(same, cA[is_same], ln[is_same])
        np.add.at(cut, cA[is_cut], ln[is_cut])
        np.add.at(hard, cA[is_hard], ln[is_hard])
    of = topo.open_face
    k = cl[of] >= 0
    np.add.at(hard, cl[of[k]], topo.open_len[k])
    perim = cut + hard + same
    width = np.divide(2.0 * carea, perim, out=np.zeros(ncomp), where=perim > 0)
    broad = width >= min_width
    cut = cut + np.where(broad, 0.0, same)
    hard = hard + np.where(broad, same, 0.0)
    tot = cut + hard
    # nothing left to judge (every boundary edge faces an ignored, fillable hole bounded by walls)
    # means the plateau is closed by walls, not that it continues smoothly
    cut_frac = np.divide(cut, tot, out=(np.zeros(ncomp) if ignore is not None else np.ones(ncomp)), where=tot > 0)
    return dict(area=carea, cut=cut, hard=hard, cut_frac=cut_frac, comp_group=comp_g)


def fillable_holes(grp, is_core_f, topo, F, area, zc, gsign, soft, exact, ramp, gl, gh, G, P: Params):
    """Unassigned soft patches that are internal holes of a single plateau window.

    A noisy panel is carved up by its steepest facets into rings and islands.  A
    hole is a connected set of soft (non-wall), non-exact, non-ramp faces outside
    every window.  It is fillable if it is bounded only by walls and by plateau
    faces of ONE group, its heights stay within ``tail_ext`` of that window and it
    is not larger than ``hole_area``.  With ``is_core_f`` given it must also touch
    a verified plateau face.  Returns ``(face mask, group per face)``.
    """
    fill = np.zeros(F, bool)
    fgroup = -np.ones(F, dtype=np.int64)
    if G == 0:
        return fill, fgroup
    sgn_f = np.where(grp >= 0, gsign[np.maximum(grp, 0)], 0)
    fa, fb = topo.fa, topo.fb
    for s in (+1, -1):
        holes = soft[s] & ~exact & ~ramp & (grp < 0)
        if not holes.any():
            continue
        nh, hl = components(topo, holes)
        harea = np.bincount(hl[holes], weights=area[holes], minlength=nh)
        ph, pg_, pcore = [], [], []
        for A_, B_ in ((fa, fb), (fb, fa)):
            m = holes[A_] & ~holes[B_]
            b_ = B_[m]
            is_pl = (grp[b_] >= 0) & (sgn_f[b_] == s)
            is_wall = ~soft[s][b_]
            ph.append(hl[A_[m]])
            pg_.append(np.where(is_pl, grp[b_], np.where(is_wall, -2, -1)))
            pcore.append(is_pl & (is_core_f[b_] if is_core_f is not None else True))
        ph = np.concatenate(ph)
        pg_ = np.concatenate(pg_)
        pcore = np.concatenate(pcore)
        bad = np.zeros(nh, bool)
        bad[ph[pg_ == -1]] = True
        hascore = np.zeros(nh, bool)
        hascore[ph[pcore]] = True
        hgroup = -np.ones(nh, dtype=np.int64)
        multi = np.zeros(nh, bool)
        okp = pg_ >= 0
        if okp.any():
            key = ph[okp] * np.int64(G + 1) + pg_[okp]
            uk = np.unique(key)
            uh, ug = uk // (G + 1), uk % (G + 1)
            multi = np.bincount(uh, minlength=nh) > 1
            hgroup[uh] = ug
        hz_lo = np.full(nh, np.inf)
        hz_hi = np.full(nh, -np.inf)
        np.minimum.at(hz_lo, hl[holes], zc[holes])
        np.maximum.at(hz_hi, hl[holes], zc[holes])
        gg = np.maximum(hgroup, 0)
        ext = P.tail_ext
        near = (hz_lo >= gl[gg] - ext) & (hz_hi <= gh[gg] + ext)
        good = (~bad) & hascore & (~multi) & (hgroup >= 0) & (harea <= P.hole_area) & near
        if good.any():
            sel = holes & good[np.maximum(hl, 0)]
            fill |= sel
            fgroup[sel] = hgroup[hl[sel]]
    return fill, fgroup


def attached(faces, grp, cl, is_core, nverts):
    """Components (not in ``is_core``) that share a vertex with a core component of the same group."""
    sel = cl >= 0
    core_f = np.zeros(len(cl), bool)
    core_f[sel] = is_core[cl[sel]]
    vg = -np.ones(nverts, dtype=np.int64)
    vf = faces[core_f]
    vg[vf.ravel()] = np.repeat(grp[core_f], 3)
    hit = (vg[faces] == grp[:, None]).any(axis=1) & sel & ~core_f
    out = np.zeros(len(is_core), bool)
    out[cl[hit]] = True
    return out & ~is_core


# --------------------------------------------------------------------- analysis
@dataclass
class Analysis:
    """Everything :func:`analyse` finds.  Face arrays have one entry per face."""

    P: Params
    topo: Topo
    normals: np.ndarray  # raw unit face normals
    snormals: np.ndarray  # smoothed unit face normals (used for classification)
    area: np.ndarray
    zc: np.ndarray  # face centroid height (bed relative)
    valid: np.ndarray
    exact: np.ndarray  # intentional exactly-flat faces
    exact_demoted: np.ndarray  # exactly-flat faces continued smoothly by noisy ones (ordinary candidates)
    exact_raw: np.ndarray  # every exactly-flat face, intentional or not (their vertices are anchors)
    exact_label: np.ndarray  # connected exactly-flat patch of every exact face (-1 elsewhere), per orientation
    exact_sign: np.ndarray  # +1 top / -1 ceiling per patch
    exact_area: np.ndarray
    exact_z: np.ndarray  # area-weighted height per patch
    bed: np.ndarray
    soft: dict
    ramp: np.ndarray
    plateau_group: np.ndarray  # accepted plateau window per face (-1 none)
    plateau_comp: np.ndarray  # connected patch per accepted face (-1 none)
    clevel: np.ndarray  # final snapped level per patch
    glevel: np.ndarray  # snapped level per window group
    gsign: np.ndarray
    windows: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # (sign, window dict, reason)
    rejected_components: list = field(default_factory=list)  # (group, area, cut_frac, reason)
    already_flat: list = field(default_factory=list)  # (sign, level, area): plateaus that need no change


def analyse(mesh: Mesh, grid: LayerGrid, P: Params) -> Analysis:
    """Find the plateaus of ``mesh`` (z measured from the bed)."""
    h = grid.layer_height
    verts, faces = mesh.verts, mesh.faces
    F = len(faces)
    z = verts[:, 2]
    n, area = mesh.face_normals_areas()
    tz = z[faces]
    zc = tz.mean(1)
    zspan = tz.max(1) - tz.min(1)
    tiny = max(1e-9, 1e-6 * float(np.median(area[area > 0])) if (area > 0).any() else 1e-9)
    valid = area > tiny
    bed = zc <= 0.5 * grid.first
    ns = smoothed_normals(mesh, n, area, valid, P)
    nz = ns[:, 2]
    cos_max = np.cos(np.radians(P.max_slope_deg))
    cos_wall = np.cos(np.radians(P.wall_slope_deg))
    topo = Topo(mesh)

    soft = {s: valid & (s * nz >= cos_wall) & ~bed for s in (+1, -1)}

    # -- exactly flat faces: intentional unless noisy faces continue them smoothly
    # "exactly flat" means equal up to float32 noise: a few ulps at this height, never less than flat_tol
    ulp = np.spacing(np.abs(tz).max(axis=1).astype(np.float32)).astype(np.float64)
    flat_tol = np.maximum(P.flat_tol, 4.04 * ulp)
    exact_raw = valid & (zspan <= flat_tol) & (np.abs(n[:, 2]) > cos_wall)
    exact = exact_raw.copy()
    demoted = np.zeros(F, bool)
    if exact_raw.any():
        for s in (+1, -1):
            mem = exact_raw & (s * n[:, 2] > 0)
            if not mem.any():
                continue
            ne, el = components(topo, mem)
            st = boundary_stats(topo, area, valid, soft, np.zeros(F, dtype=np.int64), el, ne, np.array([s]))
            perim = st["cut"] + st["hard"]
            width = np.divide(2.0 * st["area"], perim, out=np.zeros(ne), where=perim > 0)
            # thin exactly-flat strips continued smoothly by noisy faces are terraces of a
            # quantised or clamped surface; a broad exactly-flat plateau is a real one
            bad = (st["cut_frac"] > P.exact_max_cut) & (width < P.exact_demote_width)
            sel = el >= 0
            demoted[sel] = bad[el[sel]]
        exact &= ~demoted

    # connected exactly-flat patches, per orientation (a stray flipped sliver must not
    # change the orientation of a whole patch)
    exact_label = -np.ones(F, dtype=np.int64)
    ex_sign, ex_area, ex_z = [], [], []
    for s in (+1, -1):
        mem = exact & (s * n[:, 2] > 0)
        if not mem.any():
            continue
        ne, el = components(topo, mem)
        base = len(ex_sign)
        exact_label[mem] = el[mem] + base
        ca = np.bincount(el[mem], weights=area[mem], minlength=ne)
        cz = np.bincount(el[mem], weights=(area * zc)[mem], minlength=ne) / np.maximum(ca, 1e-12)
        ex_sign.extend([s] * ne)
        ex_area.extend(ca.tolist())
        ex_z.extend(cz.tolist())

    ramp = np.zeros(F, bool)
    if P.planar_ramp_deg is not None:
        ramp = planar_ramps(topo, n, area, valid, P)
        ramp &= ~exact_raw

    grp = -np.ones(F, dtype=np.int64)
    windows: list[dict] = []
    rejected: list = []
    wts = area * np.abs(n[:, 2])
    for s in (+1, -1):
        cand = valid & (s * nz >= cos_max) & ~exact & ~bed & ~ramp
        if not cand.any():
            continue
        wins = find_windows(zc[cand], wts[cand], h, P)
        kept = []
        ci_all = np.flatnonzero(cand)
        for w in wins:
            w["sign"] = s
            # centroid heights hide the rise WITHIN a big face: add the faces' own z-extent
            inw = ci_all[(zc[ci_all] >= w["lo"]) & (zc[ci_all] <= w["hi"])]
            if len(inw):
                w["span"] += float(np.percentile(zspan[inw], 99))
            if w["mass"] < P.min_area:
                rejected.append((s, w, "too small"))
            elif w["span"] > P.max_range:
                rejected.append((s, w, f"height range {w['span']:.2f} mm exceeds max_range"))
            elif w["dens"] < P.min_prominence * w["ped"]:
                rejected.append((s, w, "no distinct peak (looks like a ramp or dome)"))
            elif w["span"] > P.flat_top_range and w["peaky"] < P.min_peakiness:
                rejected.append((s, w, f"heights spread evenly over {w['span']:.2f} mm (a ramp or curved surface)"))
            else:
                kept.append(w)
        if not kept:
            continue
        kept.sort(key=lambda w: w["lo"])
        lo = np.array([w["lo"] for w in kept])
        hi = np.array([w["hi"] for w in kept])
        ci = np.flatnonzero(cand)
        k = np.searchsorted(lo, zc[ci], side="right") - 1
        ok = (k >= 0) & (zc[ci] <= hi[np.maximum(k, 0)])
        base = len(windows)
        grp[ci[ok]] = base + k[ok]
        windows.extend(kept)

    G = len(windows)
    gsign = np.array([w["sign"] for w in windows], dtype=np.int64) if G else np.zeros(0, np.int64)
    gl = np.array([w["lo"] for w in windows])
    gh = np.array([w["hi"] for w in windows])
    tol_level = 0.25 * h

    def same_level(fidx, g):
        """Exactly-flat faces whose height lies inside window ``g``: the same plateau."""
        return exact[fidx] & (zc[fidx] >= gl[g] - tol_level) & (zc[fidx] <= gh[g] + tol_level)

    hole_args = (topo, F, area, zc, gsign, soft, exact, ramp, gl, gh, G, P)

    # -- stage 1: components and their standalone verdict
    ncomp0, cl0 = components(topo, grp >= 0, same=grp)
    core0 = np.zeros(ncomp0, dtype=bool)
    attached0 = np.zeros(ncomp0, dtype=bool)
    if ncomp0:
        st0 = boundary_stats(topo, area, valid, soft, grp, cl0, ncomp0, gsign, same_level=same_level, min_width=P.same_level_width)
        core0 = (st0["area"] >= P.min_area) & (st0["cut_frac"] <= P.core_cut_frac)
        # A big component gets the benefit of the doubt: judge it with the internal
        # holes (carved out by its steepest noise facets) not counted as cut.
        big0 = (st0["area"] >= P.core_big_area) & ~core0
        if big0.any():
            fillable, _ = fillable_holes(grp, None, *hole_args)
            stb = boundary_stats(topo, area, valid, soft, grp, cl0, ncomp0, gsign, ignore=fillable, same_level=same_level, min_width=P.same_level_width)
            core0 |= big0 & (stb["cut_frac"] <= P.core_cut_frac)
        attached0 = attached(faces, grp, cl0, core0, mesh.n_verts)

    # -- stage 2: fill internal holes of verified plateaus
    if G and core0.any():
        seed_f = np.zeros(F, bool)
        seed_f[cl0 >= 0] = (core0 | attached0)[cl0[cl0 >= 0]]
        for _ in range(P.hole_rounds):
            ncur, ccur = components(topo, grp >= 0, same=grp)
            ccore = np.zeros(ncur, bool)
            ccore[ccur[seed_f & (ccur >= 0)]] = True
            is_core_f = np.zeros(F, bool)
            is_core_f[ccur >= 0] = ccore[ccur[ccur >= 0]]
            fill, fgroup = fillable_holes(grp, is_core_f, *hole_args)
            if not fill.any():
                break
            grp[fill] = fgroup[fill]

    # -- stage 3: final components, accept / reject
    ncomp, cl = components(topo, grp >= 0, same=grp)
    accepted = np.zeros(ncomp, dtype=bool)
    comp_group = np.zeros(ncomp, dtype=np.int64)
    rej_comp: list = []
    if ncomp:
        stats = boundary_stats(topo, area, valid, soft, grp, cl, ncomp, gsign, same_level=same_level, min_width=P.same_level_width)
        comp_group = stats["comp_group"]
        main = (stats["area"] >= P.min_area) & (stats["cut_frac"] <= P.max_cut_frac)
        att = attached(faces, grp, cl, main, mesh.n_verts)
        accepted = main | (att & (stats["cut_frac"] <= P.max_cut_frac))
        for c in np.flatnonzero(~accepted):
            if stats["area"][c] < P.min_area:
                why = "too small"
            else:
                why = f"{100 * stats['cut_frac'][c]:.0f}% of its edge continues smoothly (ramp, skirt or dome flank)"
            rej_comp.append((int(stats["comp_group"][c]), float(stats["area"][c]), float(stats["cut_frac"][c]), why))
    keep_f = cl >= 0
    keep_f[keep_f] = accepted[cl[keep_f]]
    plateau_group = np.where(keep_f, grp, -1)

    # -- a plateau that already prints on a single layer has no layer lines: leave it alone
    already: list = []
    live0 = plateau_group >= 0
    if G and live0.any():
        gid = plateau_group[live0]
        fz = tz[live0]
        o = np.argsort(gid, kind="stable")
        gs = gid[o]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        ug = gs[starts]
        gmin = np.minimum.reduceat(fz.min(1)[o], starts)
        gmax = np.maximum.reduceat(fz.max(1)[o], starts)
        mg = P.plane_margin * grid.layer_height + 1e-6
        one = grid.layer_number(gmin - mg) == grid.layer_number(gmax + mg)
        if one.any():
            garea = np.bincount(gid, weights=area[live0], minlength=G)
            gmed = grouped_wmedian(gid, zc[live0], area[live0], G)
            for g in ug[one].tolist():
                already.append((int(gsign[g]), float(gmed[g]), float(garea[g])))
            kill = np.zeros(G, bool)
            kill[ug[one]] = True
            plateau_group[live0 & kill[np.maximum(plateau_group, 0)]] = -1

    # -- level of each group: snapped weighted median of the accepted faces; components
    #    that meet at a deliberate step keep their height relative to their neighbour
    glevel = np.full(G, np.nan)
    live1 = plateau_group >= 0
    if G and live1.any():
        gmed = grouped_wmedian(plateau_group[live1], zc[live1], area[live1], G)
        has = ~np.isnan(gmed)
        glevel[has] = grid.snap(gmed[has])
    acc = np.zeros(ncomp, dtype=bool)
    live = plateau_group >= 0
    if ncomp and live.any():
        acc[np.unique(cl[live])] = True
    clevel = plateau_levels(
        mesh, cl, acc, comp_group, zc, area, soft, valid,
        np.nan_to_num(glevel, nan=0.0), grid, P,
    ) if ncomp else np.zeros(0)
    plateau_comp = np.where(live, cl, -1)
    return Analysis(
        P=P, topo=topo, normals=n, snormals=ns, area=area, zc=zc, valid=valid, exact=exact,
        exact_demoted=demoted, exact_raw=exact_raw, exact_label=exact_label, exact_sign=np.array(ex_sign, dtype=np.int64),
        exact_area=np.array(ex_area), exact_z=np.array(ex_z), bed=bed, soft=soft, ramp=ramp,
        plateau_group=plateau_group, plateau_comp=plateau_comp, clevel=clevel, glevel=glevel,
        gsign=gsign, windows=windows, rejected=rejected,
        rejected_components=rej_comp, already_flat=already,
    )
