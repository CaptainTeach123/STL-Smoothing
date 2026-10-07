"""Public entry point: flatten almost-flat surfaces onto single layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._deform import exact_needs_snap, feather, repair_flips, unfold, vertex_targets
from ._detect import Analysis, analyse, components, wmedian, grouped_wmedian
from .layers import LayerGrid
from .mesh import Mesh
from .params import Params


@dataclass
class Plateau:
    """One flattened (or snapped) surface, reported in heights above the bed."""

    kind: str  # "smoothed": noisy surface made flat; "snapped": already flat, moved onto the layer grid
    facing: str  # "up" (a top) or "down" (a ceiling / underside)
    regions: int  # number of separate patches
    area: float  # mm^2 (true surface area)
    level_before: float  # area-weighted median height before
    z_low: float  # robust extent before (0.5 % / 99.5 % of the vertex heights)
    z_high: float
    level_after: float
    layers_before: int  # distinct layers the surface ended on before
    layers_after: int

    @property
    def changed(self) -> bool:
        return abs(self.level_after - self.level_before) > 1e-6 or self.z_high - self.z_low > 1e-6


@dataclass
class FlattenResult:
    verts: np.ndarray  # (V, 3) new vertices, same order as the input mesh, original coordinates
    plateaus: list[Plateau] = field(default_factory=list)
    smoothed_faces: np.ndarray | None = None  # bool per face: part of a smoothed plateau
    snapped_faces: np.ndarray | None = None  # bool per face: exactly flat, snapped
    facing: np.ndarray | None = None  # int8 per face: +1 plateau on a top, -1 on a ceiling, 0 none
    face_plateau: np.ndarray | None = None  # per face: index into ``plateaus`` (-1 none)
    skipped: list[str] = field(default_factory=list)  # candidates that were left alone, with the reason
    already_flat: list = field(default_factory=list)  # (facing, level, area) of surfaces that need no change
    z_offset: float = 0.0  # lowest point of the input (heights are measured from it)
    max_dz: float = 0.0
    n_moved: int = 0
    flipped_faces: int = 0
    degenerate_faces_added: int = 0

    @property
    def region_faces(self) -> np.ndarray:
        """Faces of every plateau the algorithm touched."""
        out = np.zeros(len(self.smoothed_faces), bool)
        out |= self.smoothed_faces
        out |= self.snapped_faces
        return out


def flatten(mesh: Mesh, grid: LayerGrid | None = None, params: Params | None = None, **overrides) -> FlattenResult:
    """Flatten the almost-flat surfaces of ``mesh``.

    Heights are measured from the lowest point of the mesh (slicers drop a part
    onto the bed), the layer grid ``grid`` is laid out from there, and every
    plateau is moved, in z only, onto a layer boundary.  ``overrides`` are
    :class:`Params` field names.
    """
    grid = grid or LayerGrid()
    P = params or Params()
    if overrides:
        unknown = set(overrides) - Params.names()
        if unknown:
            raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        P = P.replace(**overrides)

    if not np.isfinite(mesh.verts).all():
        raise ValueError("the mesh has NaN or infinite coordinates")
    if mesh.n_faces == 0:
        return FlattenResult(verts=mesh.verts.copy(), smoothed_faces=np.zeros(0, bool), snapped_faces=np.zeros(0, bool))

    z_off = float(mesh.verts[:, 2].min())
    work = Mesh(mesh.verts - np.array([0.0, 0.0, z_off]), mesh.faces)
    A = analyse(work, grid, P)
    tgt = vertex_targets(work, grid, A, P)
    pinned = ~np.isnan(tgt)
    z = work.verts[:, 2]
    bed_v = z <= 0.5 * grid.first  # bed contact never moves
    # Vertices of exactly flat faces are anchors: a flat top that is already flat
    # (a raised block, a deliberate shelf) keeps its height even when a noisy
    # neighbour moves, unless it is itself pinned to a plateau.
    anchor_v = np.zeros(work.n_verts, bool)
    anchor_v[work.faces[A.exact_raw].ravel()] = True
    anchor_v &= ~pinned
    fixed_v = bed_v | anchor_v
    dz = np.where(pinned, tgt - z, 0.0)
    dz[bed_v] = 0.0
    delta = feather(work, A, dz, pinned & ~bed_v, fixed_v, P)
    new_z = unfold(work, A, z, z + delta, pinned | fixed_v, P)
    new_z = repair_flips(work, new_z, z, new_z - z, P)
    new_z = np.maximum(new_z, 0.0)  # the wall-order projection must never push anything below the bed

    # Vertices that moved are rounded to float32, which is what an STL stores: all
    # vertices of a plateau then share the very same value, and a vertex that is
    # already where it should be stays bit-identical to the input.
    out = mesh.verts.copy()
    moved = np.abs(new_z - z) > 1e-9
    out[moved, 2] = (new_z[moved] + z_off).astype(np.float32).astype(np.float64)
    moved = out[:, 2] != mesh.verts[:, 2]

    new_work = work.verts.copy()
    new_work[:, 2] = new_z
    flips = int(work.flipped_faces(new_work).sum())
    deg_before = work.degenerate_faces(work.verts, tol=1e-9)
    deg_after = work.degenerate_faces(new_work, tol=1e-9)

    res = FlattenResult(
        verts=out,
        z_offset=z_off,
        max_dz=float(np.abs(out[:, 2] - mesh.verts[:, 2]).max()),
        n_moved=int(moved.sum()),
        flipped_faces=flips,
        degenerate_faces_added=int((deg_after & ~deg_before).sum()),
    )
    _describe(res, work, grid, A, P)
    return res


# -------------------------------------------------------------------- reporting
def _layer_count(grid: LayerGrid, zc: np.ndarray, w: np.ndarray, min_frac: float = 0.005) -> int:
    ends = np.round(grid.layer_end(zc), 6)
    ids, inv = np.unique(ends, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    return int((mass > min_frac * mass.sum()).sum())


def _grouped_percentiles(kid, vals, K, qs):
    """np.percentile(vals[kid == k], qs) (linear) for every k, vectorised.  kid in 0..K-1, every k non-empty."""
    o = np.lexsort((vals, kid))
    v = vals[o]
    cnt = np.bincount(kid, minlength=K)
    start = np.cumsum(cnt) - cnt
    out = []
    for q in qs:
        pos = (q / 100.0) * (cnt - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, cnt - 1)
        frac = pos - lo
        a, b = v[start + lo], v[start + hi]
        out.append(a + (b - a) * frac)
    return out


def _describe(res: FlattenResult, work: Mesh, grid: LayerGrid, A: Analysis, P: Params) -> None:
    faces = work.faces
    z = work.verts[:, 2]
    F = work.n_faces
    pg = A.plateau_group
    res.smoothed_faces = pg >= 0
    res.snapped_faces = np.zeros(F, bool)
    res.facing = np.zeros(F, dtype=np.int8)
    res.face_plateau = -np.ones(F, dtype=np.int64)
    live = pg >= 0
    res.facing[live] = np.where(A.gsign[pg[live]] > 0, 1, -1)

    pc = A.plateau_comp
    fi = np.flatnonzero(pc >= 0)
    if len(fi):
        sg = np.where(A.gsign[pg[fi]] > 0, 1, -1)
        lv = np.round(A.clevel[pc[fi]], 6)
        # group id of (sign, level), numbered in the order the original loop reported them: (-sign, level)
        o = np.lexsort((lv, -sg))
        sg_s, lv_s = sg[o], lv[o]
        new = np.r_[True, (sg_s[1:] != sg_s[:-1]) | (lv_s[1:] != lv_s[:-1])]
        kid_sorted = np.cumsum(new) - 1
        kid = np.empty(len(fi), np.int64)
        kid[o] = kid_sorted
        K = int(kid_sorted[-1]) + 1
        k_sign = sg_s[new]
        k_level = lv_s[new]
        area = A.area[fi]
        zc = A.zc[fi]
        k_area = np.bincount(kid, weights=area, minlength=K)
        # robust z extent over the UNIQUE vertices of each group
        V = work.n_verts
        pk = np.unique(np.repeat(kid, 3) * np.int64(V) + faces[fi].ravel())
        vk, vv = pk // V, pk % V
        zlo, zhi = _grouped_percentiles(vk, z[vv], K, (0.5, 99.5))
        # number of patches
        ncomp_total = int(pc.max()) + 1
        pcm = np.unique(kid * np.int64(ncomp_total) + pc[fi])
        k_ncomp = np.bincount(pcm // ncomp_total, minlength=K)
        k_med = grouped_wmedian(kid, zc, area, K)
        # layers before: distinct layer ends carrying > 0.5 % of the group's area
        ends = np.round(grid.layer_end(zc), 6)
        ue, einv = np.unique(ends, return_inverse=True)
        mass = np.bincount(kid * np.int64(len(ue)) + einv, weights=area, minlength=K * len(ue)).reshape(K, len(ue))
        k_layers = (mass > 0.005 * mass.sum(axis=1, keepdims=True)).sum(axis=1)
        base = len(res.plateaus)
        res.face_plateau[fi] = base + kid
        for k in range(K):  # O(number of plateaus), pure Python scalars only
            res.plateaus.append(Plateau(
                kind="smoothed", facing="up" if k_sign[k] > 0 else "down", regions=int(k_ncomp[k]),
                area=float(k_area[k]), level_before=float(k_med[k]), z_low=float(zlo[k]), z_high=float(zhi[k]),
                level_after=float(k_level[k]), layers_before=int(k_layers[k]), layers_after=1))

    if P.snap_exact_flat and len(A.exact_area):
        need = exact_needs_snap(A, grid, P)
        cidx = np.flatnonzero(need)
        if len(cidx):
            lab = A.exact_label
            sel = (lab >= 0) & need[np.maximum(lab, 0)]
            pidx = np.full(len(need), -1, np.int64)
            pidx[cidx] = len(res.plateaus) + np.arange(len(cidx))
            res.snapped_faces = sel
            res.facing[sel] = np.where(A.exact_sign[lab[sel]] > 0, 1, -1)
            res.face_plateau[sel] = pidx[lab[sel]]
            lvl = grid.snap(A.exact_z[cidx])
            for j, c in enumerate(cidx.tolist()):
                res.plateaus.append(Plateau(
                    kind="snapped", facing="up" if A.exact_sign[c] > 0 else "down", regions=1,
                    area=float(A.exact_area[c]), level_before=float(A.exact_z[c]), z_low=float(A.exact_z[c]),
                    z_high=float(A.exact_z[c]), level_after=float(lvl[j]), layers_before=1, layers_after=1))

    for sign, lvl, area in A.already_flat:
        res.already_flat.append((("top" if sign > 0 else "ceiling"), lvl, area))
    min_report = 0.25 * P.min_area
    for sign, w, why in A.rejected:
        if w["mass"] < min_report:
            continue
        res.skipped.append(f"{'top' if sign > 0 else 'ceiling'} surface near {w['rep']:.2f} mm ({w['mass']:.0f} mm²): {why}")
    for g, area, cut, why in A.rejected_components:
        if area < min_report:
            continue
        lvl = A.windows[g]["rep"] if g < len(A.windows) else float("nan")
        res.skipped.append(f"{'top' if A.gsign[g] > 0 else 'ceiling'} patch near {lvl:.2f} mm ({area:.0f} mm²): {why}")
