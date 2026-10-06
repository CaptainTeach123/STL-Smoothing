"""Public entry point: flatten almost-flat surfaces onto single layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._deform import exact_needs_snap, feather, repair_flips, unfold, vertex_targets
from ._detect import Analysis, analyse, components, wmedian
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


def _describe(res: FlattenResult, work: Mesh, grid: LayerGrid, A: Analysis, P: Params) -> None:
    faces = work.faces
    z = work.verts[:, 2]
    F = work.n_faces
    pg = A.plateau_group
    res.smoothed_faces = pg >= 0
    res.snapped_faces = np.zeros(F, bool)
    tz = z[faces]

    # -- smoothed plateaus: one entry per (orientation, final level); the patches of a
    #    window group share a level except across a deliberate step
    pc = A.plateau_comp
    if (pc >= 0).any():
        fi = np.flatnonzero(pc >= 0)
        face_sign = np.where(A.gsign[pg[fi]] > 0, 1, -1)
        face_level = np.round(A.clevel[pc[fi]], 6)
        keys = {}
        for k, (sg, lv) in enumerate(zip(face_sign.tolist(), face_level.tolist())):
            keys.setdefault((sg, lv), []).append(k)
        for (sg, lv), ks in sorted(keys.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            m = np.zeros(F, bool)
            m[fi[ks]] = True
            area = A.area[m]
            vs = np.unique(faces[m])
            zlo, zhi = np.percentile(z[vs], [0.5, 99.5])
            ncomp = len(np.unique(pc[m]))
            res.plateaus.append(
                Plateau(
                    kind="smoothed",
                    facing="up" if sg > 0 else "down",
                    regions=int(ncomp),
                    area=float(area.sum()),
                    level_before=float(wmedian(A.zc[m], area)),
                    z_low=float(zlo),
                    z_high=float(zhi),
                    level_after=float(lv),
                    layers_before=_layer_count(grid, A.zc[m], area),
                    layers_after=1,
                )
            )

    # -- exactly flat plateaus that were nudged onto the grid
    if P.snap_exact_flat and len(A.exact_area):
        need = exact_needs_snap(A, grid, P)
        for c in np.flatnonzero(need):
            res.snapped_faces |= A.exact_label == c
            res.plateaus.append(
                Plateau(
                    kind="snapped",
                    facing="up" if A.exact_sign[c] > 0 else "down",
                    regions=1,
                    area=float(A.exact_area[c]),
                    level_before=float(A.exact_z[c]),
                    z_low=float(A.exact_z[c]),
                    z_high=float(A.exact_z[c]),
                    level_after=float(grid.snap(A.exact_z[c])),
                    layers_before=1,
                    layers_after=1,
                )
            )

    for sign, lvl, area in A.already_flat:
        res.already_flat.append((("top" if sign > 0 else "ceiling"), lvl, area))

    # -- what was found but left alone, and why
    for sign, w, why in A.rejected:
        res.skipped.append(
            f"{'top' if sign > 0 else 'ceiling'} surface near {w['rep']:.2f} mm "
            f"({w['mass']:.0f} mm²): {why}"
        )
    for g, area, cut, why in A.rejected_components:
        lvl = A.windows[g]["rep"] if g < len(A.windows) else float("nan")
        res.skipped.append(
            f"{'top' if A.gsign[g] > 0 else 'ceiling'} patch near {lvl:.2f} mm ({area:.0f} mm²): {why}"
        )
