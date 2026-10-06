"""Objective scoring of a smoothing result against a ``scenes.Scene``."""

from __future__ import annotations

import numpy as np

from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh
from stl_smoothing.slicing import contour_length, contour_segments, slice_planes  # noqa: F401


def layers_spanned(mesh: Mesh, grid: LayerGrid, face_mask: np.ndarray,
                   verts: np.ndarray | None = None, min_frac: float = 0.002) -> int:
    """Number of distinct layers the selected faces end on (area weighted)."""
    v = mesh.verts if verts is None else verts
    _, area = mesh.face_normals_areas(v)
    zc = v[mesh.faces].mean(axis=1)[:, 2]
    layer = grid.layer_number(zc)
    sel = np.flatnonzero(face_mask)
    if len(sel) == 0:
        return 0
    tot = area[sel].sum()
    ids, inv = np.unique(layer[sel], return_inverse=True)
    w = np.bincount(inv, weights=area[sel])
    return int((w > min_frac * tot).sum())


def target_flat_ok(mesh: Mesh, new_verts: np.ndarray, face_mask: np.ndarray, tol: float = 1e-4):
    """Largest vertex-height spread inside each connected target component."""
    ncomp, lab = mesh.face_components(face_mask)
    worst = 0.0
    for c in range(ncomp):
        vs = np.unique(mesh.faces[lab == c])
        z = new_verts[vs, 2]
        worst = max(worst, float(z.max() - z.min()))
    return worst


def evaluate(scene, new_verts: np.ndarray, grid: LayerGrid | None = None) -> dict:
    """Score ``new_verts`` (same connectivity as ``scene.mesh``)."""
    grid = grid or LayerGrid(0.2)
    m = scene.mesh
    res: dict = {"scene": scene.name}
    # --- geometry sanity
    res["xy_moved_max"] = float(np.abs(new_verts[:, :2] - m.verts[:, :2]).max())
    res["flipped"] = int(m.flipped_faces(new_verts).sum())
    res["max_dz"] = float(np.abs(new_verts[:, 2] - m.verts[:, 2]).max())
    res["n_changed"] = int((np.abs(new_verts[:, 2] - m.verts[:, 2]) > 1e-9).sum())
    # collateral deformation: how much did the surface normal turn on faces that
    # are NOT supposed to be flattened (walls stretched by a moved neighbour,
    # creases, ...)?  Reported as degrees (max and 99.9th percentile).
    n0, a0 = m.face_normals_areas()
    n1, a1 = m.face_normals_areas(new_verts)
    tmask = np.zeros(m.n_faces, bool)
    for mask in scene.targets.values():
        tmask |= mask
    other = ~tmask & (a0 > 1e-9)
    ang = np.degrees(np.arccos(np.clip((n0 * n1).sum(1), -1, 1)))
    res["normal_turn_max"] = float(ang[other].max()) if other.any() else 0.0
    res["normal_turn_p999"] = float(np.percentile(ang[other], 99.9)) if other.any() else 0.0
    # --- targets
    tg = {}
    for name, mask in scene.targets.items():
        before = contour_length(m, grid, mask)
        after = contour_length(m, grid, mask, new_verts)
        tg[name] = {
            "contour_before": before,
            "contour_after": after,
            "layers_before": layers_spanned(m, grid, mask),
            "layers_after": layers_spanned(m, grid, mask, new_verts),
            "spread_after": target_flat_ok(m, new_verts, mask),
        }
        if name in scene.expect_levels:
            zc = new_verts[np.unique(m.faces[mask]), 2]
            tg[name]["level_err"] = float(abs(np.median(zc) - scene.expect_levels[name]))
    res["targets"] = tg
    # --- protected features
    pr = {}
    for name, (vmask, tol) in scene.protected.items():
        d = np.abs(new_verts[vmask, 2] - m.verts[vmask, 2])
        pr[name] = {"max_drift": float(d.max()) if d.size else 0.0, "tol": tol,
                    "ok": bool((d <= tol).all())}
    res["protected"] = pr
    # --- on-grid: all target vertices on layer boundaries?
    return res


def verdict(res: dict, contour_eps_mm: float = 0.5) -> tuple[bool, list[str]]:
    """Pass / fail summary with the reasons for failure."""
    why = []
    if res["flipped"]:
        why.append(f"{res['flipped']} flipped faces")
    if res["xy_moved_max"] > 1e-9:
        why.append("xy moved")
    for n, t in res["targets"].items():
        if t["layers_after"] != 1:
            why.append(f"target {n}: ends on {t['layers_after']} layers")
        if t["contour_after"] > contour_eps_mm:
            why.append(f"target {n}: {t['contour_after']:.1f} mm of layer edges remain")
    for n, p in res["protected"].items():
        if not p["ok"]:
            why.append(f"protected {n}: drift {p['max_drift']:.3f} > {p['tol']}")
    return (not why), why
