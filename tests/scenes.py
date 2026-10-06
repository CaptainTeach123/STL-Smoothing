"""Procedural test scenes with known ground truth.

Each scene is a closed, manifold, outward-wound triangle mesh built from a
top height field ``ztop(x, y)`` and a bottom height field ``zbot(x, y)``
joined by vertical side walls.  Scenes carry:

* ``targets``   - face masks that *should* end up on a single layer
* ``protected`` - vertex masks that must not move (more than ``tol`` mm in z)

so any candidate algorithm can be scored objectively (see ``metrics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from stl_smoothing.mesh import Mesh


# --------------------------------------------------------------------- noise
class WaveField:
    """Smooth pseudo-random field built from a sum of plane waves."""

    def __init__(self, seed: int, wavelengths=(20.0, 70.0), n_waves: int = 10):
        rng = np.random.default_rng(seed)
        lam = rng.uniform(*wavelengths, n_waves)
        th = rng.uniform(0, 2 * np.pi, n_waves)
        self.k = np.stack([np.cos(th), np.sin(th)], axis=1) * (2 * np.pi / lam)[:, None]
        self.phase = rng.uniform(0, 2 * np.pi, n_waves)
        self.amp = rng.uniform(0.5, 1.0, n_waves)
        self.lo = self.hi = None

    def raw(self, x, y):
        arg = np.multiply.outer(x, self.k[:, 0]) + np.multiply.outer(y, self.k[:, 1]) + self.phase
        return (np.sin(arg) * self.amp).sum(axis=-1)

    def calibrate(self, xlim, ylim, n=120):
        gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
        v = self.raw(gx, gy)
        self.lo, self.hi = float(v.min()), float(v.max())
        return self

    def __call__(self, x, y, ptp: float):
        """Field value spanning roughly ``[-ptp/2, +ptp/2]`` over the calibrated area."""
        v = self.raw(x, y)
        return ptp * ((v - self.lo) / (self.hi - self.lo) - 0.5)


# -------------------------------------------------------------- mesh builder
def heightfield_solid(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    cell: float,
    ztop: Callable[[np.ndarray, np.ndarray], np.ndarray],
    zbot: Callable[[np.ndarray, np.ndarray], np.ndarray] | float = 0.0,
    jitter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Closed solid between two height fields, with optional vertex jitter."""
    nx = max(1, int(round((x1 - x0) / cell)))
    ny = max(1, int(round((y1 - y0) / cell)))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)  # (ny+1, nx+1)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        dx = (x1 - x0) / nx * jitter
        dy = (y1 - y0) / ny * jitter
        X = X.copy()
        Y = Y.copy()
        X[1:-1, 1:-1] += rng.uniform(-dx, dx, X[1:-1, 1:-1].shape)
        Y[1:-1, 1:-1] += rng.uniform(-dy, dy, Y[1:-1, 1:-1].shape)
    zt = np.asarray(ztop(X, Y), dtype=np.float64)
    zb = np.full_like(zt, zbot) if np.isscalar(zbot) else np.asarray(zbot(X, Y), dtype=np.float64)
    nv = X.size
    verts = np.concatenate(
        [np.stack([X.ravel(), Y.ravel(), zt.ravel()], axis=1),
         np.stack([X.ravel(), Y.ravel(), zb.ravel()], axis=1)]
    )

    def vid(i, j, bottom=False):
        return j * (nx + 1) + i + (nv if bottom else 0)

    I, J = np.meshgrid(np.arange(nx), np.arange(ny))
    I, J = I.ravel(), J.ravel()
    flip = ((I + J) % 2).astype(bool)  # alternate the diagonal
    a, b, c, d = vid(I, J), vid(I + 1, J), vid(I + 1, J + 1), vid(I, J + 1)
    # CCW seen from +z: (a,b,c),(a,c,d)  or  (a,b,d),(b,c,d)
    t1 = np.where(flip[:, None], np.stack([a, b, d], 1), np.stack([a, b, c], 1))
    t2 = np.where(flip[:, None], np.stack([b, c, d], 1), np.stack([a, c, d], 1))
    top = np.concatenate([t1, t2])
    bot = top[:, ::-1] + nv
    faces = [top, bot]

    def wall(ids_top, ids_bot, outward):
        # quad strip along a border; orient so the normal faces `outward`
        b0, b1 = ids_bot[:-1], ids_bot[1:]
        t0, t1_ = ids_top[:-1], ids_top[1:]
        tri_a = np.stack([b0, b1, t1_], 1)
        tri_b = np.stack([b0, t1_, t0], 1)
        tri = np.concatenate([tri_a, tri_b])
        p = verts[tri]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        ok = (n @ np.asarray(outward, dtype=float)) >= 0
        # all triangles of a straight wall share the same orientation
        if ok.sum() < len(ok) / 2:
            tri = tri[:, ::-1]
        return tri

    ii = np.arange(nx + 1)
    jj = np.arange(ny + 1)
    faces.append(wall(vid(ii, 0), vid(ii, 0, True), (0, -1, 0)))
    faces.append(wall(vid(ii, ny), vid(ii, ny, True), (0, 1, 0)))
    faces.append(wall(vid(0, jj), vid(0, jj, True), (-1, 0, 0)))
    faces.append(wall(vid(nx, jj), vid(nx, jj, True), (1, 0, 0)))
    return Mesh(verts.astype(np.float64), np.concatenate(faces).astype(np.int64))


# --------------------------------------------------------------------- scenes
@dataclass
class Scene:
    name: str
    mesh: Mesh
    targets: dict = field(default_factory=dict)  # name -> face mask
    protected: dict = field(default_factory=dict)  # name -> (vertex mask, tol_mm)
    info: dict = field(default_factory=dict)
    expect_levels: dict = field(default_factory=dict)  # target name -> expected plateau z


def _face_xy(mesh: Mesh):
    c = mesh.face_centroids()
    return c[:, 0], c[:, 1]


def _top_faces(mesh: Mesh, nz_min=0.5):
    n, _ = mesh.face_normals_areas()
    return n[:, 2] > nz_min


def _bottom_faces(mesh: Mesh, nz_max=-0.5):
    n, _ = mesh.face_normals_areas()
    return n[:, 2] < nz_max


def wavy_slab(ptp=1.2, base=10.0, cell=0.5, seed=1, size=(100.0, 80.0), jitter=0.2) -> Scene:
    wf = WaveField(seed).calibrate((0, size[0]), (0, size[1]))
    m = heightfield_solid(0, size[0], 0, size[1], cell,
                          lambda x, y: base + wf(x, y, ptp), 0.0, jitter, seed)
    sc = Scene("wavy_slab", m, info={"ptp": ptp, "base": base})
    sc.targets["top"] = _top_faces(m)
    return sc


def tilted_flat(angle_deg=0.3, base=10.0, cell=0.5, size=(100.0, 80.0), jitter=0.2) -> Scene:
    t = np.tan(np.radians(angle_deg))
    m = heightfield_solid(0, size[0], 0, size[1], cell,
                          lambda x, y: base + (x - size[0] / 2) * t, 0.0, jitter, 3)
    sc = Scene("tilted_flat", m, info={"angle": angle_deg})
    sc.targets["top"] = _top_faces(m)
    return sc


def ramp(angle_deg=2.0, base=6.0, cell=0.5, size=(100.0, 60.0), jitter=0.2) -> Scene:
    t = np.tan(np.radians(angle_deg))
    m = heightfield_solid(0, size[0], 0, size[1], cell,
                          lambda x, y: base + x * t, 0.0, jitter, 4)
    sc = Scene("ramp", m, info={"angle": angle_deg})
    # whole top surface must stay put
    sc.protected["top"] = (np.ones(m.n_verts, bool) & (m.verts[:, 2] > 0.5), 0.02)
    return sc


def dome(base_radius=35.0, height=10.0, z0=5.0, cell=0.5, jitter=0.2) -> Scene:
    rs = (base_radius ** 2 + height ** 2) / (2 * height)
    cx = cy = base_radius + 5
    size = 2 * cx

    def zt(x, y):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        cap = np.sqrt(np.maximum(rs * rs - r2, 0)) - (rs - height)
        return z0 + np.where(r2 < base_radius ** 2, cap, 0.0)

    m = heightfield_solid(0, size, 0, size, cell, zt, 0.0, jitter, 5)
    sc = Scene("dome", m, info={"rs": rs})
    r = np.hypot(m.verts[:, 0] - cx, m.verts[:, 1] - cy)
    # The dome itself must be preserved (only the exact-flat apex disc may move
    # by what the layer grid forces: well under one layer).
    sc.protected["dome"] = ((r < base_radius - 2.0) & (m.verts[:, 2] > 0.5), 0.12)
    return sc


def panel_with_dome(ptp=1.2, panel=10.0, dome_h=8.0, dome_r=18.0, steep=True, cell=0.5,
                    seed=2, skirt=0.0, skirt_w=6.0, jitter=0.2, size=(110.0, 90.0)) -> Scene:
    """Noisy panel with a dome in the middle (the Batwing situation).

    ``steep=True`` -> the dome rises from the panel through a near-vertical wall.
    ``skirt``      -> amplitude (mm) of a gentle skirt that climbs towards the dome.
    """
    cx, cy = size[0] / 2, size[1] / 2
    wf = WaveField(seed).calibrate((0, size[0]), (0, size[1]))

    def zt(x, y):
        r = np.hypot(x - cx, y - cy)
        z = panel + wf(x, y, ptp)
        if skirt:
            z = z + skirt * np.exp(-np.maximum(r - dome_r, 0) ** 2 / (2 * skirt_w ** 2)) * (r > dome_r)
        inside = r < dome_r
        cap = dome_h * np.sqrt(np.maximum(1 - (r / dome_r) ** 2, 0))
        if steep:
            z = np.where(inside, panel + 1.0 + cap, z)
        else:
            z = np.where(inside, np.maximum(panel + skirt + cap, z), z)
        return z

    m = heightfield_solid(0, size[0], 0, size[1], cell, zt, 0.0, jitter, seed)
    sc = Scene("panel_with_dome" + ("_skirt" if skirt else ""), m,
               info={"ptp": ptp, "panel": panel, "dome_r": dome_r})
    fx, fy = _face_xy(m)
    fr = np.hypot(fx - cx, fy - cy)
    up = _top_faces(m)
    sc.targets["panel"] = up & (fr > dome_r + (3.0 if not skirt else 3.0 + 2 * skirt_w))
    r = np.hypot(m.verts[:, 0] - cx, m.verts[:, 1] - cy)
    sc.protected["dome"] = ((r < dome_r - 3.0) & (m.verts[:, 2] > 0.5), 0.12)
    return sc


def two_level_panels(ptp=1.0, lo=10.0, hi=10.6, cell=0.5, seed=7, size=(100.0, 60.0)) -> Scene:
    wf = WaveField(seed).calibrate((0, size[0]), (0, size[1]))
    xm = size[0] / 2

    def zt(x, y):
        return np.where(x < xm, lo, hi) + wf(x, y, ptp)

    m = heightfield_solid(0, size[0], 0, size[1], cell, zt, 0.0, 0.2, seed)
    sc = Scene("two_level_panels", m, info={"lo": lo, "hi": hi})
    fx, fy = _face_xy(m)
    up = _top_faces(m)
    sc.targets["left"] = up & (fx < xm - 2)
    sc.targets["right"] = up & (fx > xm + 2)
    return sc


def ceiling_pocket(ptp=1.0, ceil=4.0, cell=0.5, seed=9, size=(90.0, 70.0)) -> Scene:
    """A flat top over a pocket whose noisy ceiling is a down-facing surface."""
    wf = WaveField(seed).calibrate((0, size[0]), (0, size[1]))
    p0, p1 = (20.0, 15.0), (70.0, 55.0)

    def zb(x, y):
        inside = (x > p0[0]) & (x < p1[0]) & (y > p0[1]) & (y < p1[1])
        # pocket carved *up* from the bottom: ceiling at `ceil` (+noise) inside
        return np.where(inside, ceil + wf(x, y, ptp), 0.0)

    m = heightfield_solid(0, size[0], 0, size[1], cell, lambda x, y: 12.0 + 0 * x, zb, 0.2, seed)
    sc = Scene("ceiling_pocket", m, info={"ceil": ceil})
    fx, fy = _face_xy(m)
    inside = (fx > p0[0] + 3) & (fx < p1[0] - 3) & (fy > p0[1] + 3) & (fy < p1[1] - 3)
    sc.targets["ceiling"] = _bottom_faces(m) & inside
    return sc


def terraces_with_ramp(ptp=0.8, lo=10.0, hi=12.0, ramp_deg=3.0, cell=0.5, seed=11,
                       size=(130.0, 60.0)) -> Scene:
    """Two noisy plateaus joined by a gentle ramp that is a real feature."""
    wf = WaveField(seed).calibrate((0, size[0]), (0, size[1]))
    x_a = 35.0
    x_b = x_a + (hi - lo) / np.tan(np.radians(ramp_deg))

    def zt(x, y):
        base = np.where(x < x_a, lo, np.where(x > x_b, hi, lo + (x - x_a) * np.tan(np.radians(ramp_deg))))
        mix = np.where((x < x_a) | (x > x_b), 1.0, 0.0)  # noise only on the plateaus
        return base + wf(x, y, ptp) * mix

    m = heightfield_solid(0, size[0], 0, size[1], cell, zt, 0.0, 0.2, seed)
    sc = Scene("terraces_with_ramp", m, info={"x_a": x_a, "x_b": x_b})
    fx, fy = _face_xy(m)
    up = _top_faces(m)
    sc.targets["low"] = up & (fx < x_a - 3)
    sc.targets["high"] = up & (fx > x_b + 3)
    # the middle of the ramp is a real, evenly sloped feature
    mid = (m.verts[:, 0] > x_a + 12) & (m.verts[:, 0] < x_b - 12) & (m.verts[:, 2] > 0.5)
    sc.protected["ramp_mid"] = (mid, 0.15)
    return sc


def all_synthetic(small: bool = False) -> list[Scene]:
    cell = 1.0 if small else 0.5
    return [
        wavy_slab(cell=cell),
        tilted_flat(cell=cell),
        ramp(cell=cell),
        dome(cell=cell),
        panel_with_dome(cell=cell),
        panel_with_dome(cell=cell, skirt=1.0),
        panel_with_dome(cell=cell, steep=False, skirt=0.8),
        two_level_panels(cell=cell),
        ceiling_pocket(cell=cell),
        terraces_with_ramp(cell=cell),
    ]


# ------------------------------------------------- the real Batwing, re-noised
def batwing_before(after: Mesh, ptp=1.2, seed=5, panel_z=9.8, wavelengths=(25.0, 80.0)) -> Scene:
    """Recreate a "Before" model from the already-flat Batwing "After" mesh.

    The panels (faces exactly flat at ``panel_z``) get a smooth random height
    field of the given peak-to-peak range, mimicking the 9.4-10.6 mm
    scatter in the original generator output.
    """
    n, area = after.face_normals_areas()
    zc = after.face_centroids()[:, 2]
    flat = (n[:, 2] > 0.99999) & (np.abs(zc - panel_z) < 1e-3)
    ncomp, lab = after.face_components(flat)
    comp_area = np.bincount(lab[flat], weights=area[flat], minlength=ncomp)
    big = np.flatnonzero(comp_area > 50.0)  # ignore specks
    panel_faces = flat & np.isin(lab, big)
    pv = np.zeros(after.n_verts, bool)
    pv[after.faces[panel_faces].ravel()] = True

    xy = after.verts[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    wf = WaveField(seed, wavelengths=wavelengths, n_waves=12).calibrate((lo[0], hi[0]), (lo[1], hi[1]))
    z = after.verts[:, 2].copy()
    # only vertices strictly inside the panel get the field; vertices that also
    # belong to walls / dome move with it (they are shared), which stretches the
    # walls exactly like the original generator output did.
    z[pv] = panel_z + wf(xy[pv, 0], xy[pv, 1], ptp)
    before = Mesh(np.column_stack([after.verts[:, :2], z]), after.faces)

    sc = Scene("batwing_before", before, info={"ptp": ptp, "panel_z": panel_z, "seed": seed})
    sc.targets["panel"] = panel_faces
    # Dome / pumpkin: vertices well above the panel that are not shared with it.
    nb = np.zeros(after.n_verts, bool)
    nb[after.faces[~panel_faces].ravel()] = True
    near_panel = np.zeros(after.n_verts, bool)
    from scipy import sparse
    g = after.vertex_graph(weighted=False)
    near_panel = (g @ pv.astype(float)) > 0
    pumpkin = nb & (after.verts[:, 2] > panel_z + 1.5) & ~near_panel & ~pv
    sc.protected["pumpkin"] = (pumpkin, 0.12)
    return sc
