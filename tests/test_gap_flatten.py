"""Tests that close the gaps found by hand-made mutation testing of the flattening pipeline.

Every test here passes on the shipped code and fails when the behaviour named in its
docstring is broken.  They need no external model (nothing is skipped on a bare checkout).
"""

import numpy as np
import pytest

import metrics
import scenes
from stl_smoothing import flatten as flatten_module
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh

G = LayerGrid(0.2)


def top_vertices(mesh, mask=None):
    up = scenes._top_faces(mesh) if mask is None else mask
    return np.unique(mesh.faces[up])


def sin_slab(amp, base=10.0, period=40.0, size=(100.0, 80.0)):
    """A smooth panel whose height spans exactly base +- amp (extremes sit on grid vertices)."""
    return scenes.heightfield_solid(
        0, size[0], 0, size[1], 1.0,
        lambda x, y: base + amp * np.sin(2 * np.pi * x / period) * np.sin(2 * np.pi * y / period), 0.0)


def noisy_panel_with_block(block_top, seed=31, size=(100.0, 80.0)):
    """Noisy panel with one intentional, exactly flat raised/lowered block."""
    wf = scenes.WaveField(seed).calibrate((0, size[0]), (0, size[1]))
    b = (15.0, 25.0, 15.0, 30.0)

    def zt(x, y):
        z = 10.0 + wf(x, y, 1.0)
        return np.where((x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3]), block_top, z)

    return scenes.heightfield_solid(0, size[0], 0, size[1], 0.5, zt, 0.0, 0.0, seed)


# --------------------------------------------------- which layer boundary is chosen
@pytest.mark.parametrize("base, expected", [(10.07, 10.0), (10.13, 10.2)])
def test_plateau_goes_to_the_boundary_nearest_its_median(base, expected):
    """README: 'The plateau goes to the boundary nearest its median height.'

    A level that is merely 'within one layer of the original' (what the older tests accept)
    is not enough: snapping up or down instead of to the nearest boundary must fail."""
    sc = scenes.wavy_slab(ptp=0.6, base=base, cell=1.0)
    med = np.median(sc.mesh.verts[top_vertices(sc.mesh, sc.targets["top"]), 2])
    assert abs(med - base) < 0.05  # the scene is what the test thinks it is
    res = flatten(sc.mesh, G)
    z = res.verts[top_vertices(sc.mesh, sc.targets["top"]), 2]
    assert np.allclose(z, expected, atol=1e-5), f"median {med:.3f} went to {z[0]:.3f}, not {expected}"
    assert res.plateaus[0].level_after == pytest.approx(expected, abs=1e-5)


def test_plateau_level_follows_the_median_not_the_mean():
    """Heights skewed towards the bottom: median 9.855, mean 9.926 -> 9.8 (the mean would give 10.0)."""
    size = (100.0, 80.0)
    wf = scenes.WaveField(1).calibrate((0, size[0]), (0, size[1]))
    m = scenes.heightfield_solid(0, size[0], 0, size[1], 1.0,
                                 lambda x, y: 9.7 + 1.2 * (wf(x, y, 1.0) + 0.5) ** 3, 0.0, 0.2, 1)
    n, a = m.face_normals_areas()
    up = n[:, 2] > 0.5
    zc = m.face_centroids()[:, 2]
    mean = float((zc[up] * a[up]).sum() / a[up].sum())
    order = np.argsort(zc[up])
    cw = np.cumsum(a[up][order])
    med = float(zc[up][order][np.searchsorted(cw, cw[-1] / 2)])
    assert G.snap(med) != G.snap(mean)  # scene sanity: the two statistics disagree
    res = flatten(m, G)
    assert res.plateaus[0].level_after == pytest.approx(float(G.snap(med)), abs=1e-5)


# ---------------------------------------------------------- anchors / bed / spikes
@pytest.mark.parametrize("block_top", [9.55, 10.93])
def test_exactly_flat_block_off_the_sampling_planes_never_moves(block_top):
    """README: 'Faces that are already exactly flat are anchors and never move.'  Bit-identical,
    rim vertices included (a noisy neighbour that is pulled to the plateau must not drag them)."""
    m = noisy_panel_with_block(block_top)
    n, _ = m.face_normals_areas()
    tz = m.verts[m.faces][:, :, 2]
    block = (np.abs(n[:, 2]) > 0.999) & (np.ptp(tz, axis=1) < 1e-9) & (np.abs(tz[:, 0] - block_top) < 1e-9)
    assert block.sum() > 500
    vs = np.unique(m.faces[block])
    res = flatten(m, G)
    assert res.n_moved > 0  # the panel itself was flattened
    assert (res.verts[vs, 2] == m.verts[vs, 2]).all(), f"{int((res.verts[vs, 2] != m.verts[vs, 2]).sum())} block vertices moved"


def test_model_stays_on_the_bed():
    """README: 'bed contact never moves': the wobbly underside of a part must not lift off the bed."""
    size = (100.0, 80.0)
    wf = scenes.WaveField(5).calibrate((0, size[0]), (0, size[1]))
    m = scenes.heightfield_solid(0, size[0], 0, size[1], 1.0, lambda x, y: 10.0 + wf(x, y, 1.0),
                                 lambda x, y: 0.3 * (wf(x, y, 1.0) + 0.5), 0.2, 5)
    z0 = m.verts[:, 2]
    bed = z0 <= z0.min() + 0.1
    assert bed.sum() > 500
    res = flatten(m, G)
    assert res.n_moved > 0
    assert (res.verts[bed, 2] == z0[bed]).all()
    assert res.verts[:, 2].min() == z0.min()


def test_single_vertex_spikes_are_absorbed_into_the_plateau():
    """Hot-pixel spikes inside a flat panel are pulled onto it (spike_max), they do not stand out
    as stray layer edges around a lone vertex."""
    size = (100.0, 80.0)
    wf = scenes.WaveField(3).calibrate((0, size[0]), (0, size[1]))
    m = scenes.heightfield_solid(0, size[0], 0, size[1], 0.5, lambda x, y: 10.0 + wf(x, y, 1.0), 0.0, 0.2, 3)
    top = scenes._top_faces(m)
    v = m.verts.copy()
    cand = np.flatnonzero((v[:, 2] > 5) & (v[:, 0] > 5) & (v[:, 0] < 95) & (v[:, 1] > 5) & (v[:, 1] < 75))
    spikes = np.random.default_rng(1).choice(cand, 10, replace=False)
    v[spikes, 2] += 0.35
    spiked = Mesh(v, m.faces)
    res = flatten(spiked, G)
    # a lone spike belongs to steep faces only, so judge the flat faces and the spike vertices
    assert metrics.contour_length(spiked, G, top, res.verts) < 0.5
    plateau = np.median(res.verts[top_vertices(m), 2])
    assert np.allclose(res.verts[spikes, 2], plateau, atol=1e-5)


# ------------------------------------------------------ shape versus wobble guards
def test_open_surface_without_walls_is_flattened():
    """A bare top sheet (no bottom, no walls, only open edges) is a common generator output.  Its
    open boundary counts as a wall; if it did not, the whole sheet would be taken for a ramp."""
    sc = scenes.wavy_slab(ptp=1.0, cell=1.0)
    n, _ = sc.mesh.face_normals_areas()
    sheet = Mesh(sc.mesh.verts, sc.mesh.faces[n[:, 2] > 0.5])
    assert sheet.edge_manifold_stats()["boundary_edges"] > 0
    res = flatten(sheet, G)
    z = res.verts[np.unique(sheet.faces), 2]
    assert np.ptp(z) < 1e-5 and res.plateaus and res.plateaus[0].layers_before > 1
    assert metrics.contour_length(sheet, G, np.ones(sheet.n_faces, bool), res.verts) < 0.5


def test_max_slope_option_is_honoured():
    """A 0.3 degree tilt is 'flat' at the default 5 degrees but not at --max-slope 0.1."""
    sc = scenes.tilted_flat(angle_deg=0.3, cell=1.0)
    assert flatten(sc.mesh, G).n_moved > 0
    assert flatten(sc.mesh, G, max_slope_deg=0.1).n_moved == 0


def test_min_area_option_is_honoured():
    sc = scenes.wavy_slab(cell=1.0)  # one 8000 mm2 panel
    assert flatten(sc.mesh, G, min_area=40.0).n_moved > 0
    assert flatten(sc.mesh, G, min_area=9000.0).n_moved == 0


@pytest.mark.parametrize("angle", [0.6, 0.8, 0.9])
def test_exactly_planar_shallow_ramp_is_kept(angle):
    """A clean ramp that rises less than flat_top_range (1 mm) has no height spread to give it
    away: only the exact-planarity test keeps it from being flattened onto one layer."""
    sc = scenes.ramp(angle_deg=angle, size=(60.0, 60.0), cell=1.0)
    assert 60.0 * np.tan(np.radians(angle)) < 1.0
    res = flatten(sc.mesh, G)
    assert res.n_moved == 0


def test_evenly_spread_surface_is_a_shape_even_without_the_planar_test():
    """Second line of defence: with the planar test off, a ramp whose heights are spread evenly
    over more than flat_top_range is still not a plateau (README step 3, 'peakiness')."""
    sc = scenes.ramp(angle_deg=1.1, cell=1.0)  # 1.9 mm rise, bounded by walls
    assert flatten(sc.mesh, G, planar_ramp_deg=None).n_moved == 0


# ------------------------------------------------------ sampling-plane margins
@pytest.mark.parametrize("amp, moves", [(0.06, False), (0.08, False), (0.09, True), (0.095, True), (0.11, True)])
def test_plane_margin_decides_between_left_alone_and_levelled(amp, moves):
    """A surface inside one layer band is left alone only while it stays at least plane_margin
    (5 % of the layer = 0.01 mm) clear of both sampling planes.  amp=0.09 is exactly one margin
    away (inclusive: still levelled); amp=0.08 is clear."""
    m = sin_slab(amp)
    res = flatten(m, G)
    assert (res.n_moved > 0) == moves
    if not moves:
        assert res.already_flat and res.already_flat[0][0] == "top"


@pytest.mark.parametrize("z, nudged", [(10.5, True), (10.51, True), (10.49, True), (10.52, False), (10.48, False), (9.8, False)])
def test_exactly_flat_top_near_a_sampling_plane_is_nudged_only_within_the_margin(z, nudged):
    """The planes of 0.2 mm layers are at 10.5, 10.3 ...: a flat top within 0.01 mm (inclusive) of one
    may print on either layer and is moved to the nearest boundary; 0.02 mm away is left bit-identical."""
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: z + 0 * x, 0.0)
    res = flatten(m, G)
    top = m.verts[:, 2] > 5
    if nudged:
        assert res.n_moved > 0
        assert np.allclose(res.verts[top, 2], G.snap(z), atol=1e-5)
    else:
        assert res.n_moved == 0


# ------------------------------------------------------ report fidelity
def test_plateau_report_describes_what_was_done():
    sc = scenes.wavy_slab(cell=1.0, ptp=1.2)
    m = sc.mesh
    tv = top_vertices(m, sc.targets["top"])
    res = flatten(m, G)
    p = res.plateaus[0]
    z_after = res.verts[tv, 2]
    assert p.level_after == pytest.approx(float(np.median(z_after)), abs=1e-5)
    assert p.z_low < p.level_before < p.z_high
    assert p.z_low == pytest.approx(np.percentile(m.verts[tv, 2], 0.5), abs=0.05)
    assert p.z_high == pytest.approx(np.percentile(m.verts[tv, 2], 99.5), abs=0.05)
    assert abs(p.layers_before - metrics.layers_spanned(m, G, sc.targets["top"])) <= 1
    assert p.area == pytest.approx(float(m.face_normals_areas()[1][sc.targets["top"]].sum()), rel=0.02)
    assert p.regions == 1 and p.layers_after == 1
    # bookkeeping of the result agrees with the geometry
    changed = (res.verts != m.verts).any(axis=1)
    assert res.n_moved == int(changed.sum())
    assert res.max_dz == pytest.approx(float(np.abs(res.verts[:, 2] - m.verts[:, 2]).max()), abs=1e-9)
    assert res.z_offset == 0.0 and res.flipped_faces == int(m.flipped_faces(res.verts).sum()) == 0


def test_skipped_and_already_flat_are_reported():
    sc = scenes.wavy_slab(cell=1.0, ptp=1.2)
    res = flatten(sc.mesh, G, max_range=0.5)
    assert res.n_moved == 0 and not res.plateaus
    assert any("max_range" in s and s.startswith("top surface") for s in res.skipped)
    calm = flatten(scenes.wavy_slab(cell=1.0, base=10.0, ptp=0.12).mesh, G)
    assert calm.n_moved == 0 and calm.already_flat and not calm.skipped


# ------------------------------------------------------ pipeline wiring
def test_pipeline_runs_unfold_and_flip_repair(monkeypatch):
    """The wall-protection stages must actually run on the moved geometry.  (None of the synthetic
    scenes folds a wall, so their effect cannot be seen in the output of these scenes.)"""
    calls = []

    def spy(name, fn):
        def wrapper(*a, **k):
            calls.append(name)
            return fn(*a, **k)
        return wrapper

    monkeypatch.setattr(flatten_module, "unfold", spy("unfold", flatten_module.unfold))
    monkeypatch.setattr(flatten_module, "repair_flips", spy("repair_flips", flatten_module.repair_flips))
    monkeypatch.setattr(flatten_module, "feather", spy("feather", flatten_module.feather))
    flatten(scenes.panel_with_dome(cell=1.0).mesh, G)
    assert calls == ["feather", "unfold", "repair_flips"]

# ------------------------------------------------------ grids other than the default
@pytest.mark.parametrize("grid", [LayerGrid(0.2, 0.3), LayerGrid(0.28), LayerGrid(0.12, 0.2)], ids=str)
def test_levels_land_on_the_boundaries_of_the_users_own_grid(grid):
    """test_other_layer_heights only checks 'one layer, no edges' - a plateau placed exactly ON a
    sampling plane (the worst place, the one a slicer may round either way) also passes that."""
    sc = scenes.wavy_slab(cell=1.0)
    res = flatten(sc.mesh, grid)
    z = res.verts[top_vertices(sc.mesh, sc.targets["top"]), 2]
    assert np.ptp(z) < 1e-5 and abs(z[0] - float(grid.snap(z[0]))) < 1e-5, f"plateau at {z[0]:.4f}"
    # an exactly flat top that sits on one of this grid's sampling planes is nudged to one of its boundaries
    plane = float(grid.boundary(52)) - 0.5 * grid.layer_height
    flat = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: plane + 0 * x, 0.0)
    r2 = flatten(flat, grid)
    zt = r2.verts[r2.verts[:, 2] > 5, 2]
    assert r2.n_moved > 0 and abs(zt[0] - float(grid.snap(zt[0]))) < 1e-5 and abs(zt[0] - plane) == pytest.approx(0.5 * grid.layer_height, abs=1e-5)


def test_default_min_area_leaves_a_small_wobbly_patch_alone():
    side = 6.0  # 36 mm2 < 40 mm2
    wf = scenes.WaveField(3, wavelengths=(30.0, 60.0)).calibrate((0, side), (0, side))
    m = scenes.heightfield_solid(0, side, 0, side, 0.5, lambda x, y: 10.0 + wf(x, y, 0.5), 0.0, 0.0)
    assert flatten(m, G).n_moved == 0
    assert flatten(m, G, min_area=5.0).n_moved > 0


def test_tiny_exactly_flat_pad_on_a_sampling_plane_is_not_snapped():
    """min_flat_area: only exactly flat patches of at least 5 mm2 are nudged (a 2 x 2 mm pad is left bit-identical)."""
    tiny = scenes.heightfield_solid(0, 2, 0, 2, 1.0, lambda x, y: 10.5 + 0 * x, 0.0)
    assert flatten(tiny, G).n_moved == 0
    bigger = scenes.heightfield_solid(0, 2.4, 0, 2.4, 1.2, lambda x, y: 10.5 + 0 * x, 0.0)
    assert flatten(bigger, G).n_moved > 0


def test_heavy_facet_jitter_is_still_flattened():
    """+-0.15 mm per-vertex noise on 0.5 mm cells (faces tilted by up to ~30 degrees) on top of a 1.2 mm
    wobble: the normal smoothing must reach far enough (smooth_radius) for the panel to be recognised."""
    size = (100.0, 80.0)
    wf = scenes.WaveField(1).calibrate((0, size[0]), (0, size[1]))
    rng = np.random.default_rng(7)
    m = scenes.heightfield_solid(0, size[0], 0, size[1], 0.5,
                                 lambda x, y: 10.0 + wf(x, y, 1.2) + rng.uniform(-0.15, 0.15, x.shape), 0.0, 0.2, 1)
    sc = scenes.Scene("hf", m)
    sc.targets["top"] = scenes._top_faces(m)
    res = flatten(m, G)
    t = metrics.evaluate(sc, res.verts, G)["targets"]["top"]
    assert t["layers_after"] == 1 and t["contour_after"] < 0.01 * t["contour_before"], t
