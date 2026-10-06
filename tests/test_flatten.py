"""Behaviour of the flattening algorithm on scenes with known ground truth."""

import os

import numpy as np
import pytest

import metrics
import scenes
from stl_smoothing import stlio
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh

G = LayerGrid(0.2)
CELL = 1.0  # coarse cells keep the suite fast


def run(scene, grid=G, **kw):
    res = flatten(scene.mesh, grid, **kw)
    return res, metrics.evaluate(scene, res.verts, grid)


def assert_flat(res_metrics, name=None):
    for n, t in res_metrics["targets"].items():
        if name and n != name:
            continue
        assert t["layers_after"] == 1, f"{n} still ends on {t['layers_after']} layers"
        assert t["contour_after"] < 0.5, f"{n} has {t['contour_after']:.1f} mm of layer edges left"


def assert_sane(res, m):
    assert m["flipped"] == 0
    assert m["xy_moved_max"] == 0.0
    assert res.flipped_faces == 0


# ------------------------------------------------------------ the core promise
def test_wavy_slab_ends_on_one_layer():
    sc = scenes.wavy_slab(cell=CELL)
    res, m = run(sc)
    assert_flat(m)
    assert_sane(res, m)
    # all vertices of the top share one height, on a layer boundary
    top = np.unique(sc.mesh.faces[sc.targets["top"]])
    z = res.verts[top, 2]
    assert z.max() - z.min() < 1e-5
    assert abs(z[0] - G.snap(z[0])) < 1e-5


def test_plateau_level_stays_near_the_original():
    sc = scenes.wavy_slab(cell=CELL, base=10.0, ptp=1.2)
    res, _ = run(sc)
    top = np.unique(sc.mesh.faces[sc.targets["top"]])
    assert abs(np.median(res.verts[top, 2]) - np.median(sc.mesh.verts[top, 2])) <= G.layer_height


def test_idempotent():
    sc = scenes.panel_with_dome(cell=CELL)
    r1 = flatten(sc.mesh, G)
    r2 = flatten(Mesh(r1.verts, sc.mesh.faces), G)
    assert np.abs(r2.verts - r1.verts).max() < 1e-6
    assert not r2.plateaus or all(p.kind != "smoothed" for p in r2.plateaus)


def test_only_z_changes_and_connectivity_is_kept():
    sc = scenes.panel_with_dome(cell=CELL)
    res = flatten(sc.mesh, G)
    np.testing.assert_array_equal(res.verts[:, :2], sc.mesh.verts[:, :2])
    assert res.verts.shape == sc.mesh.verts.shape


def test_ceiling_is_flattened():
    sc = scenes.ceiling_pocket(cell=CELL)
    res, m = run(sc)
    assert_flat(m)
    assert_sane(res, m)
    assert any(p.facing == "down" for p in res.plateaus)


def test_dome_is_preserved_next_to_a_flattened_panel():
    sc = scenes.panel_with_dome(cell=CELL)
    res, m = run(sc)
    assert_flat(m)
    assert m["protected"]["dome"]["ok"]


# ------------------------------------------------------- things to leave alone
@pytest.mark.parametrize("make", [scenes.ramp, scenes.dome], ids=["ramp", "dome"])
def test_intentional_shapes_are_untouched(make):
    sc = make(cell=CELL)
    res, m = run(sc)
    assert m["max_dz"] < 1e-6
    assert res.n_moved == 0


def test_surface_already_on_one_layer_is_left_alone():
    # waviness well inside one layer band, clear of the sampling planes
    sc = scenes.wavy_slab(cell=CELL, base=10.0, ptp=0.12)
    res = flatten(sc.mesh, G)
    assert res.n_moved == 0


def test_exactly_flat_plateau_on_a_boundary_is_not_touched():
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 9.8 + 0 * x, 0.0)
    res = flatten(m, G)
    assert res.n_moved == 0


def test_exactly_flat_plateau_on_a_sampling_plane_is_nudged_to_the_grid():
    # 10.5 is exactly the sampling plane of the layer 10.4-10.6: ambiguous for a slicer
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 10.5 + 0 * x, 0.0)
    res = flatten(m, G)
    top = res.verts[:, 2] > 5
    assert res.n_moved > 0
    assert np.allclose(res.verts[top, 2], [10.4, 10.6][int(abs(res.verts[top, 2][0] - 10.6) < 1e-6)], atol=1e-5)
    assert abs(res.verts[top, 2][0] - 10.5) == pytest.approx(0.1, abs=1e-5)


def test_intentional_exact_blocks_are_kept():
    sc = scenes.plate_with_exact_blocks(cell=CELL)
    res, m = run(sc)
    assert_flat(m)
    assert m["protected"]["block1"]["ok"] and m["protected"]["block2"]["ok"]


def test_distinct_levels_stay_distinct():
    sc = scenes.two_level_panels(cell=CELL)  # 1.0 mm step between two noisy panels
    res, m = run(sc)
    assert_flat(m)
    lo = np.unique(sc.mesh.faces[sc.targets["left"]])
    hi = np.unique(sc.mesh.faces[sc.targets["right"]])
    step = np.median(res.verts[hi, 2]) - np.median(res.verts[lo, 2])
    assert abs(step - 1.0) <= G.layer_height


def test_ramp_between_plateaus_survives():
    sc = scenes.terraces_with_ramp(cell=CELL)
    res, m = run(sc)
    assert_flat(m)
    assert m["protected"]["ramp_mid"]["ok"]


def test_embossed_bar_on_a_noisy_panel_is_kept():
    size = (100.0, 60.0)
    wf = scenes.WaveField(3).calibrate((0, size[0]), (0, size[1]))

    def top(x, y):
        z = 10.0 + wf(x, y, 1.2)
        bar = (x > 30) & (x < 70) & (y > 28) & (y < 32)
        return np.where(bar, z + 0.6, z)

    m = scenes.heightfield_solid(0, size[0], 0, size[1], 0.5, top, 0.0, 0.2, 3)
    res = flatten(m, G)
    c = m.face_centroids()
    bar_v = np.unique(m.faces[(c[:, 0] > 36) & (c[:, 0] < 64) & (c[:, 1] > 29) & (c[:, 1] < 31) & (m.face_normals_areas()[0][:, 2] > 0.5)])
    panel_v = np.unique(m.faces[(c[:, 0] > 5) & (c[:, 0] < 25) & (c[:, 1] > 5) & (c[:, 1] < 25) & (m.face_normals_areas()[0][:, 2] > 0.5)])
    step = np.median(res.verts[bar_v, 2]) - np.median(res.verts[panel_v, 2])
    assert abs(step - 0.6) <= 0.21, f"emboss became {step:.2f} mm"


# ------------------------------------------------------------ robustness
@pytest.mark.parametrize("jitter", [0.02, 0.04, 0.05])
def test_facet_jitter_does_not_defeat_detection(jitter):
    size = (100.0, 80.0)
    wf = scenes.WaveField(1).calibrate((0, size[0]), (0, size[1]))
    rng = np.random.default_rng(7)
    m = scenes.heightfield_solid(
        0, size[0], 0, size[1], 0.5,
        lambda x, y: 10.0 + wf(x, y, 1.2) + rng.uniform(-jitter, jitter, x.shape), 0.0, 0.2, 1,
    )
    sc = scenes.Scene("hf", m)
    sc.targets["top"] = m.face_normals_areas()[0][:, 2] > 0.5
    res, mt = run(sc)
    assert_flat(mt)


def test_quantised_heights_are_flattened():
    size = (100.0, 80.0)
    wf = scenes.WaveField(1).calibrate((0, size[0]), (0, size[1]))
    m = scenes.heightfield_solid(
        0, size[0], 0, size[1], 0.5,
        lambda x, y: np.round((10.0 + wf(x, y, 1.2)) / 0.02) * 0.02, 0.0, 0.2, 1,
    )
    sc = scenes.Scene("q", m)
    sc.targets["top"] = m.face_normals_areas()[0][:, 2] > 0.5
    res, mt = run(sc)
    assert_flat(mt)


def test_model_not_sitting_on_the_bed():
    sc = scenes.wavy_slab(cell=CELL)
    shifted = Mesh(sc.mesh.verts + np.array([0, 0, 5.13]), sc.mesh.faces)
    res = flatten(shifted, G)
    assert res.z_offset == pytest.approx(5.13)
    top = np.unique(sc.mesh.faces[sc.targets["top"]])
    rel = res.verts[top, 2] - 5.13
    assert rel.max() - rel.min() < 1e-4
    assert abs(rel[0] - G.snap(rel[0])) < 1e-4  # on the layer grid measured from the lowest point


@pytest.mark.parametrize("grid", [LayerGrid(0.1), LayerGrid(0.28), LayerGrid(0.2, 0.3)], ids=str)
def test_other_layer_heights(grid):
    sc = scenes.wavy_slab(cell=CELL)
    res, m = run(sc, grid=grid)
    assert_flat(m)
    assert_sane(res, m)


def test_open_mesh_does_not_crash():
    sc = scenes.wavy_slab(cell=CELL)
    keep = np.ones(sc.mesh.n_faces, bool)
    keep[np.random.default_rng(0).choice(sc.mesh.n_faces, 200, replace=False)] = False
    holey = Mesh(sc.mesh.verts, sc.mesh.faces[keep])
    res = flatten(holey, G)
    assert np.isfinite(res.verts).all()


def test_degenerate_inputs():
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert flatten(empty, G).verts.shape == (0, 3)
    tri = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([[0, 1, 2]]))
    assert np.isfinite(flatten(tri, G).verts).all()
    collapsed = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([[0, 0, 1], [0, 1, 2]]))
    assert np.isfinite(flatten(collapsed, G).verts).all()


def test_unknown_override_is_rejected():
    sc = scenes.wavy_slab(cell=CELL)
    with pytest.raises(TypeError):
        flatten(sc.mesh, G, not_a_parameter=1)


def test_max_range_protects_gentle_shapes():
    sc = scenes.wavy_slab(cell=CELL, ptp=1.2)
    assert flatten(sc.mesh, G).n_moved > 0
    assert flatten(sc.mesh, G, max_range=0.5).n_moved == 0  # varies more than 0.5 mm: left alone


def test_report_lists_what_was_found():
    sc = scenes.wavy_slab(cell=CELL)
    res = flatten(sc.mesh, G)
    assert len(res.plateaus) == 1
    p = res.plateaus[0]
    assert p.kind == "smoothed" and p.facing == "up"
    assert p.layers_before > 1 and p.layers_after == 1
    assert res.smoothed_faces.sum() > 0
    assert (res.face_plateau[res.smoothed_faces] == 0).all()


# --------------------------------------------------------------- real model
SAMPLE = os.environ.get("STL_SMOOTHING_SAMPLE")


@pytest.mark.skipif(not SAMPLE or not os.path.exists(SAMPLE or ""), reason="set STL_SMOOTHING_SAMPLE to the Batwing STL")
def test_batwing_before_is_fixed_and_after_is_left_alone():
    after = Mesh.from_triangles(stlio.read_stl(SAMPLE).tris)
    for seed in (5, 31):
        sc = scenes.batwing_before(after, seed=seed, jitter=0.02)
        res, m = run(sc)
        assert_flat(m)
        assert_sane(res, m)
        assert m["protected"]["pumpkin"]["ok"]
    # the already-fixed model keeps its 9.8 mm panels
    res = flatten(after, G)
    panel = scenes.batwing_before(after, seed=5).targets["panel"]
    panel_v = np.unique(after.faces[panel])
    assert np.abs(res.verts[panel_v, 2] - after.verts[panel_v, 2]).max() < 1e-6
