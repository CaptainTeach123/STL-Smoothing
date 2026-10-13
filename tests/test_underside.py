"""The model's underside: a wobbly bottom must be flattened onto the bed, not onto the second layer.

Found on a generated "batman logo": its bottom wobbled between 0 and 0.8 mm, so a slicer printed the
first layer on 0.7 % of the footprint.  Flattening it to the nearest layer boundary (0.4 mm) kept it off
the bed; the bed is the only right level for the surface a part stands on.
"""

import numpy as np

import scenes
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh
from stl_smoothing.summary import summarize

SIZE = (100.0, 60.0)


def _plate(bottom, seed=3, cell=1.5):
    """A closed plate with a wobbly top near 10 mm and the given underside height field."""
    wf_t = scenes.WaveField(seed + 1).calibrate((0, SIZE[0]), (0, SIZE[1]))
    return scenes.heightfield_solid(0, SIZE[0], 0, SIZE[1], cell, lambda x, y: 10.0 + wf_t(x, y, 0.8),
                                    bottom, 0.2, seed)


def _down_faces(mesh: Mesh, verts=None):
    """Down-facing faces, their centroid height above the lowest point, and their areas."""
    m = Mesh(mesh.verts if verts is None else verts, mesh.faces)
    n, a = m.face_normals_areas()
    zc = m.verts[m.faces][:, :, 2].mean(1) - m.verts[:, 2].min()
    return n[:, 2] < -0.9, zc, a


def _share_on_bed(mesh, verts, grid):
    down, zc, a = _down_faces(mesh, verts)
    return a[down & (zc < 0.5 * grid.first)].sum() / a[down].sum()


def test_a_wobbly_underside_ends_up_flat_on_the_bed():
    wf = scenes.WaveField(7).calibrate((0, SIZE[0]), (0, SIZE[1]))
    mesh = _plate(lambda x, y: 0.4 + wf(x, y, 0.8))  # 0.0 .. 0.8 mm, like the logo
    grid = LayerGrid(0.2)
    assert _share_on_bed(mesh, mesh.verts, grid) < 0.15, "the test model must start mostly off the bed"
    res = flatten(mesh, grid)
    on_bed = _share_on_bed(mesh, res.verts, grid)
    assert on_bed > 0.97, f"only {100 * on_bed:.1f}% of the underside prints in the first layer"
    bottoms = [p for p in res.plateaus if p.facing == "down"]
    assert bottoms and all(p.level_after == 0.0 for p in bottoms)
    tops = [p for p in res.plateaus if p.facing == "up"]
    assert tops and tops[0].layers_before > 1, "the top is still flattened as before"
    lines = []
    summarize(res, mesh, Mesh(res.verts, mesh.faces), grid, False, lines.append)
    assert any("the bed" in line for line in lines), lines


def test_an_underside_that_floats_one_layer_up_is_not_left_alone():
    """All of it above the first layer but inside the second: it 'prints on one layer', yet not on the bed."""
    wf = scenes.WaveField(11).calibrate((0, SIZE[0]), (0, SIZE[1]))
    mesh = _plate(lambda x, y: 0.30 + wf(x, y, 0.12))  # 0.24 .. 0.36 mm
    v = mesh.verts.copy()
    corner = np.flatnonzero((v[:, 0] == 0) & (v[:, 1] == 0) & (v[:, 2] < 5))[0]
    v[corner, 2] = 0.0  # one corner touches the bed and defines z = 0
    mesh = Mesh(v, mesh.faces)
    grid = LayerGrid(0.2)
    assert _share_on_bed(mesh, mesh.verts, grid) < 0.05
    res = flatten(mesh, grid)
    assert _share_on_bed(mesh, res.verts, grid) > 0.97


def test_an_underside_already_in_the_first_layer_is_left_alone():
    wf = scenes.WaveField(5).calibrate((0, SIZE[0]), (0, SIZE[1]))
    mesh = _plate(lambda x, y: 0.04 + wf(x, y, 0.08))  # 0.0 .. 0.08 mm: prints on the bed already
    res = flatten(mesh, LayerGrid(0.2))
    down, _, _ = _down_faces(mesh)
    bottom_v = np.unique(mesh.faces[down])
    assert np.array_equal(res.verts[bottom_v], mesh.verts[bottom_v])


def test_a_recess_in_the_underside_keeps_its_height():
    """A pocket 3 mm up into the bottom is a real shape: only the part that stands on the bed moves."""
    wf = scenes.WaveField(9).calibrate((0, SIZE[0]), (0, SIZE[1]))

    def bottom(x, y):
        pocket = (x > 30) & (x < 70) & (y > 15) & (y < 45)
        return np.where(pocket, 3.0, 0.4) + wf(x, y, 0.6)

    mesh = _plate(bottom)
    grid = LayerGrid(0.2)
    res = flatten(mesh, grid)
    down, zc, a = _down_faces(mesh, res.verts)
    c = Mesh(res.verts, mesh.faces).face_centroids()
    pocket = down & (c[:, 0] > 33) & (c[:, 0] < 67) & (c[:, 1] > 18) & (c[:, 1] < 42)
    assert pocket.any()
    assert np.median(zc[pocket]) > 2.4, "the pocket's ceiling must stay up"
    rim = down & ~((c[:, 0] > 27) & (c[:, 0] < 73) & (c[:, 1] > 12) & (c[:, 1] < 48))
    assert a[rim & (zc < 0.1)].sum() / a[rim].sum() > 0.95, "the part around the pocket goes onto the bed"
