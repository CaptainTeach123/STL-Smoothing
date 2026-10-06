"""Tests for STL I/O, welded meshes and the layer grid."""

import struct

import numpy as np
import pytest

import scenes
from stl_smoothing import stlio
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh
from stl_smoothing.slicing import contour_length


def _cube_tris():
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    return v[f]


# ----------------------------------------------------------------------- I/O
def test_binary_roundtrip(tmp_path):
    tris = _cube_tris()
    p = tmp_path / "cube.stl"
    stlio.write_stl(p, tris, header="hello")
    d = stlio.read_stl(p)
    assert d.source_format == "binary"
    assert d.header.startswith(b"hello")
    np.testing.assert_array_equal(d.tris, tris)


def test_ascii_roundtrip(tmp_path):
    tris = _cube_tris()
    p = tmp_path / "cube_ascii.stl"
    stlio.write_stl(p, tris, binary=False, name="cube")
    d = stlio.read_stl(p)
    assert d.source_format == "ascii"
    assert d.name == "cube"
    np.testing.assert_allclose(d.tris, tris)


def test_binary_header_starting_with_solid_is_still_binary(tmp_path):
    tris = _cube_tris()
    p = tmp_path / "tricky.stl"
    stlio.write_stl(p, tris, header=b"solid exported by something")
    assert stlio.read_stl(p).source_format == "binary"


def test_wrong_triangle_count_is_tolerated(tmp_path):
    tris = _cube_tris()
    p = tmp_path / "badcount.stl"
    stlio.write_stl(p, tris)
    raw = bytearray(p.read_bytes())
    struct.pack_into("<I", raw, 80, 999999)  # lie about the count
    p.write_bytes(bytes(raw))
    d = stlio.read_stl(p)
    assert len(d.tris) == len(tris)


def test_attributes_preserved(tmp_path):
    tris = _cube_tris()
    attrs = np.arange(len(tris), dtype=np.uint16)
    p = tmp_path / "attr.stl"
    stlio.write_stl(p, tris, attrs=attrs)
    np.testing.assert_array_equal(stlio.read_stl(p).attrs, attrs)


def test_garbage_rejected(tmp_path):
    p = tmp_path / "junk.stl"
    p.write_bytes(b"this is not an stl file at all")
    with pytest.raises(stlio.StlError):
        stlio.read_stl(p)


def test_empty_ascii_solid(tmp_path):
    p = tmp_path / "empty.stl"
    p.write_text("solid nothing\nendsolid nothing\n")
    d = stlio.read_stl(p)
    assert len(d.tris) == 0


# ---------------------------------------------------------------------- mesh
def test_weld_cube():
    m = Mesh.from_triangles(_cube_tris())
    assert m.n_verts == 8 and m.n_faces == 12
    assert m.edge_manifold_stats()["closed"]
    np.testing.assert_allclose(m.to_triangles(), _cube_tris())


def test_weld_minus_zero():
    t = _cube_tris().copy()
    t[0, 0, 0] = -0.0
    assert Mesh.from_triangles(t).n_verts == 8


def test_weld_tolerance():
    t = _cube_tris().copy()
    t[3, 1] += 3e-7  # a hair off, would not weld exactly
    assert Mesh.from_triangles(t).n_verts > 8
    assert Mesh.from_triangles(t, tol=1e-4).n_verts == 8


def test_face_components_and_adjacency():
    m = Mesh.from_triangles(_cube_tris())
    n, lab = m.face_components(np.ones(m.n_faces, bool))
    assert n == 1 and (lab == 0).all()
    top = m.face_normals_areas()[0][:, 2] > 0.5
    n, lab = m.face_components(top)
    assert n == 1 and (lab[~top] == -1).all()


def test_flip_detection():
    m = Mesh.from_triangles(_cube_tris())
    v = m.verts.copy()
    v[v[:, 2] > 0.5, 2] = -0.5  # push the top through the bottom
    assert m.flipped_faces(v).any()
    assert not m.flipped_faces(m.verts.copy()).any()


# -------------------------------------------------------------- layer grid
def test_layer_grid_snap_and_ends():
    g = LayerGrid(0.2)
    np.testing.assert_allclose(g.snap([9.71, 9.79, 9.89, 9.91]), [9.8, 9.8, 9.8, 10.0])
    # a surface sampled at mid layer: 9.69 ends layer 48 (top 9.6), 9.71 ends layer 49 (top 9.8)
    np.testing.assert_allclose(g.layer_end([9.69, 9.71, 9.8, 9.89]), [9.6, 9.8, 9.8, 9.8])
    assert g.layer_number(0.05) == 0 and g.layer_number(0.15) == 1


def test_layer_grid_first_layer():
    g = LayerGrid(0.2, first_layer=0.3)
    np.testing.assert_allclose(g.boundary([0, 1, 2, 3]), [0.0, 0.3, 0.5, 0.7])
    np.testing.assert_allclose(g.snap([0.31, 0.62, 0.6]), [0.3, 0.7, 0.5], atol=1e-12)
    assert g.layer_number(0.2) == 1  # above the mid plane of the 0.3 first layer


def test_layer_grid_rejects_bad_values():
    with pytest.raises(ValueError):
        LayerGrid(0)
    with pytest.raises(ValueError):
        LayerGrid(0.2, first_layer=-1)


def test_snapped_height_is_robust_to_slicer_sampling():
    g = LayerGrid(0.2)
    z = g.snap(np.array([7.1, 9.7, 3.333]))
    # distance to the nearest mid-layer sampling plane is half a layer
    mids = np.arange(0, 20, 0.2) + 0.1
    for zz in z:
        assert np.min(np.abs(mids - zz)) == pytest.approx(0.1, abs=1e-9)


# ------------------------------------------------------------- scene sanity
@pytest.mark.parametrize("scene", scenes.all_synthetic(small=True), ids=lambda s: s.name)
def test_scenes_are_closed_and_outward(scene):
    m = scene.mesh
    assert m.edge_manifold_stats()["closed"]
    t = m.verts[m.faces]
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    assert vol > 0


def test_contour_length_counts_layer_crossings():
    # a 10 x 10 square tilted so it climbs 1 mm along x: 5 sample planes cross it
    m = scenes.heightfield_solid(0, 10, 0, 10, 0.5, lambda x, y: 5.0 + 0.1 * x, 0.0)
    top = m.face_normals_areas()[0][:, 2] > 0.5
    L = contour_length(m, LayerGrid(0.2), top)
    assert L == pytest.approx(5 * 10.0, rel=0.05)
