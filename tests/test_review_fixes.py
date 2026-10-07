"""Regression tests for the defects an independent review found (each one failed before its fix)."""

import struct

import numpy as np
import pytest

import scenes
from stl_smoothing import stlio
from stl_smoothing.cli import auto_weld_tol, run
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh

G = LayerGrid(0.2)


def uv_sphere(radius=50.0, n_lon=32, n_lat=16, z0=None):
    """Closed UV sphere with pole fans (what Blender's default sphere looks like)."""
    z0 = radius if z0 is None else z0
    lat = np.linspace(0, np.pi, n_lat + 1)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    verts = [[0, 0, radius]]
    for a in lat[1:-1]:
        for b in lon:
            verts.append([radius * np.sin(a) * np.cos(b), radius * np.sin(a) * np.sin(b), radius * np.cos(a)])
    verts.append([0, 0, -radius])
    verts = np.array(verts) + np.array([0, 0, z0])
    faces = []
    ring = lambda i: 1 + i * n_lon  # noqa: E731
    for j in range(n_lon):
        faces.append([0, ring(0) + (j + 1) % n_lon, ring(0) + j])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = ring(i) + j, ring(i) + (j + 1) % n_lon
            c, d = ring(i + 1) + j, ring(i + 1) + (j + 1) % n_lon
            faces += [[a, b, d], [a, d, c]]
    last = len(verts) - 1
    for j in range(n_lon):
        faces.append([last, ring(n_lat - 2) + j, ring(n_lat - 2) + (j + 1) % n_lon])
    return Mesh(verts, np.array(faces, dtype=np.int64))


# ------------------------------------------------ smoothing must not cross creases
def test_gable_roof_on_a_coarse_mesh_is_not_levelled():
    # an 11 degree roof made of a handful of large triangles: averaging the normals of
    # neighbouring faces used to pool both roof planes and make the roof look flat
    m = scenes.heightfield_solid(0, 100, 0, 40, 10.0,
                                 lambda x, y: 10 + (50 - np.abs(x - 50)) * np.tan(np.radians(11)), 0.0)
    res = flatten(m, G)
    assert res.n_moved == 0


def test_default_uv_sphere_is_untouched():
    res = flatten(uv_sphere(), G)
    assert res.n_moved == 0


def test_two_triangle_ramp_is_a_ramp_even_with_a_tight_max_range():
    # a CAD wedge: the whole top is two triangles tilted 3 degrees (2.1 mm rise)
    m = scenes.heightfield_solid(0, 40, 0, 40, 40.0, lambda x, y: 10 + x * np.tan(np.radians(3)), 0.0)
    assert (m.face_normals_areas()[0][:, 2] > 0.5).sum() == 2
    assert flatten(m, G).n_moved == 0
    assert flatten(m, G, max_range=0.5).n_moved == 0


# ------------------------------------------------------------- levels of steps
@pytest.mark.parametrize("noise", [0.3, 0.6])
def test_stepped_tiers_keep_their_own_levels(noise):
    """Tiers 0.4 mm apart: no tier may be dragged towards the median of the whole stack."""
    size = (80.0, 40.0)
    wf = scenes.WaveField(4).calibrate((0, size[0]), (0, size[1]))
    riser = 0.4

    def top(x, y):
        tier = np.minimum((x // 20).astype(int), 3)
        return 10.0 + riser * tier + wf(x, y, noise)

    m = scenes.heightfield_solid(0, size[0], 0, size[1], 0.5, top, 0.0, 0.2, 4)
    res = flatten(m, G)
    c = m.face_centroids()
    up = m.face_normals_areas()[0][:, 2] > 0.5
    after = []
    for t in range(4):
        sel = up & (c[:, 0] > 20 * t + 3) & (c[:, 0] < 20 * t + 17)
        v = np.unique(m.faces[sel])
        after.append(np.median(res.verts[v, 2]))
        shift = after[-1] - np.median(m.verts[v, 2])
        assert abs(shift) <= 0.21, f"tier {t} moved {shift:+.2f} mm"
    if noise <= 0.3:  # well separated: every riser survives
        assert np.all(np.diff(after) > 0.3), after


@pytest.mark.parametrize("rim", ["bevel15", "fillet2"])
def test_wobbly_panel_with_a_sloped_or_rounded_rim_is_flattened(rim):
    size = (80.0, 60.0)
    wf = scenes.WaveField(2).calibrate((0, size[0]), (0, size[1]))

    def top(x, y):
        d = np.minimum.reduce([x, size[0] - x, y, size[1] - y])
        z = 10.0 + wf(x, y, 1.0)
        if rim == "bevel15":
            drop = np.where(d < 5, (5 - d) * np.tan(np.radians(15)), 0.0)
        else:
            r = 2.0
            drop = np.where(d < r, r - np.sqrt(np.maximum(r * r - (r - d) ** 2, 0)), 0.0)
        return z - drop

    m = scenes.heightfield_solid(0, size[0], 0, size[1], 0.5, top, 0.0, 0.2, 2)
    res = flatten(m, G)
    c = m.face_centroids()
    inner = (m.face_normals_areas()[0][:, 2] > 0.5) & (np.minimum.reduce([c[:, 0], size[0] - c[:, 0], c[:, 1], size[1] - c[:, 1]]) > 7)
    v = np.unique(m.faces[inner])
    assert np.ptp(res.verts[v, 2]) < 1e-4, "the interior should be one flat plateau"


# ------------------------------------------------------------ robustness
def test_non_finite_coordinates_are_rejected():
    m = scenes.wavy_slab(cell=1.0).mesh
    bad = Mesh(m.verts.copy(), m.faces)
    bad.verts[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        flatten(bad, G)


def test_nothing_moves_below_the_bed():
    for sc in (scenes.wavy_slab(cell=1.0), scenes.panel_with_dome(cell=1.0), scenes.ceiling_pocket(cell=1.0)):
        res = flatten(sc.mesh, G)
        assert res.verts[:, 2].min() >= sc.mesh.verts[:, 2].min() - 1e-12


def test_tiny_layer_heights_are_rejected():
    for bad in (0.0, -0.2, 1e-9, float("nan"), float("inf"), 1000.0):
        with pytest.raises(ValueError):
            LayerGrid(bad)
    with pytest.raises(ValueError):
        LayerGrid(0.2, first_layer=1e-9)


def test_flat_tolerance_covers_float32_noise():
    # a flat pad whose corner heights differ by a few float32 ulps (z = 20 mm: 1 ulp = 1.9e-6)
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 20.0 + 0 * x, 0.0)
    v = m.verts.copy()
    rng = np.random.default_rng(1)
    top = v[:, 2] > 5
    v[top, 2] = (v[top, 2].astype(np.float32) + rng.integers(-2, 3, top.sum()).astype(np.float32) * np.float32(1.9e-6)).astype(np.float64)
    res = flatten(Mesh(v, m.faces), G)
    # it is an exactly-flat pad (not a wobbly panel): nothing "levelled", nothing dragged
    assert all(p.kind != "smoothed" for p in res.plateaus)


# ------------------------------------------------------------------ welding
def test_auto_weld_tolerance_follows_the_coordinates():
    small = np.full((2, 3, 3), 10.0)
    big = np.full((2, 3, 3), 500.0)
    assert auto_weld_tol(small) == pytest.approx(5e-5)
    assert auto_weld_tol(big) == pytest.approx(2.5e-4)
    # two copies of one corner one float32 step apart at x = 400 weld with the auto tolerance
    x = np.float32(400.0)
    nx = np.nextafter(x, np.float32(1e9))
    tris = np.array([[[x, 0, 0], [401, 0, 0], [400, 1, 0]], [[nx, 0, 0], [400, 1, 0], [399, 1, 0]]], dtype=np.float64)
    assert Mesh.from_triangles(tris).n_verts == 5  # the corner (400,1,0) is shared, the other copy is not
    assert Mesh.from_triangles(tris, tol=auto_weld_tol(tris)).n_verts == 4


# ------------------------------------------------------------------- reading
def _binary_with_solid_header(tmp_path, tail=b"", count=None):
    tris = scenes.wavy_slab(cell=2.0).mesh.to_triangles()
    p = tmp_path / "b.stl"
    stlio.write_stl(p, tris, header=b"solid exported by some cad program")
    raw = bytearray(p.read_bytes())
    if count is not None:
        struct.pack_into("<I", raw, 80, count)
    p.write_bytes(bytes(raw) + tail)
    return p, len(tris)


@pytest.mark.parametrize("tail,count", [(b"", None), (b"\n", None), (b"\r\n", None), (b"", 0), (b"", 999_999)],
                         ids=["exact", "lf", "crlf", "count0", "count-too-big"])
def test_binary_stl_with_a_solid_header_is_still_binary(tmp_path, tail, count):
    p, n = _binary_with_solid_header(tmp_path, tail, count)
    d = stlio.read_stl(p)
    assert d.source_format == "binary" and len(d.tris) == n


def test_ascii_with_bom_and_a_tricky_solid_name(tmp_path):
    tris = scenes.wavy_slab(cell=4.0).mesh.to_triangles()
    p = tmp_path / "a.stl"
    stlio.write_stl(p, tris, binary=False, name="vertex 1 2 3")
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    d = stlio.read_stl(p)
    assert d.source_format == "ascii" and len(d.tris) == len(tris)


# ---------------------------------------------------------------------- CLI
def _stl(tmp_path, name="m.stl"):
    p = tmp_path / name
    stlio.write_stl(p, scenes.wavy_slab(cell=1.0).mesh.to_triangles())
    return p


def cli(*argv):
    out, err = [], []
    code = run([str(a) for a in argv], out=out.append, err=err.append)
    return code, "\n".join(out), "\n".join(err)


def test_errors_go_to_the_error_stream_with_distinct_exit_codes(tmp_path):
    p = _stl(tmp_path)
    code, out, err = cli(tmp_path / "missing.stl")
    assert code == 2 and "no such file" in err and not out
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"not an stl")
    code, out, err = cli(bad)
    assert code == 1 and err.startswith("error:")
    code, out, err = cli(p, "-l", "0")
    assert code == 2 and "layer_height" in err


def test_unwritable_destinations_fail_before_any_work(tmp_path):
    p = _stl(tmp_path)
    assert cli(p, "-o", tmp_path / "nodir" / "x.stl")[0] == 2
    assert cli(p, "-o", tmp_path)[0] == 2  # a directory
    assert cli(p, "--analyze", "--report", tmp_path / "nodir" / "r.png")[0] == 2
    code, _, err = cli(p, "--analyze", "--report", tmp_path / "r.txt")
    assert code == 2 and "unsupported" in err


def test_report_without_extension_gets_png(tmp_path):
    pytest.importorskip("matplotlib")
    p = _stl(tmp_path)
    code, out, _ = cli(p, "--analyze", "--report", tmp_path / "pic")
    assert code == 0 and (tmp_path / "pic.png").exists() and "pic.png" in out


def test_a_failing_report_never_prevents_the_stl(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import stl_smoothing.report as report

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(report, "render_comparison", boom)
    p = _stl(tmp_path)
    code, out, err = cli(p, "--report", tmp_path / "r.png")
    assert code == 0 and "could not write the report" in out
    assert (tmp_path / "m_smoothed.stl").exists()


def test_non_finite_input_is_a_clean_error(tmp_path):
    tris = scenes.wavy_slab(cell=2.0).mesh.to_triangles()
    tris[7, 1, 2] = np.inf
    p = tmp_path / "inf.stl"
    stlio.write_stl(p, tris)
    code, out, err = cli(p)
    assert code == 1 and "NaN or infinite" in err


def test_layer_height_sanity_bound(tmp_path):
    p = _stl(tmp_path)
    for h in ("1e-9", "100"):
        code, _, err = cli(p, "-l", h)
        assert code == 2 and "layer_height" in err


def test_long_lists_collapse(tmp_path):
    # many separate exactly-flat pads on a sampling plane: the summary must stay short
    n = 6
    def top(x, y):
        pad = ((x // 10) % 2 == 0) & ((y // 10) % 2 == 0)
        return np.where(pad, 10.5, 6.0)
    m = scenes.heightfield_solid(0, 10 * (2 * n), 0, 10 * (2 * n), 2.0, top, 0.0)
    p = tmp_path / "pads.stl"
    stlio.write_stl(p, m.to_triangles())
    code, out, _ = cli(p, "--analyze")
    assert code == 0
    assert len(out.splitlines()) < 40
