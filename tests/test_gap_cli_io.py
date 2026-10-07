"""CLI wiring, output fidelity and STL I/O details that no existing test pinned down."""

import re
import struct

import numpy as np
import pytest

import metrics
import scenes
from stl_smoothing import stlio
from stl_smoothing.cli import run
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh
from stl_smoothing.slicing import contour_length, slice_planes


def cli(*argv):
    lines = []
    code = run([str(a) for a in argv], out=lines.append)
    return code, "\n".join(lines)


def write(tmp_path, fname, mesh, **kw):
    p = tmp_path / fname
    stlio.write_stl(p, mesh.to_triangles(), **kw)
    return p


@pytest.fixture()
def wavy(tmp_path):
    sc = scenes.wavy_slab(cell=1.0)
    return write(tmp_path, "wavy.stl", sc.mesh, header="test model"), sc


def top_z(path):
    z = stlio.read_stl(path).tris[:, :, 2].ravel()
    return z[z > 5]


def flat_at_10_5(tmp_path):
    return write(tmp_path, "flat.stl", scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 10.5 + 0 * x, 0.0))


# ------------------------------------------------------------ option wiring
def test_first_layer_option_selects_the_layer_grid(wavy, tmp_path):
    p, _ = wavy
    out = tmp_path / "default.stl"
    assert cli(p, "-o", out)[0] == 0
    assert np.allclose(top_z(out), 10.0, atol=1e-4)  # boundaries 0.2, 0.4 ...
    out2 = tmp_path / "first.stl"
    assert cli(p, "-o", out2, "-l", "0.2", "--first-layer", "0.3")[0] == 0
    assert np.allclose(top_z(out2), 10.1, atol=1e-4)  # boundaries 0.3, 0.5 ... 10.1


@pytest.mark.parametrize("extra", [["--no-snap-exact"], ["--set", "snap_exact_flat=false"], ["--set", "snap_exact_flat=no"]])
def test_snapping_of_exactly_flat_surfaces_can_be_switched_off(tmp_path, extra):
    p = flat_at_10_5(tmp_path)
    out = tmp_path / "o.stl"
    code, text = cli(p, "-o", out, *extra)
    assert code == 0 and "Nothing was changed" in text and not out.exists()
    code, text = cli(p, "-o", out)  # default and explicit true: nudged off the sampling plane
    assert code == 0 and out.exists() and "off a slicer sampling plane" in text
    out.unlink()
    assert cli(p, "-o", out, "--set", "snap_exact_flat=true")[0] == 0 and out.exists()


def test_max_slope_and_min_area_options_reach_the_algorithm(tmp_path):
    p = write(tmp_path, "t.stl", scenes.tilted_flat(angle_deg=0.3, cell=1.0).mesh)
    assert "Flattened 1 surface" in cli(p, "--analyze")[1]
    assert "Nothing to flatten" in cli(p, "--analyze", "--max-slope", "0.1")[1]
    assert "Nothing to flatten" in cli(p, "--analyze", "--min-area", "100000")[1]


def test_weld_tolerance_option(tmp_path):
    """Corners that differ in the last float32 bit (a common exporter artefact) are welded by default,
    so the surface is still seen as one piece; --weld-tol 0 asks for exact matches only."""
    t32 = scenes.wavy_slab(cell=1.0).mesh.to_triangles().astype(np.float32)
    rng = np.random.default_rng(0)
    hit = rng.random(t32.shape[:2]) < 0.4
    for k in (0, 1):
        up = np.where(rng.random(t32.shape[:2]) < 0.5, np.float32(np.inf), np.float32(-np.inf))
        t32[..., k] = np.where(hit, np.nextafter(t32[..., k], up), t32[..., k])
    p = tmp_path / "cracked.stl"
    stlio.write_stl(p, t32.astype(np.float64))
    out = tmp_path / "o.stl"
    code, text = cli(p, "-o", out)
    assert code == 0 and "Flattened 1 surface" in text
    assert np.ptp(top_z(out)) < 1e-5
    out.unlink()
    code, text = cli(p, "-o", out, "--weld-tol", "0")
    assert "Nothing to flatten" in text and not out.exists()
    assert "Flattened 1 surface" in cli(p, "-o", out, "--weld-tol", "1e-4")[1]


# ------------------------------------------------------------ output fidelity
def test_attribute_bytes_and_ascii_solid_name_survive(tmp_path):
    sc = scenes.wavy_slab(cell=1.0)
    attrs = (np.arange(sc.mesh.n_faces) % 65535).astype(np.uint16)
    p = write(tmp_path, "a.stl", sc.mesh, attrs=attrs)
    out = tmp_path / "o.stl"
    assert cli(p, "-o", out)[0] == 0
    np.testing.assert_array_equal(stlio.read_stl(out).attrs, attrs)
    pa = write(tmp_path, "ascii.stl", sc.mesh, binary=False, name="my_relief")
    outa = tmp_path / "oa.stl"
    assert cli(pa, "-o", outa, "--ascii")[0] == 0
    assert stlio.read_stl(outa).name == "my_relief"
    assert outa.read_text().startswith("solid my_relief")


# ------------------------------------------------------------ what the summary says
def test_summary_numbers(wavy):
    p, _ = wavy
    code, text = cli(p, "--analyze")
    assert re.search(r"Layer edges on those surfaces: ([\d,]{3,}) mm -> 0 mm", text), text
    assert re.search(r"top\s+1 patch\s+[\d,]+ mm²\s+9\.\d\d–10\.\d\d mm\s+[4-7] layers\s+->\s+10\.00 mm", text), text
    assert "Note: a flattened surface varied by more than 1 mm" in text
    assert re.search(r"Moved [\d,]+ vertices, largest move 0\.\d\d mm, in z only; 0 flipped faces", text), text


def test_running_the_tool_on_its_own_output_changes_nothing(tmp_path):
    """README: 'Running it again on its own output normally changes nothing.'  Through the files, i.e.
    after the float32 round trip of the STL (9.8 mm is not exactly representable in float32)."""
    sc = scenes.wavy_slab(cell=1.0, base=9.8, ptp=0.8)
    p = write(tmp_path, "w.stl", sc.mesh)
    out = tmp_path / "once.stl"
    assert cli(p, "-o", out)[0] == 0 and out.exists()
    assert np.allclose(top_z(out), 9.8, atol=1e-5)
    code, text = cli(out, "-o", tmp_path / "twice.stl")
    assert "Nothing was changed" in text and not (tmp_path / "twice.stl").exists(), text


def test_documented_defaults():
    """The README option table and --help promise these defaults."""
    from stl_smoothing.cli import build_parser
    from stl_smoothing.params import Params

    P = Params()
    assert (P.max_slope_deg, P.max_range, P.min_area) == (5.0, 2.0, 40.0)
    a = build_parser().parse_args(["x.stl"])
    assert a.layer_height == 0.2 and a.first_layer is None and a.weld_tol is None  # None = automatic
    assert a.max_range is None and a.max_slope is None and a.min_area is None  # "unset" = the Params default


def test_levelled_and_snapped_messages(tmp_path):
    graze = write(tmp_path, "graze.stl", scenes.heightfield_solid(
        0, 100, 0, 80, 1.0, lambda x, y: 10 + 0.095 * np.sin(2 * np.pi * x / 40) * np.sin(2 * np.pi * y / 40), 0.0))
    code, text = cli(graze, "--analyze")
    assert "Levelled 1 nearly-flat surface that grazed a slicer sampling plane" in text and "Flattened" not in text
    code, text = cli(flat_at_10_5(tmp_path), "--analyze")
    assert "Moved 1 already-flat surface off a slicer sampling plane" in text
    assert re.search(r"flat at 10\.500 mm -> 10\.[46]0 mm \(on a sampling plane\)", text), text


def test_verbose_lists_what_was_left_alone_and_why(wavy):
    p, _ = wavy
    code, text = cli(p, "--analyze", "--max-range", "0.5")
    assert "Left alone: 1 candidate (use -v for details)" in text and "exceeds max_range" not in text
    code, text = cli(p, "--analyze", "--max-range", "0.5", "-v")
    assert "top surface near" in text and "exceeds max_range" in text


def test_report_is_written_for_ceilings_too(tmp_path):
    pytest.importorskip("matplotlib")
    size = (90.0, 70.0)
    wf = scenes.WaveField(9).calibrate((0, size[0]), (0, size[1]))
    both = scenes.heightfield_solid(0, size[0], 0, size[1], 1.0, lambda x, y: 12.0 + wf(x, y, 1.0),
                                    lambda x, y: np.where((x > 20) & (x < 70) & (y > 15) & (y < 55), 4.0 + wf(x, y, 1.0), 0.0), 0.2, 9)
    p = write(tmp_path, "both.stl", both)
    png = tmp_path / "r.png"
    code, text = cli(p, "--analyze", "--report", png)
    assert code == 0 and png.exists()
    ceil = tmp_path / "r_ceilings.png"
    assert ceil.exists() and ceil.stat().st_size > 5000, text
    assert text.count("Wrote report") == 2


# ------------------------------------------------------------ STL I/O
def test_ascii_with_incomplete_triangle_is_rejected(tmp_path):
    p = tmp_path / "bad.stl"
    p.write_text("solid x\n facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex 1 0 0\n endloop\n endfacet\nendsolid x\n")
    with pytest.raises(stlio.StlError):
        stlio.read_stl(p)


def test_ascii_coordinates_are_float32_like_binary_ones(tmp_path):
    p = tmp_path / "f.stl"
    p.write_text("solid x\n facet normal 0 0 1\n outer loop\n vertex 0.1 0.2 0.3\n vertex 1.1 0 0\n vertex 0 1.1 0\n"
                 " endloop\n endfacet\nendsolid x\n")
    t = stlio.read_stl(p).tris
    assert t[0, 0, 0] == float(np.float32(0.1)) and t[0, 0, 2] == float(np.float32(0.3))


def test_ascii_writer_keeps_float32_precision(tmp_path):
    rng = np.random.default_rng(3)
    tris = rng.uniform(-100, 100, (20, 3, 3)).astype(np.float32).astype(np.float64)
    p = tmp_path / "p.stl"
    stlio.write_stl(p, tris, binary=False)
    np.testing.assert_array_equal(stlio.read_stl(p).tris, tris)  # 9 significant digits round-trip a float32 exactly


def test_binary_writer_truncates_long_header_and_writes_facet_normals(tmp_path):
    tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1], [1, 0, 0]]], dtype=float)
    p = tmp_path / "h.stl"
    stlio.write_stl(p, tris, header=b"x" * 200)
    raw = p.read_bytes()
    assert len(raw) == 84 + 50 * 2 and raw[:80] == b"x" * 80
    assert stlio.read_stl(p).source_format == "binary"
    rec = np.frombuffer(raw, dtype=stlio._BIN_DTYPE, count=2, offset=84)
    np.testing.assert_allclose(rec["n"], [[0, 0, 1], [0, 1, 0]], atol=1e-7)
    pa = tmp_path / "a.stl"
    stlio.write_stl(pa, tris, binary=False)
    normals = re.findall(r"facet normal (\S+) (\S+) (\S+)", pa.read_text())
    np.testing.assert_allclose(np.array(normals, dtype=float), [[0, 0, 1], [0, 1, 0]], atol=1e-7)


# ------------------------------------------------------------ layer grid / slicer emulation
def test_snap_never_goes_below_the_first_layer():
    g = LayerGrid(0.2)
    np.testing.assert_allclose(g.snap([0.0, 0.04, 0.1, 0.29]), [0.2, 0.2, 0.2, 0.2], atol=1e-12)
    g3 = LayerGrid(0.2, first_layer=0.3)
    np.testing.assert_allclose(g3.snap([0.0, 0.2]), [0.3, 0.3], atol=1e-12)


def test_first_layer_sampling_plane_is_counted():
    """A surface that climbs from 0 to 0.2 mm crosses only the first layer's sampling plane (0.1 mm)."""
    m = scenes.heightfield_solid(0, 10, 0, 10, 1.0, lambda x, y: 0.019 * x, 0.0)
    top = m.face_normals_areas()[0][:, 2] > 0.5
    assert 0.1 in np.round(slice_planes(LayerGrid(0.2), 0.0, 0.2), 9)
    assert contour_length(m, LayerGrid(0.2), top) == pytest.approx(10.0, rel=0.02)
