"""End-to-end tests of the command line tool."""

import numpy as np
import pytest

import scenes
from stl_smoothing import stlio
from stl_smoothing.cli import run
from stl_smoothing.mesh import Mesh


@pytest.fixture()
def wavy_stl(tmp_path):
    sc = scenes.wavy_slab(cell=1.0)
    p = tmp_path / "wavy.stl"
    stlio.write_stl(p, sc.mesh.to_triangles(), header="test model")
    return p, sc


def cli(*argv):
    lines = []
    code = run([str(a) for a in argv], out=lines.append)
    return code, "\n".join(lines)


def test_default_output_name_and_result(wavy_stl):
    p, sc = wavy_stl
    code, text = cli(p)
    out = p.with_name("wavy_smoothed.stl")
    assert code == 0 and out.exists()
    assert "Flattened 1 surface" in text
    assert "Layer edges on those surfaces" in text
    new = Mesh.from_triangles(stlio.read_stl(out).tris)
    assert new.n_faces == sc.mesh.n_faces
    top_z = new.verts[new.verts[:, 2] > 5, 2]
    assert top_z.max() - top_z.min() < 1e-5


def test_header_and_triangle_count_survive(wavy_stl):
    p, sc = wavy_stl
    cli(p)
    d = stlio.read_stl(p.with_name("wavy_smoothed.stl"))
    assert d.header.startswith(b"test model")
    assert len(d.tris) == sc.mesh.n_faces


def test_analyze_writes_nothing(wavy_stl):
    p, _ = wavy_stl
    code, text = cli(p, "--analyze")
    assert code == 0 and "Analyze only" in text
    assert not p.with_name("wavy_smoothed.stl").exists()


def test_explicit_output_and_ascii(wavy_stl, tmp_path):
    p, _ = wavy_stl
    out = tmp_path / "x.stl"
    code, _ = cli(p, "-o", out, "--ascii")
    assert code == 0
    assert stlio.read_stl(out).source_format == "ascii"


def test_refuses_to_overwrite_the_input(wavy_stl):
    p, _ = wavy_stl
    before = p.read_bytes()
    code, text = cli(p, "-o", p)
    assert code == 2 and "overwrite" in text
    assert p.read_bytes() == before


def test_nothing_to_do_writes_no_file(tmp_path):
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 9.8 + 0 * x, 0.0)
    p = tmp_path / "flat.stl"
    stlio.write_stl(p, m.to_triangles())
    code, text = cli(p)
    assert code == 0 and "no STL written" in text
    assert not p.with_name("flat_smoothed.stl").exists()


def test_errors(tmp_path):
    assert cli(tmp_path / "missing.stl")[0] == 2
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"not an stl")
    assert cli(bad)[0] == 1
    empty = tmp_path / "empty.stl"
    empty.write_text("solid nothing\nendsolid nothing\n")
    code, text = cli(empty)
    assert code == 1 and "no triangles" in text


def test_layer_height_option(wavy_stl):
    p, _ = wavy_stl
    code, text = cli(p, "--analyze", "-l", "0.28", "--first-layer", "0.3")
    assert code == 0 and "0.28 mm layers" in text
    assert cli(p, "-l", "0")[0] == 2


def test_max_range_option_protects_the_surface(wavy_stl):
    p, _ = wavy_stl
    code, text = cli(p, "--analyze", "--max-range", "0.5")
    assert "nothing to flatten" in text


def test_set_option(wavy_stl):
    p, _ = wavy_stl
    code, text = cli(p, "--analyze", "--set", "max_range=0.5")
    assert "nothing to flatten" in text
    with pytest.raises(SystemExit):
        cli(p, "--set", "no_such_param=1")
    with pytest.raises(SystemExit):
        cli(p, "--set", "max_range")
    with pytest.raises(SystemExit):
        cli(p, "--set", "max_range=abc")


def test_open_mesh_warning(tmp_path):
    sc = scenes.wavy_slab(cell=1.0)
    tris = sc.mesh.to_triangles()[:-30]
    p = tmp_path / "open.stl"
    stlio.write_stl(p, tris)
    code, text = cli(p, "--analyze")
    assert code == 0 and "not closed" in text


def test_report_png(wavy_stl, tmp_path):
    pytest.importorskip("matplotlib")
    p, _ = wavy_stl
    png = tmp_path / "r.png"
    code, text = cli(p, "--analyze", "--report", png)
    assert code == 0 and png.exists() and png.stat().st_size > 5000
    assert "layers before" in text


def test_verbose_lists_skipped(wavy_stl):
    p, _ = wavy_stl
    code, text = cli(p, "--analyze", "-v")
    assert code == 0
