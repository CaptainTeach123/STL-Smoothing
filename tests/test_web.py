"""The Python side of the browser app: bytes I/O, picture data, and the process() entry point."""

import json
import os

import numpy as np
import pytest

import scenes
from stl_smoothing import preview, stlio, web
from stl_smoothing.cli import run
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid
from stl_smoothing.mesh import Mesh


@pytest.fixture()
def wavy(tmp_path):
    sc = scenes.wavy_slab(cell=1.0)
    p = tmp_path / "wavy.stl"
    stlio.write_stl(p, sc.mesh.to_triangles(), header="hello world")
    return p, sc


def process(p, tmp_path, opts=None, **kw):
    out = tmp_path / "out.stl"
    pv = tmp_path / "pv"
    msgs = []
    meta = json.loads(web.process(str(p), str(out), json.dumps(opts or {}), progress=msgs.append,
                                  preview_dir=str(pv), **kw))
    return meta, out, pv, msgs


# ----------------------------------------------------------------- bytes I/O
def test_stl_bytes_round_trip_matches_the_file_functions(tmp_path):
    tris = scenes.wavy_slab(cell=2.0).mesh.to_triangles()
    p = tmp_path / "a.stl"
    stlio.write_stl(p, tris, header="abc")
    assert p.read_bytes() == stlio.stl_bytes(tris, header="abc")
    d = stlio.parse_stl(p.read_bytes())
    assert d.source_format == "binary" and d.header.startswith(b"abc")
    np.testing.assert_array_equal(d.tris, tris.astype(np.float32).astype(np.float64))  # STL stores float32
    a = stlio.stl_bytes(tris, binary=False, name="x")
    assert stlio.parse_stl(a).source_format == "ascii"


def test_parse_stl_reports_the_given_source_name():
    with pytest.raises(stlio.StlError, match="my model"):
        stlio.parse_stl(b"garbage that is not an stl", source="my model")


# ------------------------------------------------------------------ process()
def test_process_matches_the_command_line(wavy, tmp_path):
    p, sc = wavy
    meta, out, pv, msgs = process(p, tmp_path)
    assert meta["ok"] and meta["triangles"] == sc.mesh.n_faces
    assert meta["flattened"] == 1 and meta["n_moved"] > 0
    assert meta["edges_before"] > 100 and meta["edges_after"] == 0
    assert meta["closed"] and meta["flipped"] == 0
    cli_out = tmp_path / "cli.stl"
    assert run([str(p), "-o", str(cli_out)], out=lambda s: None) == 0
    assert out.read_bytes() == cli_out.read_bytes(), "the web path and the CLI must write identical STLs"


def test_process_summary_is_the_cli_summary(wavy, tmp_path):
    p, _ = wavy
    meta, *_ = process(p, tmp_path)
    lines = []
    run([str(p), "--analyze"], out=lines.append)
    cli_lines = lines[1:-1]  # minus the file line and the "Analyze only" line
    # identical except that hints name the page's settings instead of command-line options
    assert [x for x in meta["summary"] if "run again with" not in x] == [x for x in cli_lines if "run again with" not in x]
    hint = [x for x in meta["summary"] if "run again with" in x]
    assert hint and "Largest wobble" in hint[0] and "--max-range" not in hint[0]
    assert not any("-v" in x.split() for x in meta["summary"])


def test_web_summary_has_no_command_line_hints(tmp_path):
    sc = scenes.two_level_panels(cell=2.0)
    p = tmp_path / "m.stl"
    stlio.write_stl(p, sc.mesh.to_triangles())
    text = "\n".join(process(p, tmp_path, {"layer_height": 0.2, "max_range": 0.3})[0].get("summary", []))
    assert "use -v" not in text and "--max-range" not in text


def test_process_reports_progress_in_order(wavy, tmp_path):
    p, _ = wavy
    _, _, _, msgs = process(p, tmp_path)
    assert msgs[0].startswith("Reading") and msgs[-1] == "Done" and len(msgs) >= 5


def test_process_options_reach_the_algorithm(wavy, tmp_path):
    p, _ = wavy
    meta, out, *_ = process(p, tmp_path, {"layer_height": 0.28, "first_layer": 0.3})
    assert meta["layer_height"] == 0.28 and meta["first_layer"] == 0.3
    assert "0.28 mm layers" in "\n".join(meta["summary"])
    protected, *_ = process(p, tmp_path, {"max_range": 0.5})
    assert protected["n_moved"] == 0 and protected["flattened"] == 0
    assert not protected["preview"] and protected["output_bytes"] == 0


def test_nothing_to_do_writes_no_output_and_no_picture(tmp_path):
    m = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 9.8 + 0 * x, 0.0)
    p = tmp_path / "flat.stl"
    stlio.write_stl(p, m.to_triangles())
    meta, out, pv, _ = process(p, tmp_path)
    assert meta["ok"] and meta["n_moved"] == 0 and not out.exists() and not pv.exists()


@pytest.mark.parametrize("make,needle", [
    (lambda tmp: b"not an stl at all", "not a recognisable"),
    (lambda tmp: stlio.stl_bytes(np.zeros((0, 3, 3))), "no triangles"),
])
def test_process_rejects_bad_files_with_a_message(tmp_path, make, needle):
    p = tmp_path / "bad.stl"
    p.write_bytes(make(tmp_path))
    meta, *_ = process(p, tmp_path)
    assert meta["ok"] is False and needle in meta["error"]


def test_process_rejects_non_finite_coordinates(tmp_path):
    tris = scenes.wavy_slab(cell=2.0).mesh.to_triangles()
    tris[3, 0, 2] = np.nan
    p = tmp_path / "nan.stl"
    stlio.write_stl(p, tris)
    meta, *_ = process(p, tmp_path)
    assert meta["ok"] is False and "NaN" in meta["error"]


def test_process_rejects_bad_options(wavy, tmp_path):
    p, _ = wavy
    for bad in ({"layer_height": 0}, {"layer_height": "abc"}, {"layer_height": 1e-9}):
        meta, *_ = process(p, tmp_path, bad)
        assert meta["ok"] is False and meta["error"]


def test_process_missing_file(tmp_path):
    meta = json.loads(web.process(str(tmp_path / "nope.stl"), str(tmp_path / "o.stl")))
    assert meta["ok"] is False


# ---------------------------------------------------------------- picture data
def test_preview_files_are_consistent(wavy, tmp_path):
    p, sc = wavy
    meta, out, pv, _ = process(p, tmp_path)
    m = meta["preview"]
    F = meta["triangles"]
    assert m["n_faces"] == F and len(m["views"]) == 1
    view = m["views"][0]
    assert view["key"] == "top" and view["after_layers"] == 1 and view["before_layers"] > 1
    assert len(view["levels"]) == len(view["colours"]) >= view["before_layers"]
    assert view["after_text"].endswith("(%.2f mm)" % view["levels"][-1]) or "one layer" in view["after_text"]
    files = m["files"]
    arrs = {k: np.fromfile(v["file"], dtype=v["dtype"]) for k, v in files.items()}
    assert arrs["xy"].size == F * 6
    for state in ("before", "after"):
        order = arrs[f"top_{state}_order"]
        assert order.max() < F and len(np.unique(order)) == len(order)
        assert arrs[f"top_{state}_shade"].size == F and arrs[f"top_{state}_level"].size == F
        lvl = arrs[f"top_{state}_level"]
        assert set(np.unique(lvl)) <= set(range(len(view["levels"]))) | {255}
        assert arrs[f"top_{state}_segs"].size % 4 == 0
    # the flattened panel sits on ONE level after, several before
    before_levels = np.unique(arrs["top_before_level"][arrs["top_before_level"] != 255])
    after_levels = np.unique(arrs["top_after_level"][arrs["top_after_level"] != 255])
    assert len(before_levels) > 1 and len(after_levels) == 1
    # no layer edges are left on the flattened surface
    assert arrs["top_after_segs"].size < 0.05 * arrs["top_before_segs"].size


def test_preview_has_a_ceiling_view_for_ceilings(tmp_path):
    sc = scenes.ceiling_pocket(cell=1.0)
    p = tmp_path / "c.stl"
    stlio.write_stl(p, sc.mesh.to_triangles())
    meta, *_ = process(p, tmp_path)
    keys = [v["key"] for v in meta["preview"]["views"]]
    assert "bottom" in keys
    assert next(v for v in meta["preview"]["views"] if v["key"] == "bottom")["mirror_x"] is True


def test_palette():
    assert preview.palette_hex(0.0) == "#4b63d1" and preview.palette_hex(1.0) == "#b8232f"
    assert preview.palette_hex(0.4) == "#c4d3f5"
    cols = preview.level_colours(np.array([9.4, 9.6, 9.8, 10.0]), final=np.array([9.8]))
    assert cols[9.8] == "#c4d3f5"  # the level the surface ends on is pale blue
    assert cols[9.4] == "#4b63d1" and cols[10.0] == "#b8232f"
    assert preview.level_colours(np.array([]), np.array([])) == {}


def test_nearest_level():
    lv = np.array([9.4, 9.8, 10.2])
    np.testing.assert_array_equal(preview.nearest_level(lv, np.array([9.0, 9.5, 9.7, 9.95, 10.05, 11.0])), [0, 0, 1, 1, 2, 2])


def test_report_and_preview_agree_on_the_levels(wavy, tmp_path):
    pytest.importorskip("matplotlib")
    from stl_smoothing.report import render_comparison

    p, sc = wavy
    grid = LayerGrid(0.2)
    res = flatten(sc.mesh, grid)
    after = Mesh(res.verts, sc.mesh.faces)
    region = np.isin(res.face_plateau, [i for i, q in enumerate(res.plateaus) if q.layers_before > 1])
    info = render_comparison(sc.mesh, after, grid, region, str(tmp_path / "r.png"))
    meta, _ = preview.build_preview(sc.mesh, after, grid, res)
    v = meta["views"][0]
    assert v["before_layers"] == len(info["before_layers"]) and v["after_layers"] == len(info["after_layers"])


def test_picture_limits_layer_edges_instead_of_exhausting_memory(wavy, tmp_path, monkeypatch):
    p, _ = wavy
    meta, *_ = process(p, tmp_path)
    assert "edge_note" not in meta["preview"]["views"][0]
    monkeypatch.setattr(preview, "MAX_EDGE_SEGMENTS", 50)
    (tmp_path / "again").mkdir()
    meta, _, pv, _ = process(p, tmp_path / "again")
    view = meta["preview"]["views"][0]
    assert "too many layer edges" in view["edge_note"]
    assert meta["preview"]["files"]["top_before_segs"]["length"] == 0
    assert meta["ok"] and meta["flattened"] == 1  # smoothing itself is unaffected


def test_unexpected_errors_become_a_message_with_details(wavy, tmp_path, monkeypatch):
    p, _ = wavy

    def boom(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr(web, "flatten", boom)
    meta, out, *_ = process(p, tmp_path)
    assert meta["ok"] is False and "MemoryError" in meta["error"] and "Traceback" in meta["detail"]
    assert not out.exists()


def test_file_errors_are_sentences(tmp_path):
    p = tmp_path / "x.stl"
    p.write_bytes(b"this is not an stl file")
    meta, *_ = process(p, tmp_path)
    assert meta["error"].startswith("That file could not be read: not a recognisable")
    assert meta["error"].endswith("STL file.")
