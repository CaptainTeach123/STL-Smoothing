"""The browser runs numpy on a 32-bit platform (WebAssembly), where ``np.intp`` is 32 bits.

numpy's ``bincount``, ``repeat`` and ``ufunc.reduceat`` then refuse 64-bit integer index arrays ("Cannot
cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'") - the failure the
first real in-browser run hit.  These tests run the pipeline on numpy patched to behave like that
platform (see wasm32_emu.py), so a call that skips the wrappers in ``stl_smoothing._compat`` fails here
instead of only in the browser, and the result must be identical to the normal one.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import scenes
import wasm32_emu
from stl_smoothing import _compat, stlio, web

SRC = Path(__file__).resolve().parent.parent / "src" / "stl_smoothing"


def test_the_emulation_really_rejects_64_bit_indices(monkeypatch):
    wasm32_emu.install(monkeypatch)
    big = np.array([0, 1, 1], dtype=np.int64)
    with pytest.raises(TypeError, match="safe"):
        np.bincount(big)
    with pytest.raises(TypeError, match="safe"):
        np.repeat(np.arange(3), big)
    with pytest.raises(TypeError, match="safe"):
        np.minimum.reduceat(np.arange(4.0), np.array([0, 2], dtype=np.int64))
    with pytest.raises(TypeError, match="safe"):
        np.bincount(np.array([0, 1], dtype=np.uint32))
    assert np.argsort(np.array([2.0, 1.0])).dtype == np.int32
    assert np.arange(3).dtype == np.int32
    assert list(_compat.bincount(big)) == [1, 2]
    assert list(_compat.reduceat(np.add, np.arange(4.0), np.array([0, 2], dtype=np.int64))) == [1.0, 5.0]


def test_wrappers_leave_other_inputs_alone():
    assert list(_compat.bincount(np.array([], dtype=np.int64), minlength=2)) == [0, 0]
    assert list(_compat.bincount([2, 0], weights=np.array([0.5, 1.5]))) == [1.5, 0.0, 0.5]
    assert list(_compat.bincount(np.array([True, False, True]))) == [1, 2]
    assert _compat.as_index(np.array([1.5])).dtype == np.float64
    assert _compat.as_index(np.arange(3, dtype=np.intp)).dtype == np.intp


def _process(folder, scene):
    folder.mkdir()
    p = folder / "m.stl"
    stlio.write_stl(p, scene.mesh.to_triangles(), header="x")
    out = folder / "out.stl"
    meta = json.loads(web.process(str(p), str(out), json.dumps({"layer_height": 0.2}), preview_dir=str(folder / "pv")))
    arrays = {f.name: f.read_bytes() for f in sorted((folder / "pv").glob("*.bin"))} if (folder / "pv").exists() else {}
    return meta, (out.read_bytes() if out.exists() else b""), arrays


SCENES = scenes.all_synthetic(small=True) + scenes.all_holdout(small=True)


@pytest.mark.parametrize("scene", SCENES, ids=[s.name for s in SCENES])
def test_pipeline_gives_the_same_answer_with_32_bit_indices(scene, tmp_path, monkeypatch):
    ref_meta, ref_out, ref_pv = _process(tmp_path / "ref", scene)
    with monkeypatch.context() as m:
        calls = wasm32_emu.install(m)
        w_meta, w_out, w_pv = _process(tmp_path / "w32", scene)
    assert calls["bincount"] > 0, "the emulation was never exercised"
    assert w_meta["ok"] and ref_meta["ok"]
    assert w_meta["summary"] == ref_meta["summary"]
    assert w_meta["n_moved"] == ref_meta["n_moved"]
    assert w_out == ref_out
    assert w_pv == ref_pv


def test_the_picture_code_runs_under_the_emulation(tmp_path, monkeypatch):
    """The layer-edge (slicing) code calls np.repeat with an array of counts, the strictest case."""
    scene = scenes.panel_with_dome(cell=2.4, ptp=1.2, jitter=0.2)
    ref_meta, _, ref_pv = _process(tmp_path / "ref", scene)
    assert ref_pv, "the scene should produce a picture"
    with monkeypatch.context() as m:
        calls = wasm32_emu.install(m)
        w_meta, _, w_pv = _process(tmp_path / "w32", scene)
    assert calls["repeat"] > 0 and calls["reduceat"] > 0
    assert w_pv == ref_pv


def test_numpy_index_functions_are_only_called_through_the_wrappers():
    """``np.bincount`` / ``.reduceat`` anywhere else would skip the 32-bit protection."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "_compat.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\bnp\.bincount\(|\.reduceat\(", code):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "use stl_smoothing._compat instead:\n" + "\n".join(offenders)


def test_selftest_passes_natively_and_with_32_bit_indices(monkeypatch):
    assert json.loads(web.selftest())["ok"] is True
    with monkeypatch.context() as m:
        wasm32_emu.install(m)
        assert json.loads(web.selftest())["ok"] is True


def test_selftest_catches_the_original_browser_failure(monkeypatch):
    """With the casts removed the built-in model fails, so the page would report an engine error at load."""
    wasm32_emu.install(monkeypatch)
    monkeypatch.setattr(_compat, "as_index", np.asarray)
    result = json.loads(web.selftest())
    assert result["ok"] is False
    assert "safe" in result["error"] and "Traceback" in result["detail"]
