"""The browser runs numpy on a 32-bit platform (WebAssembly), where ``np.intp`` is 32 bits.

numpy's ``bincount`` and ``ufunc.reduceat`` then refuse 64-bit integer index arrays ("Cannot cast array
data from dtype('int64') to dtype('int32') according to the rule 'safe'") - the failure the first real
in-browser run hit.  These tests behave like that platform: the index dtype becomes int32 and the
strict versions of those functions raise on any 64-bit integer array, so a call that skips the
wrappers in ``stl_smoothing._compat`` fails here instead of only in the browser.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import scenes
from stl_smoothing import _compat, stlio, web

SRC = Path(__file__).resolve().parent.parent / "src" / "stl_smoothing"
STRICT_UFUNCS = ("minimum", "maximum", "add", "logical_or")


def _check_index_dtype(name, a):
    a = np.asarray(a)
    if a.dtype.kind in "iu" and a.dtype.itemsize > 4:
        raise TypeError(f"{name}: Cannot cast array data from dtype('{a.dtype}') to dtype('int32') "
                        "according to the rule 'safe'")


class _StrictUfunc:
    """Stands in for a numpy ufunc; ``reduceat`` behaves like numpy on a 32-bit platform."""

    def __init__(self, ufunc, calls):
        self._u = ufunc
        self._calls = calls

    def __call__(self, *a, **k):
        return self._u(*a, **k)

    def __getattr__(self, name):
        return getattr(self._u, name)

    def reduceat(self, a, indices, *rest, **kw):
        self._calls["reduceat"] += 1
        _check_index_dtype("reduceat", indices)
        return self._u.reduceat(a, indices, *rest, **kw)


def _install(m):
    """Make numpy behave like a 32-bit platform for the rest of the ``monkeypatch`` scope."""
    calls = {"bincount": 0, "reduceat": 0}
    real_bincount = np.bincount

    def strict_bincount(x, weights=None, minlength=0):
        calls["bincount"] += 1
        _check_index_dtype("bincount", x)
        return real_bincount(x, weights=weights, minlength=minlength)

    m.setattr(_compat, "INTP", np.dtype(np.int32))
    m.setattr(np, "bincount", strict_bincount)
    for name in STRICT_UFUNCS:
        m.setattr(np, name, _StrictUfunc(getattr(np, name), calls))
    return calls


@pytest.fixture()
def like_wasm(monkeypatch):
    return _install(monkeypatch)


def test_strict_functions_really_reject_64_bit_indices(like_wasm):
    with pytest.raises(TypeError, match="safe"):
        np.bincount(np.array([0, 1, 1], dtype=np.int64))
    with pytest.raises(TypeError, match="safe"):
        np.minimum.reduceat(np.arange(4.0), np.array([0, 2], dtype=np.int64))
    assert list(_compat.bincount(np.array([0, 1, 1], dtype=np.int64))) == [1, 2]
    assert list(_compat.reduceat(np.add, np.arange(4.0), np.array([0, 2], dtype=np.int64))) == [1.0, 5.0]


def test_wrappers_leave_other_inputs_alone():
    assert list(_compat.bincount(np.array([], dtype=np.int64), minlength=2)) == [0, 0]
    assert list(_compat.bincount([2, 0], weights=np.array([0.5, 1.5]))) == [1.5, 0.0, 0.5]
    assert list(_compat.bincount(np.array([True, False, True]))) == [1, 2]
    assert _compat.as_index(np.array([1.5])).dtype == np.float64
    assert _compat.as_index(np.arange(3, dtype=np.intp)).dtype == np.intp


def _process(tmp_path, scene, name):
    p = tmp_path / f"{name}.stl"
    stlio.write_stl(p, scene.mesh.to_triangles(), header="x")
    out = tmp_path / f"{name}_out.stl"
    meta = json.loads(web.process(str(p), str(out), json.dumps({"layer_height": 0.2}), preview_dir=str(tmp_path / f"{name}_pv")))
    return meta, (out.read_bytes() if out.exists() else b"")


SCENES = scenes.all_synthetic(small=True) + scenes.all_holdout(small=True)


@pytest.mark.parametrize("scene", SCENES, ids=[s.name for s in SCENES])
def test_pipeline_gives_the_same_answer_with_32_bit_indices(scene, tmp_path, monkeypatch):
    (tmp_path / "ref").mkdir()
    (tmp_path / "w32").mkdir()
    ref_meta, ref_out = _process(tmp_path / "ref", scene, "m")
    with monkeypatch.context() as m:
        calls = _install(m)
        w_meta, w_out = _process(tmp_path / "w32", scene, "m")
    assert calls["bincount"] > 0, "the strict versions were never used"
    assert w_meta["ok"] and ref_meta["ok"]
    assert w_meta["summary"] == ref_meta["summary"]
    assert w_meta["n_moved"] == ref_meta["n_moved"]
    assert w_out == ref_out


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
