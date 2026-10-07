"""Make numpy on a 64-bit machine behave like numpy on WebAssembly (Pyodide), where ``np.intp`` and the
default integer are 32 bits.  This is an approximation of the platform, not the platform:

* functions that return index arrays (``argsort``, ``flatnonzero``, ``searchsorted``, ``unique``'s
  index outputs, ``nonzero``, one-argument ``where``, ``lexsort``, ``cumsum`` of small ints, an integer
  ``arange``) return ``int32``, so later arithmetic follows numpy's real int32 promotion rules;
* ``np.bincount(x)``, ``np.repeat(a, repeats)`` and ``ufunc.reduceat(a, indices)`` receive their index
  argument through a *safe* cast to ``intp`` in numpy's C code, so a 64-bit (or unsigned 32-bit) integer
  array raises ``TypeError`` there, exactly as it does in the browser.

Methods of ndarray (``arr.argsort()``) cannot be patched; the shipped code uses the functions above.
"""

from __future__ import annotations

import numpy as np

from stl_smoothing import _compat

I32 = np.dtype(np.int32)
LIMIT = 2**31 - 1
WRAPPED_UFUNCS = ("minimum", "maximum", "add", "logical_or")


def _intp(a):
    """What a successful ``intp`` result looks like on wasm32."""
    a = np.asarray(a)
    if a.size and (a.max() > LIMIT or a.min() < -LIMIT - 1):
        raise OverflowError("value does not fit in a 32-bit index")
    return a.astype(np.int32, copy=False)


def _strict(name, x):
    a = np.asarray(x)
    if a.dtype.kind == "f" or (a.dtype.kind in "iub" and not np.can_cast(a.dtype, I32, "safe")):
        raise TypeError(f"{name}: Cannot cast array data from dtype('{a.dtype}') to dtype('int32') "
                        "according to the rule 'safe'")
    return a


class _Ufunc:
    """Stands in for a numpy ufunc; ``reduceat`` checks its indices like numpy does on wasm32."""

    def __init__(self, ufunc, calls):
        self._u = ufunc
        self._calls = calls

    def reduceat(self, a, indices, *rest, **kw):
        self._calls["reduceat"] += 1
        return self._u.reduceat(a, _strict("reduceat", indices), *rest, **kw)

    def __call__(self, *a, **k):
        return self._u(*a, **k)

    def __getattr__(self, name):
        return getattr(self._u, name)


def install(monkeypatch) -> dict:
    """Patch numpy for the rest of the ``monkeypatch`` scope; returns call counters."""
    calls = {"bincount": 0, "repeat": 0, "reduceat": 0}
    orig = {n: getattr(np, n) for n in (
        "bincount", "repeat", "argsort", "flatnonzero", "searchsorted", "lexsort", "unique", "arange",
        "cumsum", "nonzero", "where", *WRAPPED_UFUNCS)}

    def bincount(x, *a, **k):
        calls["bincount"] += 1
        r = orig["bincount"](_strict("bincount", x), *a, **k)
        return _intp(r) if r.dtype.kind in "iu" else r

    def repeat(a, repeats, *x, **k):
        if not np.isscalar(repeats):
            calls["repeat"] += 1
            repeats = _strict("repeat", repeats)
        return orig["repeat"](a, repeats, *x, **k)

    def searchsorted(a, v, *x, **k):
        r = orig["searchsorted"](a, v, *x, **k)
        return _intp(r) if isinstance(r, np.ndarray) else r

    def unique(ar, return_index=False, return_inverse=False, return_counts=False, *x, **k):
        r = orig["unique"](ar, return_index, return_inverse, return_counts, *x, **k)
        return (r[0],) + tuple(_intp(v) for v in r[1:]) if isinstance(r, tuple) else r

    def arange(*a, **k):
        r = orig["arange"](*a, **k)
        if "dtype" not in k and r.dtype.kind == "i" and all(isinstance(v, (int, np.integer)) for v in a):
            return _intp(r)
        return r

    def cumsum(a, axis=None, dtype=None, out=None):
        a = np.asarray(a)
        if dtype is None and (a.dtype.kind == "b" or (a.dtype.kind in "iu" and a.dtype.itemsize <= 4)):
            dtype = np.int32 if a.dtype.kind != "u" else np.uint32
        return orig["cumsum"](a, axis=axis, dtype=dtype, out=out)

    def where(c, *x):
        r = orig["where"](c, *x)
        return tuple(_intp(v) for v in r) if len(x) == 0 else r

    monkeypatch.setattr(_compat, "INTP", I32)
    monkeypatch.setattr(np, "bincount", bincount)
    monkeypatch.setattr(np, "repeat", repeat)
    monkeypatch.setattr(np, "argsort", lambda *a, **k: _intp(orig["argsort"](*a, **k)))
    monkeypatch.setattr(np, "flatnonzero", lambda a: _intp(orig["flatnonzero"](a)))
    monkeypatch.setattr(np, "searchsorted", searchsorted)
    monkeypatch.setattr(np, "lexsort", lambda *a, **k: _intp(orig["lexsort"](*a, **k)))
    monkeypatch.setattr(np, "unique", unique)
    monkeypatch.setattr(np, "arange", arange)
    monkeypatch.setattr(np, "cumsum", cumsum)
    monkeypatch.setattr(np, "where", where)
    monkeypatch.setattr(np, "nonzero", lambda a: tuple(_intp(v) for v in orig["nonzero"](a)))
    for name in WRAPPED_UFUNCS:
        monkeypatch.setattr(np, name, _Ufunc(orig[name], calls))
    return calls
