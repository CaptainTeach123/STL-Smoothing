"""Small numpy wrappers that behave the same on 32-bit platforms.

In the browser (Pyodide / WebAssembly) ``np.intp`` is 32 bits.  A few numpy functions insist on
getting index arrays as ``intp`` and refuse a 64-bit one ("Cannot cast array data from dtype('int64')
to dtype('int32') according to the rule 'safe'"), which is how the first real in-browser run failed.
The wrappers below hand them ``intp`` arrays.  On a normal 64-bit machine they change nothing.

Index arrays here always hold positions in an array of fewer than 2**31 elements, so the cast is exact.
"""

from __future__ import annotations

import numpy as np

# The index dtype handed to numpy.  Tests set it to ``np.int32`` to behave like WebAssembly.
INTP = np.intp


def as_index(a) -> np.ndarray:
    """``a`` as an array of the index dtype (integer and boolean input only; anything else is untouched)."""
    a = np.asarray(a)
    if a.dtype.kind in "iub" and a.dtype != INTP:
        return a.astype(INTP)
    return a


def bincount(x, weights=None, minlength=0) -> np.ndarray:
    """``np.bincount`` that accepts 64-bit integer input on a 32-bit platform."""
    return np.bincount(as_index(x), weights=weights, minlength=minlength)


def reduceat(ufunc, a, starts) -> np.ndarray:
    """``ufunc.reduceat(a, starts)`` that accepts 64-bit ``starts`` on a 32-bit platform."""
    return ufunc.reduceat(a, as_index(starts))
