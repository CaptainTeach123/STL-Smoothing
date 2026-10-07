"""Reading and writing STL files (binary and ASCII).

Triangles are handled as a ``(M, 3, 3)`` float64 array ("triangle soup"):
``tris[i, j]`` is the xyz position of corner ``j`` of triangle ``i``.  STL
stores float32, so values read from a file are exactly representable and
duplicate corners compare bit-for-bit equal, which lets :mod:`mesh` weld them
without any tolerance.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BIN_DTYPE = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


@dataclass
class StlData:
    """Triangle soup read from (or destined for) an STL file."""

    tris: np.ndarray  # (M, 3, 3) float64
    header: bytes = b""  # 80-byte binary header, if the source had one
    attrs: np.ndarray | None = None  # (M,) uint16 attribute bytes (binary only)
    name: str = ""  # solid name (ASCII only)
    source_format: str = "binary"  # "binary" or "ascii"
    extra: dict = field(default_factory=dict)


class StlError(ValueError):
    """The file is not a readable STL."""


def _looks_binary(raw: bytes) -> bool:
    """Binary STL has an exact size relationship with its triangle count.

    The 80-byte header of a binary file may legally start with ``solid``, so
    the size check is the only reliable test.
    """
    if len(raw) < 84:
        return False
    (count,) = struct.unpack_from("<I", raw, 80)
    return len(raw) == 84 + 50 * count


def _looks_ascii(raw: bytes) -> bool:
    """A file that starts with ``solid`` is ASCII only if it also reads like ASCII STL.

    Many binary exporters put ``solid`` at the start of the 80-byte header, so the
    keyword alone proves nothing (a binary file with trailing bytes or a wrong
    triangle count must not be mistaken for an empty ASCII file).
    """
    return b"vertex" in raw[:4096].lower() or b"endsolid" in raw[-256:].lower()


def read_stl(path: str | Path) -> StlData:
    raw = Path(path).read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":  # UTF-8 byte order mark
        raw = raw[3:]
    if _looks_binary(raw):
        return _read_binary(raw)
    head = raw[:512].lstrip()
    if head[:5].lower() == b"solid" and _looks_ascii(raw):
        return _read_ascii(raw)
    if len(raw) >= 84:
        # Some exporters write a wrong triangle count.  Trust the data if the
        # remainder is a whole number of records.
        (count,) = struct.unpack_from("<I", raw, 80)
        body = len(raw) - 84
        if body % 50 == 0 and body // 50 > 0:
            return _read_binary(raw, count=body // 50)
        if count > 0 and body >= 50 * count:
            return _read_binary(raw, count=count)
    raise StlError(f"{path}: not a recognisable binary or ASCII STL file")


def _read_binary(raw: bytes, count: int | None = None) -> StlData:
    if count is None:
        (count,) = struct.unpack_from("<I", raw, 80)
    rec = np.frombuffer(raw, dtype=_BIN_DTYPE, count=count, offset=84)
    tris = rec["v"].astype(np.float64)
    return StlData(
        tris=tris,
        header=bytes(raw[:80]),
        attrs=rec["a"].copy(),
        source_format="binary",
    )


# Vertex lines start a line (so a solid *named* "vertex 1 2 3" cannot be mistaken for one);
# the unanchored pattern is only a fallback for exporters that write the whole file on one line.
_VERTEX_LINE_RE = re.compile(rb"^[ \t]*vertex[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)", re.IGNORECASE | re.MULTILINE)
_VERTEX_RE = re.compile(rb"\bvertex\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)
_SOLID_RE = re.compile(rb"^\s*solid[ \t]*([^\r\n]*)", re.IGNORECASE)


def _read_ascii(raw: bytes) -> StlData:
    # the solid's name (first line) may contain any words: parse the vertices after it
    nl = min([i for i in (raw.find(b"\n", 0, 512), raw.find(b"\r", 0, 512)) if i >= 0], default=-1)
    start = nl + 1 if nl >= 0 else 0
    found = _VERTEX_LINE_RE.findall(raw, start) or _VERTEX_RE.findall(raw, start)
    if not found:
        return StlData(tris=np.zeros((0, 3, 3)), source_format="ascii")
    if len(found) % 3:
        raise StlError("ASCII STL has a vertex count that is not a multiple of 3")
    try:
        coords = np.array(found).astype(np.float64)
    except ValueError as exc:
        raise StlError(f"ASCII STL contains a malformed number: {exc}") from exc
    # Round-trip through float32 so that ASCII and binary inputs weld the same.
    coords = coords.astype(np.float32).astype(np.float64)
    m = _SOLID_RE.match(raw[:512])
    name = m.group(1).decode("utf-8", errors="replace").strip() if m else ""
    return StlData(
        tris=coords.reshape(-1, 3, 3),
        name=name,
        source_format="ascii",
    )


def write_stl(
    path: str | Path,
    tris: np.ndarray,
    *,
    binary: bool = True,
    header: bytes | str | None = None,
    attrs: np.ndarray | None = None,
    name: str = "stl_smoothing",
) -> None:
    """Write ``tris`` (M, 3, 3) to ``path``.  Facet normals are recomputed."""
    tris = np.asarray(tris, dtype=np.float64)
    normals = _facet_normals(tris)
    if binary:
        rec = np.zeros(len(tris), dtype=_BIN_DTYPE)
        rec["n"] = normals.astype(np.float32)
        rec["v"] = tris.astype(np.float32)
        if attrs is not None:
            rec["a"] = attrs
        if header is None:
            header = b""
        if isinstance(header, str):
            header = header.encode("ascii", errors="replace")
        header = bytes(header)[:80].ljust(80, b"\x00")
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(struct.pack("<I", len(tris)))
            fh.write(rec.tobytes())
        return

    lines = [f"solid {name}"]
    t32 = tris.astype(np.float32)
    for tri, n in zip(t32, normals.astype(np.float32)):
        lines.append(f"  facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n")


def _facet_normals(tris: np.ndarray) -> np.ndarray:
    if len(tris) == 0:
        return np.zeros((0, 3))
    c = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(c, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):  # NaN coordinates give NaN normals, never an exception
        return np.divide(c, ln, out=np.zeros_like(c), where=ln > 0)
