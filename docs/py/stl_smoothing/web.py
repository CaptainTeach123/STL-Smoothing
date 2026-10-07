"""Entry point for the in-browser app (Pyodide).

The page puts the chosen STL into the interpreter's file system and calls
:func:`process`.  Results come back as a JSON string (numbers and text) plus binary
files (the smoothed STL and the picture arrays) that the page reads back, so nothing
depends on how a particular Pyodide version converts Python objects.

Nothing here touches the network: the model never leaves the browser.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import __version__
from .flatten import flatten
from .layers import LayerGrid
from .mesh import Mesh, auto_weld_tol
from .params import Params
from .preview import MAX_PREVIEW_FACES, build_preview
from .stlio import StlError, parse_stl, stl_bytes
from .summary import layer_edge_totals, summarize

# triangles above which the page warns that a browser tab may run out of memory
SOFT_LIMIT_TRIANGLES = 1_500_000

_DTYPES = {"float32": "float32", "uint8": "uint8", "uint32": "uint32"}


def _fail(message: str) -> str:
    return json.dumps({"ok": False, "error": message, "version": __version__})


def process(path_in: str, path_out: str, options_json: str = "{}", progress=None, preview_dir: str | None = None) -> str:
    """Smooth the STL at ``path_in``.

    Writes the smoothed STL to ``path_out`` (only if something changed) and, when
    ``preview_dir`` is given, the arrays of the before/after picture into it.
    ``progress`` is an optional callable taking a status string.  Returns a JSON string.
    """
    opts = json.loads(options_json or "{}")

    def tick(msg: str) -> None:
        if progress is not None:
            progress(msg)

    try:
        grid = LayerGrid(float(opts.get("layer_height", 0.2)), opts.get("first_layer"))
        overrides = {}
        for key in ("max_range", "max_slope_deg", "min_area"):
            if opts.get(key) is not None:
                overrides[key] = float(opts[key])
        if "snap_exact" in opts:
            overrides["snap_exact_flat"] = bool(opts["snap_exact"])
        params = Params().replace(**overrides)
    except (TypeError, ValueError) as exc:
        return _fail(str(exc))

    tick("Reading the STL file")
    try:
        raw = Path(path_in).read_bytes()
        data = parse_stl(raw, source="this file")
    except (StlError, OSError) as exc:
        return _fail(str(exc))
    del raw
    if len(data.tris) == 0:
        return _fail("The file contains no triangles.")
    if not np.isfinite(data.tris).all():
        return _fail("The file contains NaN or infinite coordinates.")

    tick("Joining shared corners")
    tol = auto_weld_tol(data.tris)
    mesh = Mesh.from_triangles(data.tris, tol=tol)
    stats = mesh.edge_manifold_stats()
    lo, hi = mesh.bbox()
    meta: dict = {
        "ok": True,
        "version": __version__,
        "source_format": data.source_format,
        "triangles": int(mesh.n_faces),
        "vertices": int(mesh.n_verts),
        "extent": [float(v) for v in (hi - lo)],
        "closed": bool(stats["closed"]),
        "open_edges": int(stats["boundary_edges"]),
        "non_manifold_edges": int(stats["non_manifold_edges"]),
        "layer_height": grid.layer_height,
        "first_layer": grid.first,
        "large": bool(mesh.n_faces > SOFT_LIMIT_TRIANGLES),
    }

    tick("Looking for surfaces that should be flat")
    res = flatten(mesh, grid, params)
    after = Mesh(res.verts, mesh.faces)

    tick("Summarising")
    lines: list[str] = []
    summarize(res, mesh, after, grid, False, lines.append)
    edges = layer_edge_totals(res, mesh, after, grid)
    meta.update(
        summary=lines,
        plateaus=[
            {
                "kind": p.kind, "facing": p.facing, "regions": p.regions, "area": p.area,
                "level_before": p.level_before, "z_low": p.z_low, "z_high": p.z_high,
                "level_after": p.level_after, "layers_before": p.layers_before,
            }
            for p in res.plateaus
        ],
        edges_before=None if edges is None else edges[0],
        edges_after=None if edges is None else edges[1],
        n_moved=int(res.n_moved),
        max_dz=float(res.max_dz),
        flipped=int(res.flipped_faces),
        degenerate_added=int(res.degenerate_faces_added),
        flattened=int(sum(1 for p in res.plateaus if p.kind == "smoothed" and p.layers_before > 1)),
    )

    meta["output_bytes"] = 0
    if res.n_moved:
        tick("Writing the smoothed STL")
        header = data.header if (data.source_format == "binary" and data.header.strip(b"\x00")) else b"stl-smoothing"
        blob = stl_bytes(after.to_triangles(), binary=True, header=header, attrs=data.attrs)
        Path(path_out).write_bytes(blob)
        meta["output_bytes"] = len(blob)

    meta["preview"] = None
    if preview_dir and res.n_moved:
        if mesh.n_faces > MAX_PREVIEW_FACES:
            meta["preview_skipped"] = "The model is too large for the picture; the smoothed STL is still written."
        else:
            tick("Drawing the before/after picture")
            pv_meta, arrays = build_preview(mesh, after, grid, res)
            os.makedirs(preview_dir, exist_ok=True)
            files = {}
            for name, arr in arrays.items():
                fname = os.path.join(preview_dir, name + ".bin")
                Path(fname).write_bytes(np.ascontiguousarray(arr).tobytes())
                files[name] = {"file": fname, "dtype": str(arr.dtype), "length": int(arr.size)}
            pv_meta["files"] = files
            meta["preview"] = pv_meta
    tick("Done")
    return json.dumps(meta)
