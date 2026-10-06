"""Command line interface: ``stl-smoothing model.stl``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import __version__
from .flatten import FlattenResult, Plateau, flatten
from .layers import LayerGrid
from .mesh import Mesh
from .params import Params
from .slicing import contour_length
from .stlio import StlError, read_stl, write_stl

DEFAULT_WELD_TOL = 5e-5  # mm; below float32 resolution of typical coordinates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stl-smoothing",
        description=(
            "Remove strangely offset layer lines from generated STL models. Surfaces that are meant "
            "to be flat but wobble by a fraction of a millimetre are moved (in z only) onto a single "
            "layer boundary, so the slicer prints each of them on ONE layer."
        ),
        epilog="Heights are measured from the lowest point of the model (where a slicer puts the bed).",
    )
    p.add_argument("input", help="input STL file (binary or ASCII)")
    p.add_argument("-o", "--output", help="output STL (default: <input>_smoothed.stl)")
    g = p.add_argument_group("slicer settings")
    g.add_argument("-l", "--layer-height", type=float, default=0.2, metavar="MM", help="layer height (default 0.2)")
    g.add_argument("--first-layer", type=float, default=None, metavar="MM",
                   help="first layer height (default: same as the layer height)")
    g = p.add_argument_group("what counts as a surface that should be flat")
    g.add_argument("--max-range", type=float, default=None, metavar="MM",
                   help="widest height variation of ONE flat surface (default %g). Anything that varies more "
                        "is an intentional shape. Lower it (e.g. 1.0) to protect gently curved plates and ramps; "
                        "raise it if a very wobbly surface is left alone" % Params.max_range)
    g.add_argument("--max-slope", type=float, default=None, metavar="DEG",
                   help="a surface is 'flat' while it stays within this many degrees of horizontal "
                        "(default %g, measured on smoothed normals so facet noise does not matter)" % Params.max_slope_deg)
    g.add_argument("--min-area", type=float, default=None, metavar="MM2",
                   help="ignore flat surfaces smaller than this (default %g)" % Params.min_area)
    g.add_argument("--no-snap-exact", action="store_true",
                   help="do not nudge already-flat surfaces that sit on a slicer sampling plane onto the layer grid")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override any advanced parameter (see Params in the source); can be repeated")
    g = p.add_argument_group("output")
    g.add_argument("--analyze", action="store_true", help="only report what would change; write no STL")
    g.add_argument("--report", metavar="PNG", help="write a before/after picture of the changed surfaces "
                                                 "(needs matplotlib)")
    g.add_argument("--ascii", action="store_true", help="write an ASCII STL instead of binary")
    g.add_argument("--weld-tol", type=float, default=DEFAULT_WELD_TOL, metavar="MM",
                   help="merge corners closer than this (default %g; 0 = exact matches only)" % DEFAULT_WELD_TOL)
    g.add_argument("-v", "--verbose", action="store_true", help="also list what was found but left alone, and why")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _coerce(name: str, raw: str):
    kinds = {f.name: f.type for f in fields(Params)}
    if name not in kinds:
        raise SystemExit(f"error: --set: unknown parameter {name!r}; known: {', '.join(sorted(kinds))}")
    default = getattr(Params, name)
    if isinstance(default, bool):
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"error: --set {name}: expected true/false, got {raw!r}")
    if raw.lower() in ("none", "null") and default is None:
        return None
    try:
        return type(default)(raw) if default is not None else float(raw)
    except ValueError:
        raise SystemExit(f"error: --set {name}: cannot read {raw!r}") from None


def params_from_args(a: argparse.Namespace) -> Params:
    kw = {}
    if a.max_range is not None:
        kw["max_range"] = a.max_range
    if a.max_slope is not None:
        kw["max_slope_deg"] = a.max_slope
    if a.min_area is not None:
        kw["min_area"] = a.min_area
    if a.no_snap_exact:
        kw["snap_exact_flat"] = False
    for item in a.set:
        if "=" not in item:
            raise SystemExit(f"error: --set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        kw[k.strip()] = _coerce(k.strip(), v.strip())
    return Params().replace(**kw)


# ----------------------------------------------------------------------- output
def _fmt_plateau(p: Plateau) -> str:
    where = "top    " if p.facing == "up" else "ceiling"
    patches = f"{p.regions} patch{'es' if p.regions != 1 else ''}"
    if p.kind == "snapped":
        return f"  {where} {patches:<10} {p.area:9,.0f} mm²  flat at {p.level_before:.3f} mm -> {p.level_after:.2f} mm (on a sampling plane)"
    span = f"{p.z_low:.2f}–{p.z_high:.2f} mm"
    return (
        f"  {where} {patches:<10} {p.area:9,.0f} mm²  {span:<17} {p.layers_before} layer{'s' if p.layers_before != 1 else ' '}"
        f"  ->  {p.level_after:.2f} mm"
    )


def summarize(res: FlattenResult, before: Mesh, after: Mesh, grid: LayerGrid, verbose: bool, out=print) -> None:
    flattened = [p for p in res.plateaus if p.kind == "smoothed" and p.layers_before > 1]
    levelled = [p for p in res.plateaus if p.kind == "smoothed" and p.layers_before <= 1]
    snapped = [p for p in res.plateaus if p.kind == "snapped"]
    if flattened:
        out(f"Flattened {len(flattened)} surface{'s' if len(flattened) != 1 else ''} "
            f"(heights above the bed, {grid.layer_height:g} mm layers):")
        for p in flattened:
            out(_fmt_plateau(p))
        region = np.isin(res.face_plateau, [i for i, p in enumerate(res.plateaus) if p in flattened])
        e0 = contour_length(before, grid, region)
        e1 = contour_length(after, grid, region, after.verts)
        out(f"Layer edges on those surfaces: {e0:,.0f} mm -> {e1:,.0f} mm")
        wide = [p for p in flattened if p.z_high - p.z_low > 1.0]
        if wide:
            out("Note: a flattened surface varied by more than 1 mm. If it is really meant to be curved or "
                "sloped, run again with a smaller --max-range (for example --max-range 1).")
    elif not (levelled or snapped):
        out("No wobbly flat surfaces found; nothing to flatten.")
    if levelled:
        out(f"Levelled {len(levelled)} nearly-flat surface{'s' if len(levelled) != 1 else ''} that grazed a slicer "
            f"sampling plane (they already printed on one layer, with a few stray layer edges):")
        for p in levelled:
            out(_fmt_plateau(p))
    if snapped:
        out(f"Moved {len(snapped)} already-flat surface{'s' if len(snapped) != 1 else ''} off a slicer sampling plane:")
        for p in snapped:
            out(_fmt_plateau(p))
    if res.already_flat and verbose:
        out("Already printing on a single layer (left unchanged):")
        for facing, lvl, area in res.already_flat:
            out(f"  {facing:<7} {area:9,.0f} mm\u00b2  at {lvl:.2f} mm")
    if res.n_moved:
        out(f"Moved {res.n_moved:,} vertices, largest move {res.max_dz:.2f} mm, in z only; "
            f"{res.flipped_faces} flipped faces, {res.degenerate_faces_added} new degenerate faces.")
    if res.skipped:
        shown = res.skipped if verbose else []
        out(f"Left alone: {len(res.skipped)} candidate{'s' if len(res.skipped) != 1 else ''}"
            + ("" if verbose else " (use -v for details)"))
        for line in shown:
            out(f"  {line}")


def run(argv: list[str] | None = None, out=print) -> int:
    a = build_parser().parse_args(argv)
    P = params_from_args(a)
    try:
        grid = LayerGrid(a.layer_height, a.first_layer)
    except ValueError as exc:
        out(f"error: {exc}")
        return 2
    src = Path(a.input)
    if not src.is_file():
        out(f"error: {src}: no such file")
        return 2
    dst = Path(a.output) if a.output else src.with_name(src.stem + "_smoothed.stl")
    if not a.analyze and dst.resolve() == src.resolve():
        out("error: the output would overwrite the input; choose a different -o")
        return 2

    try:
        data = read_stl(src)
    except StlError as exc:
        out(f"error: {exc}")
        return 1
    if len(data.tris) == 0:
        out(f"error: {src}: the file contains no triangles")
        return 1
    mesh = Mesh.from_triangles(data.tris, tol=a.weld_tol if a.weld_tol > 0 else None)
    stats = mesh.edge_manifold_stats()
    lo, hi = mesh.bbox()
    ext = hi - lo
    out(f"{src.name}: {mesh.n_faces:,} triangles, {mesh.n_verts:,} vertices, "
        f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm")
    if not stats["closed"]:
        out(f"warning: the mesh is not closed ({stats['boundary_edges']} open edges, "
            f"{stats['non_manifold_edges']} non-manifold); results may be incomplete")

    res = flatten(mesh, grid, P)
    after = Mesh(res.verts, mesh.faces)
    summarize(res, mesh, after, grid, a.verbose, out)

    if a.report:
        if not (res.smoothed_faces.any() or res.snapped_faces.any()):
            out("No changed surfaces: no report written.")
        else:
            _write_reports(a.report, mesh, after, grid, res, out)

    if a.analyze:
        out("Analyze only: no STL written.")
        return 0
    if not res.n_moved:
        out("Nothing was changed; no STL written.")
        return 0
    write_stl(
        dst,
        after.to_triangles(),
        binary=not a.ascii,
        header=data.header if (data.source_format == "binary" and data.header.strip(b"\x00")) else b"stl-smoothing",
        attrs=data.attrs,
        name=data.name or src.stem,
    )
    out(f"Wrote {dst}")
    return 0


def _write_reports(path: str, before: Mesh, after: Mesh, grid: LayerGrid, res: FlattenResult, out) -> None:
    try:
        from .report import render_comparison
    except Exception as exc:  # pragma: no cover
        out(f"warning: cannot write the report: {exc}")
        return
    # Only surfaces that visibly changed (they used to end on several layers) are highlighted.
    changed = [i for i, p in enumerate(res.plateaus) if p.kind == "smoothed" and p.layers_before > 1]
    if not changed:
        out("No surface changed layers: no report written.")
        return
    region = np.isin(res.face_plateau, changed)
    base = Path(path)
    for facing, view, what in ((1, "top", "Flattened top surfaces"), (-1, "bottom", "Flattened ceilings (seen from below)")):
        mask = region & (res.facing == facing)
        if not mask.any():
            continue
        n_here = len({int(i) for i in res.face_plateau[mask]})
        target = base if facing == 1 or not (region & (res.facing == 1)).any() else base.with_name(base.stem + "_ceilings" + base.suffix)
        try:
            info = render_comparison(before, after, grid, mask, str(target), view=view, what=what)
        except RuntimeError as exc:
            out(f"warning: {exc}")
            return
        out(f"Wrote report {target} ({n_here} surface{'s' if n_here != 1 else ''}: "
            f"{len(info['before_layers'])} layers before, {len(info['after_layers'])} after)")


def main() -> None:  # pragma: no cover - console entry point
    sys.exit(run())
